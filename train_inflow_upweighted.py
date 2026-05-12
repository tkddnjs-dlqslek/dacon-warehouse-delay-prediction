import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
import lightgbm as lgb
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
id_order = test_raw['ID'].values

# Oracle NEW test predictions
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Inflow-Upweighted LightGBM for High-Inflow Correction")
print("="*70)

# Features: exclude target, ID, timeslot-leakage risks
inflow_col = 'order_inflow_15m'
feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot')]

print(f"\n  Features: {len(feat_cols)}")

# Fill NaN
train_feat = train_raw[feat_cols].copy()
test_feat  = test_raw[feat_cols].copy()
for c in feat_cols:
    med = train_feat[c].median()
    train_feat[c] = train_feat[c].fillna(med)
    test_feat[c] = test_feat[c].fillna(med)

X_tr = train_feat.values.astype(np.float32)
X_te = test_feat.values.astype(np.float32)
y_tr = y_true

# Compute inflow upweighting: weight = (inflow / mean_inflow) ^ 2
inflow_vals = train_feat[inflow_col].values
mean_inflow = inflow_vals.mean()
alpha = 2.0
weights = (inflow_vals / mean_inflow) ** alpha
weights = weights / weights.mean()  # normalize so mean weight = 1

print(f"\n  Inflow: mean={mean_inflow:.2f}  max={inflow_vals.max():.2f}")
print(f"  Weight: mean={weights.mean():.4f}  max={weights.max():.4f}  pct_above_2x: {100*(weights>2).mean():.1f}%")
print(f"  Target inflow (test unseen): ~167  weight would be: {(167/mean_inflow)**2:.2f}x")

# LightGBM parameters (conservative to avoid overfit)
params = {
    'objective': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 5.0,
    'reg_lambda': 5.0,
    'verbose': -1,
    'n_jobs': -1,
}

# GroupKFold CV
print(f"\n--- 5-fold GroupKFold (layout_id groups) ---")
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
oof_wt = np.zeros(len(y_tr))

for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_tr, groups)):
    ds_tr = lgb.Dataset(X_tr[tr_idx], y_tr[tr_idx], weight=weights[tr_idx], free_raw_data=False)
    ds_va = lgb.Dataset(X_tr[va_idx], y_tr[va_idx], free_raw_data=False)
    mdl = lgb.train(params, ds_tr, num_boost_round=300, valid_sets=[ds_va],
                    callbacks=[lgb.log_evaluation(-1)])
    oof_wt[va_idx] = mdl.predict(X_tr[va_idx])

oof_wt = np.clip(oof_wt, 0, None)
mae_wt = np.mean(np.abs(oof_wt - y_tr))
r_wt, _ = pearsonr(oof_wt, y_tr)
print(f"  Weighted LGB OOF: MAE={mae_wt:.4f}  r={r_wt:.4f}")

# Also train uniform-weight LGB for comparison
print(f"\n--- Standard LGB (no upweighting) ---")
oof_std = np.zeros(len(y_tr))
for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, y_tr, groups)):
    ds_tr = lgb.Dataset(X_tr[tr_idx], y_tr[tr_idx], free_raw_data=False)
    ds_va = lgb.Dataset(X_tr[va_idx], y_tr[va_idx], free_raw_data=False)
    mdl = lgb.train(params, ds_tr, num_boost_round=300, valid_sets=[ds_va],
                    callbacks=[lgb.log_evaluation(-1)])
    oof_std[va_idx] = mdl.predict(X_tr[va_idx])

oof_std = np.clip(oof_std, 0, None)
mae_std = np.mean(np.abs(oof_std - y_tr))
r_std, _ = pearsonr(oof_std, y_tr)
print(f"  Standard LGB OOF: MAE={mae_std:.4f}  r={r_std:.4f}")

# Train final model on all data for test
print(f"\n--- Final models on all training data ---")
ds_all = lgb.Dataset(X_tr, y_tr, weight=weights, free_raw_data=False)
mdl_wt_final = lgb.train(params, ds_all, num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
te_wt = np.clip(mdl_wt_final.predict(X_te), 0, None)

ds_all_std = lgb.Dataset(X_tr, y_tr, free_raw_data=False)
mdl_std_final = lgb.train(params, ds_all_std, num_boost_round=300, callbacks=[lgb.log_evaluation(-1)])
te_std = np.clip(mdl_std_final.predict(X_te), 0, None)

print(f"\n  Weighted LGB test predictions:")
print(f"    seen mean={te_wt[seen_mask].mean():.3f}  unseen mean={te_wt[unseen_mask].mean():.3f}")
print(f"    r(wt_test, oracle_NEW) = {np.corrcoef(te_wt, oracle_new_t)[0,1]:.4f}")
print(f"  Standard LGB test predictions:")
print(f"    seen mean={te_std[seen_mask].mean():.3f}  unseen mean={te_std[unseen_mask].mean():.3f}")
print(f"    r(std_test, oracle_NEW) = {np.corrcoef(te_std, oracle_new_t)[0,1]:.4f}")

# Correlation between weighted and standard for OOF
print(f"\n  r(wt_oof, std_oof) = {np.corrcoef(oof_wt, oof_std)[0,1]:.4f}")

# Compare OOF residuals
print(f"\n  OOF residuals by training inflow bucket:")
inflow_tr_vals = train_feat[inflow_col].values
bins_in = [0, 50, 75, 100, 125, 150, 500]
for lo, hi in zip(bins_in[:-1], bins_in[1:]):
    m = (inflow_tr_vals >= lo) & (inflow_tr_vals < hi)
    if m.sum() > 500:
        r_wt_b = (oof_wt[m] - y_tr[m]).mean()
        r_std_b = (oof_std[m] - y_tr[m]).mean()
        print(f"  [{lo:3d},{hi:3d}): n={m.sum():7d}  wt_resid={r_wt_b:+.4f}  std_resid={r_std_b:+.4f}")

# Compare test predictions for unseen by inflow bucket
inflow_te_vals = test_feat[inflow_col].values
print(f"\n  Test UNSEEN predictions by inflow bucket:")
for lo, hi in zip(bins_in[:-1], bins_in[1:]):
    m = (inflow_te_vals[unseen_mask] >= lo) & (inflow_te_vals[unseen_mask] < hi)
    if m.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={m.sum():5d}  oN={oracle_new_t[unseen_mask][m].mean():.3f}  wt={te_wt[unseen_mask][m].mean():.3f}  std={te_std[unseen_mask][m].mean():.3f}")

# Save blend candidates: oracle_NEW + w * weighted_LGB for unseen only
print(f"\n--- Blend candidates: oracle_NEW + w*wt_lgb (unseen only) ---")
for w in [0.1, 0.2, 0.3, 0.5]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*te_wt[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w={w:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.4f}")
    fname = f"FINAL_NEW_oN_wtLGB_u{int(w*100):02d}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)

print("\nDone.")
