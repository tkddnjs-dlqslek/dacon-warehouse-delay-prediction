import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo (oracle_NEW OOF)
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
residuals_train = y_true - fw4_oo

print("="*70)
print("Layout-level Meta-Regression: residual ~ layout_features")
print("="*70)

# --- Extract layout-level features from train ---
feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot')]

# Compute per-layout mean of each feature
train_raw['_resid'] = residuals_train
layout_grp = train_raw.groupby('layout_id')
layout_stats = layout_grp[feat_cols].mean()
layout_resid = layout_grp['_resid'].mean()
layout_y     = layout_grp['avg_delay_minutes_next_30m'].mean()

print(f"  Training layouts: {len(layout_stats)}")
print(f"  Features: {len(feat_cols)}")

# Fill NaN in features
layout_stats = layout_stats.fillna(layout_stats.median())

# --- Key feature: order_inflow_15m ---
inflow_col = 'order_inflow_15m' if 'order_inflow_15m' in feat_cols else None
if inflow_col:
    r_inflow, p_inflow = pearsonr(layout_stats[inflow_col].values, layout_resid.values)
    print(f"\n  r(mean_inflow, layout_resid) = {r_inflow:.4f}  (p={p_inflow:.4f})")

# --- Top correlated features ---
print("\n  Top-10 features most correlated with layout residual:")
corrs = {}
for c in feat_cols:
    try:
        r, p = pearsonr(layout_stats[c].fillna(0).values, layout_resid.values)
        if not np.isnan(r):
            corrs[c] = (r, p)
    except:
        pass
corrs_sorted = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)[:10]
for c, (r, p) in corrs_sorted:
    print(f"    {c:45s}  r={r:+.4f}  p={p:.4f}")

# --- Build ridge regression: residual ~ top features ---
print("\n" + "="*70)
print("Ridge regression: layout_resid ~ layout_features")
print("="*70)

# Use top 20 correlated features
top_feats = [c for c, _ in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)[:20]]
X_train_layout = layout_stats[top_feats].values
y_resid_layout = layout_resid.values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_layout)

ridge = Ridge(alpha=10.0)
ridge.fit(X_scaled, y_resid_layout)
pred_resid_train = ridge.predict(X_scaled)
ss_tot = np.sum((y_resid_layout - y_resid_layout.mean())**2)
ss_res = np.sum((y_resid_layout - pred_resid_train)**2)
r2 = 1 - ss_res/ss_tot
print(f"  Train R² = {r2:.4f}")
print(f"  Train MAE of residual prediction = {np.mean(np.abs(pred_resid_train - y_resid_layout)):.4f}")

# --- Apply to test unseen layouts ---
print("\n" + "="*70)
print("Extrapolate to unseen test layouts")
print("="*70)
unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()
seen_test_layouts   = test_raw[seen_mask]['layout_id'].unique()
print(f"  Unseen test layouts: {len(unseen_test_layouts)}")
print(f"  Seen test layouts:   {len(seen_test_layouts)}")

test_layout_stats = test_raw.groupby('layout_id')[top_feats].mean().fillna(
    test_raw[top_feats].median())

# Show inflow for unseen vs seen test
if inflow_col in top_feats:
    unseen_inflow = test_layout_stats.loc[unseen_test_layouts, inflow_col].mean()
    seen_inflow   = test_layout_stats.loc[seen_test_layouts, inflow_col].mean()
    train_inflow  = layout_stats[inflow_col].mean()
    print(f"\n  mean_inflow: train={train_inflow:.1f}  seen_test={seen_inflow:.1f}  unseen_test={unseen_inflow:.1f}")
    print(f"  unseen/train ratio: {unseen_inflow/train_inflow:.3f}x")

# Predict expected residual for each test layout
X_test_layout_u = test_layout_stats.loc[unseen_test_layouts, top_feats].values
X_test_layout_s = test_layout_stats.loc[seen_test_layouts, top_feats].values
X_test_u_scaled = scaler.transform(X_test_layout_u)
X_test_s_scaled = scaler.transform(X_test_layout_s)

pred_resid_unseen = ridge.predict(X_test_u_scaled)
pred_resid_seen   = ridge.predict(X_test_s_scaled)

print(f"\n  Predicted layout residual — SEEN test layouts:")
print(f"    mean={pred_resid_seen.mean():.4f}  std={pred_resid_seen.std():.4f}")
print(f"  Predicted layout residual — UNSEEN test layouts:")
print(f"    mean={pred_resid_unseen.mean():.4f}  std={pred_resid_unseen.std():.4f}")
print(f"  Implied row-level correction for unseen: Δ ≈ {pred_resid_unseen.mean():.4f}")

# --- Per-layout correction on test ---
print("\n" + "="*70)
print("Apply layout-predicted correction to oracle_NEW unseen test rows")
print("="*70)
# Map each test row to its layout's predicted correction
layout_to_corr = dict(zip(unseen_test_layouts, pred_resid_unseen))
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values

# Row-level correction based on layout prediction
ct_ridge = oracle_new_t.copy()
for lid, corr in layout_to_corr.items():
    row_mask = test_raw['layout_id'] == lid
    ct_ridge[row_mask.values] -= corr  # subtract because corr is residual (y - pred), so correction is -corr... wait
    # residual = y - pred → correction = -residual (add -residual to predictions to reduce bias)
    # Actually: we want pred_corrected = pred - residual (if residual is negative, add |residual|)
    # WAIT: residual = y - pred (positive means underprediction → pred too low)
    # To correct: new_pred = pred + |residual| = pred - residual (when residual < 0)
    # Actually new_pred = pred - residual is WRONG. Let's be clear:
    # residual = y_true - pred → to reduce mean error: new_pred = pred + residual_hat = pred - resid... NO
    # CORRECT: new_pred = pred + correction_amount
    # If residual = y - pred = -3 (pred too high by 3), then correction = -3 → new_pred = pred + (-3) = pred - 3? No, that makes it worse
    # CORRECT definition: residual = y - pred, so y = pred + residual
    # To match y: new_pred = pred + residual_hat
    # So correction_amount = residual_hat (which is negative for underprediction)
    # But training residual = -3.175 means y > pred by 3.175 (underprediction)
    # So correction = +3.175 → new_pred = pred + 3.175 (upward shift)
    # layout_resid = y - fw4_oo → positive = underprediction → add to pred
    # We already applied -corr above which is WRONG. Fix below.

# Redo correctly
ct_ridge = oracle_new_t.copy()
for lid, corr in layout_to_corr.items():
    row_mask = (test_raw['layout_id'] == lid).values
    ct_ridge[row_mask] = oracle_new_t[row_mask] + corr  # add predicted residual

ct_ridge = np.clip(ct_ridge, 0, None)
du_ridge = ct_ridge[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Row-level Ridge correction (unseen): Δunseen = {du_ridge:+.4f}")
print(f"  seen={ct_ridge[seen_mask].mean():.3f}  unseen={ct_ridge[unseen_mask].mean():.3f}")

# Save
sub_tmpl = pd.read_csv('sample_submission.csv')
fname = f"FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_ridge
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# --- Compare with simpler feature-based approaches ---
print("\n" + "="*70)
print("Single-feature linear: residual ~ mean_inflow")
print("="*70)
if inflow_col:
    x_in = layout_stats[inflow_col].values.reshape(-1,1)
    y_res = layout_resid.values
    lr = LinearRegression().fit(x_in, y_res)
    print(f"  coef={lr.coef_[0]:.4f}  intercept={lr.intercept_:.4f}")
    r2_lr = lr.score(x_in, y_res)
    print(f"  R²={r2_lr:.4f}")

    # Predict for each unseen test layout
    x_test_u_in = test_layout_stats.loc[unseen_test_layouts, inflow_col].values.reshape(-1,1)
    pred_lr = lr.predict(x_test_u_in)
    print(f"  Predicted residual for unseen test layouts: mean={pred_lr.mean():.4f}  std={pred_lr.std():.4f}")

    ct_lr = oracle_new_t.copy()
    layout_to_lr = dict(zip(unseen_test_layouts, pred_lr))
    for lid, corr in layout_to_lr.items():
        row_mask = (test_raw['layout_id'] == lid).values
        ct_lr[row_mask] = oracle_new_t[row_mask] + corr
    ct_lr = np.clip(ct_lr, 0, None)
    du_lr = ct_lr[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"\n  LR single-feature (inflow) correction: Δunseen={du_lr:+.4f}")
    print(f"  seen={ct_lr[seen_mask].mean():.3f}  unseen={ct_lr[unseen_mask].mean():.3f}")
    fname2 = "FINAL_NEW_oN_lrInflowCorr_OOF8.3825.csv"
    sub2 = sub_tmpl.copy(); sub2['avg_delay_minutes_next_30m'] = ct_lr
    sub2.to_csv(fname2, index=False)
    print(f"  Saved: {fname2}")

# --- Distribution check: per training layout y_mean vs inflow ---
print("\n" + "="*70)
print("Training: y_mean per layout vs inflow — extrapolation to inflow=161")
print("="*70)
if inflow_col:
    x_in_tr = layout_stats[inflow_col].values.reshape(-1,1)
    y_mean_tr = layout_y.values
    lr2 = LinearRegression().fit(x_in_tr, y_mean_tr)
    print(f"  y_mean ~ inflow: coef={lr2.coef_[0]:.4f}  intercept={lr2.intercept_:.4f}")
    r2_2 = lr2.score(x_in_tr, y_mean_tr)
    print(f"  R²={r2_2:.4f}")

    # Extrapolate
    for inflow_val in [101, 120, 140, 161, 180]:
        y_pred_at = lr2.predict([[inflow_val]])[0]
        print(f"  inflow={inflow_val}: predicted y_mean={y_pred_at:.3f}")

    # What oracle_NEW predicts for unseen vs what linear extrapolation says y should be
    oracle_unseen_mean = oracle_new_t[unseen_mask].mean()
    y_pred_161 = lr2.predict([[161]])[0]
    implied_delta = y_pred_161 - oracle_unseen_mean
    print(f"\n  oracle_NEW unseen mean: {oracle_unseen_mean:.3f}")
    print(f"  Linear extrapolation y at inflow=161: {y_pred_161:.3f}")
    print(f"  Implied Δ = {implied_delta:+.3f}")

print("\nDone.")
