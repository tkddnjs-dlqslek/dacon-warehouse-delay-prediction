import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo
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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Residual Extrapolation: Train regularized model to predict residuals")
print("Key: model must generalize to OOD test territory (unseen layouts)")
print("="*70)

# ============================================================
# Feature set: use continuous features likely to extrapolate well
# Exclude: layout_id, scenario_id, timeslot (categorical)
# Include: physical/operational features
# ============================================================
feat_cols_base = [c for c in train_raw.columns
                  if c not in ('ID','_row_id','layout_id','scenario_id',
                               'avg_delay_minutes_next_30m','timeslot')]

# Fill NaN
train_feat = train_raw[feat_cols_base].fillna(train_raw[feat_cols_base].median())
test_feat  = test_raw[feat_cols_base].fillna(train_raw[feat_cols_base].median())

X_tr = train_feat.values
X_te = test_feat.values

print(f"\n  Features: {len(feat_cols_base)}")
print(f"  Training rows: {len(X_tr)}")
print(f"  Test rows: {len(X_te)}")

# ============================================================
# Cross-validation for residual prediction via GroupKFold
# Use layout_id as group → ensures OOF residuals are layout-independent
# ============================================================
print("\n--- GroupKFold CV for residual model ---")
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

# Test different Ridge alpha values
alphas = [10, 100, 1000, 10000]
best_mae = float('inf')
best_alpha = None
oof_resid_preds = {}

for alpha in alphas:
    resid_oof = np.zeros(len(residuals_train))
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(X_tr, residuals_train, groups)):
        sc = StandardScaler()
        X_tr_sc = sc.fit_transform(X_tr[tr_idx])
        X_va_sc = sc.transform(X_tr[va_idx])
        rdg = Ridge(alpha=alpha)
        rdg.fit(X_tr_sc, residuals_train[tr_idx])
        resid_oof[va_idx] = rdg.predict(X_va_sc)
    corrected_oof = np.clip(fw4_oo + resid_oof, 0, None)
    mae = np.mean(np.abs(corrected_oof - y_true))
    orig_mae = np.mean(np.abs(fw4_oo - y_true))
    improvement = orig_mae - mae
    print(f"  alpha={alpha:7d}: MAE={mae:.4f}  improvement={improvement:+.4f}  (mean_resid_pred={resid_oof.mean():+.4f})")
    oof_resid_preds[alpha] = resid_oof
    if mae < best_mae:
        best_mae = mae
        best_alpha = alpha

print(f"\n  Best alpha: {best_alpha}")

# ============================================================
# Train on ALL training data with best alpha, predict test
# ============================================================
print(f"\n--- Final model with alpha={best_alpha} ---")
sc_final = StandardScaler()
X_tr_final = sc_final.fit_transform(X_tr)
X_te_final = sc_final.transform(X_te)
rdg_final = Ridge(alpha=best_alpha)
rdg_final.fit(X_tr_final, residuals_train)
resid_pred_test = rdg_final.predict(X_te_final)

print(f"  Test residual predictions:")
print(f"    All:    mean={resid_pred_test.mean():+.4f}  std={resid_pred_test.std():.4f}")
print(f"    Seen:   mean={resid_pred_test[seen_mask]:+.4f}  std={resid_pred_test[seen_mask].std():.4f}" if False else
      f"    Seen:   mean={resid_pred_test[seen_mask].mean():+.4f}  std={resid_pred_test[seen_mask].std():.4f}")
print(f"    Unseen: mean={resid_pred_test[unseen_mask].mean():+.4f}  std={resid_pred_test[unseen_mask].std():.4f}")

# Apply only to unseen test (unseen-only correction)
ct_resid = oracle_new_t.copy()
ct_resid[unseen_mask] = oracle_new_t[unseen_mask] + resid_pred_test[unseen_mask]
ct_resid = np.clip(ct_resid, 0, None)
du_resid = ct_resid[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Unseen-only residual correction: Δ={du_resid:+.4f}  seen={ct_resid[seen_mask].mean():.3f}  unseen={ct_resid[unseen_mask].mean():.3f}")

# Save
fname_r = f"FINAL_NEW_oN_residRidge_{best_alpha}_OOF8.3825.csv"
sub_r = sub_tmpl.copy(); sub_r['avg_delay_minutes_next_30m'] = ct_resid
sub_r.to_csv(fname_r, index=False)
print(f"  Saved: {fname_r}")

# Scaled versions
for scale, label in [(0.50, 'half'), (0.75, '3qtr')]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * resid_pred_test[unseen_mask]
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname_s = f"FINAL_NEW_oN_residRidge_{label}_OOF8.3825.csv"
    sub_s = sub_tmpl.copy(); sub_s['avg_delay_minutes_next_30m'] = ct_s
    sub_s.to_csv(fname_s, index=False)
    print(f"  {label}: Δ={du_s:+.4f}  unseen={ct_s[unseen_mask].mean():.3f}  Saved: {fname_s}")

# ============================================================
# Top features most important for residual prediction
# ============================================================
print(f"\n--- Top residual-predictive features (Ridge coefficients at alpha={best_alpha}) ---")
coef = rdg_final.coef_
coef_sorted = sorted(zip(feat_cols_base, coef), key=lambda x: abs(x[1]), reverse=True)[:15]
print(f"  {'feature':40s}  {'coef':>10}")
for f, c in coef_sorted:
    print(f"  {f:40s}  {c:+10.4f}")

# ============================================================
# Check: are the top features OOD for unseen test?
# ============================================================
print(f"\n--- OOD check for top-10 residual features ---")
print(f"  {'feature':40s}  {'train_mean':>10}  {'train_std':>10}  {'unseen_mean':>12}  {'sigma_OOD':>10}")
for f, c in coef_sorted[:10]:
    tr_v = train_feat[f].values
    te_u = test_feat.loc[unseen_mask, f].values
    tr_m, tr_s = tr_v.mean(), tr_v.std()
    u_m = te_u.mean()
    sigma_ood = abs(u_m - tr_m) / (tr_s + 1e-9)
    print(f"  {f:40s}  {tr_m:10.4f}  {tr_s:10.4f}  {u_m:12.4f}  {sigma_ood:10.3f}σ")

print("\nDone.")
