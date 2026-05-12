import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr, spearmanr

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
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Rebuild oracle_NEW OOF
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)

sub_tmpl = pd.read_csv('sample_submission.csv')

inflow_train_raw = train_raw['order_inflow_15m'].values
inflow_test_raw  = test_raw['order_inflow_15m'].values
# Fill NaN with median
train_median = np.nanmedian(inflow_train_raw)
test_median  = np.nanmedian(inflow_test_raw)
inflow_train = np.where(np.isnan(inflow_train_raw), train_median, inflow_train_raw)
inflow_test  = np.where(np.isnan(inflow_test_raw), train_median, inflow_test_raw)
print(f"inflow_train NaN filled: {np.isnan(inflow_train_raw).sum()} filled with median={train_median:.1f}")
print(f"inflow_test NaN filled: {np.isnan(inflow_test_raw).sum()} filled with median={train_median:.1f}")

# ============================================================
# Order inflow distribution: seen vs unseen
# ============================================================
print("="*70)
print("Order inflow distribution: seen vs unseen test")
print("="*70)
print(f"\nTrain inflow: mean={inflow_train.mean():.2f}  std={inflow_train.std():.2f}  p50={np.percentile(inflow_train,50):.1f}  p90={np.percentile(inflow_train,90):.1f}")
print(f"Test seen inflow: mean={inflow_test[seen_mask].mean():.2f}  p50={np.percentile(inflow_test[seen_mask],50):.1f}  p90={np.percentile(inflow_test[seen_mask],90):.1f}")
print(f"Test unseen inflow: mean={inflow_test[unseen_mask].mean():.2f}  p50={np.percentile(inflow_test[unseen_mask],50):.1f}  p90={np.percentile(inflow_test[unseen_mask],90):.1f}")
ratio = inflow_test[unseen_mask].mean() / inflow_test[seen_mask].mean()
print(f"Unseen/seen ratio: {ratio:.3f}x  ({(ratio-1)*100:.1f}% higher)")

# ============================================================
# Oracle_NEW residuals vs order_inflow on training
# ============================================================
print("\n" + "="*70)
print("Oracle_NEW residuals vs order_inflow on training data")
print("="*70)
resid_train = fw4_oo - y_true
r_pearson, _ = pearsonr(inflow_train, resid_train)
r_spearman, _ = spearmanr(inflow_train, resid_train)
print(f"\n  r(inflow, residual) = {r_pearson:.4f} (Pearson)  {r_spearman:.4f} (Spearman)")
print(f"  inflow low (<50 pct) residual: {resid_train[inflow_train < np.percentile(inflow_train, 50)].mean():.3f}")
print(f"  inflow mid (50-80 pct) residual: {resid_train[(inflow_train >= np.percentile(inflow_train, 50)) & (inflow_train < np.percentile(inflow_train, 80))].mean():.3f}")
print(f"  inflow high (>80 pct) residual: {resid_train[inflow_train >= np.percentile(inflow_train, 80)].mean():.3f}")

# Per-inflow-bucket analysis
inflow_bins = [0, 50, 100, 150, 200, 300, 500, 2000]
print(f"\n  {'inflow_bucket':20s} {'n':>7} {'y_mean':>8} {'pred_mean':>10} {'residual':>9}")
for lo, hi in zip(inflow_bins[:-1], inflow_bins[1:]):
    mask = (inflow_train >= lo) & (inflow_train < hi)
    if mask.sum() > 0:
        ym = y_true[mask].mean()
        pm = fw4_oo[mask].mean()
        rm = (pm - ym)
        print(f"  [{lo:4d},{hi:4d}): n={mask.sum():7d}  y={ym:8.3f}  pred={pm:10.3f}  resid={rm:+9.3f}")

# ============================================================
# Test inflow distribution by oracle_NEW prediction level
# ============================================================
print("\n" + "="*70)
print("Unseen test: inflow by oracle_NEW prediction level")
print("="*70)
oN_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
p_oN_u = oracle_new_t[unseen_mask]
inflow_u = inflow_test[unseen_mask]
for lo, hi in zip(oN_bins[:-1], oN_bins[1:]):
    mask = (p_oN_u >= lo) & (p_oN_u < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  oN={p_oN_u[mask].mean():.2f}  inflow_mean={inflow_u[mask].mean():.1f}  inflow_p50={np.percentile(inflow_u[mask],50):.1f}  inflow_p90={np.percentile(inflow_u[mask],90):.1f}")

# ============================================================
# Feature-based correction: does inflow predict correction?
# ============================================================
print("\n" + "="*70)
print("Feature-based correction: inflow → correction model")
print("="*70)
from sklearn.linear_model import Ridge, LinearRegression
import warnings; warnings.filterwarnings('ignore')

# Build correction model on training data
X_train = inflow_train.reshape(-1, 1)
y_correction = resid_train  # want to predict the residual (negative = underpredict)
lr = LinearRegression().fit(X_train, y_correction)
print(f"\n  Linear: resid = {lr.coef_[0]:.6f} * inflow + {lr.intercept_:.3f}")
print(f"  R² = {lr.score(X_train, y_correction):.4f}")
print(f"  Correction at inflow=100: {lr.predict([[100]])[0]:+.3f}")
print(f"  Correction at inflow=150: {lr.predict([[150]])[0]:+.3f}")
print(f"  Correction at inflow=200: {lr.predict([[200]])[0]:+.3f}")

# Apply inflow-based correction to unseen test
correction_unseen = lr.predict(inflow_test[unseen_mask].reshape(-1, 1))
print(f"\n  Inflow correction for unseen test:")
print(f"    mean={correction_unseen.mean():.3f}  std={correction_unseen.std():.3f}")
print(f"    pct10={np.percentile(correction_unseen,10):.3f}  pct50={np.percentile(correction_unseen,50):.3f}  pct90={np.percentile(correction_unseen,90):.3f}")

# Apply correction (subtract the predicted underprediction)
ct = oracle_new_t.copy()
ct[unseen_mask] = ct[unseen_mask] - correction_unseen  # subtract negative residual = add correction
ct = np.clip(ct, 0, None)
du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  After inflow-based correction: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# Save inflow-corrected candidate
fname = "FINAL_NEW_oN_inflowCorr_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# ============================================================
# Multi-feature correction model
# ============================================================
print("\n" + "="*70)
print("Multi-feature correction model (top features)")
print("="*70)
feature_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'task_reassign_15m', 'charge_queue_length', 'max_zone_density',
                'avg_trip_distance', 'blocked_path_15m', 'urgent_order_ratio']
X_train_multi_raw = train_raw[feature_cols].values.copy()
X_test_unseen_raw = test_raw.loc[unseen_mask, feature_cols].values.copy()
# Fill NaN with column medians
for j in range(X_train_multi_raw.shape[1]):
    col_median = np.nanmedian(X_train_multi_raw[:, j])
    X_train_multi_raw[np.isnan(X_train_multi_raw[:, j]), j] = col_median
    X_test_unseen_raw[np.isnan(X_test_unseen_raw[:, j]), j] = col_median
X_train_multi = X_train_multi_raw
X_test_unseen = X_test_unseen_raw

lr_multi = Ridge(alpha=1.0).fit(X_train_multi, y_correction)
corr_multi = lr_multi.predict(X_test_unseen)
print(f"  Multi-feature correction: mean={corr_multi.mean():.3f}  std={corr_multi.std():.3f}")
print(f"  R² on training: {lr_multi.score(X_train_multi, y_correction):.4f}")

ct2 = oracle_new_t.copy()
ct2[unseen_mask] = ct2[unseen_mask] - corr_multi
ct2 = np.clip(ct2, 0, None)
du2 = ct2[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  After multi-feature correction: seen={ct2[seen_mask].mean():.3f}  unseen={ct2[unseen_mask].mean():.3f}  Δ={du2:+.3f}")

fname2 = "FINAL_NEW_oN_multiCorr_OOF8.3825.csv"
sub2 = sub_tmpl.copy(); sub2['avg_delay_minutes_next_30m'] = ct2
sub2.to_csv(fname2, index=False)
print(f"  Saved: {fname2}")

# ============================================================
# Inflow-weighted delta (use inflow to scale the correction)
# ============================================================
print("\n" + "="*70)
print("Inflow-scaled correction: scale w30mae weight by relative inflow")
print("="*70)
w30mae_t = np.clip(np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2], 0, None)

# Normalize inflow relative to training mean
inflow_norm = inflow_test[unseen_mask] / inflow_train.mean()
print(f"  Inflow normalization for unseen: mean={inflow_norm.mean():.3f}  min={inflow_norm.min():.3f}  max={inflow_norm.max():.3f}")

# Scale the correction by normalized inflow: higher inflow → more correction
# correction_scale = clip(inflow_norm - 1, 0, 1) → proportional to excess inflow above training mean
correction_scale = np.clip(inflow_norm - 1.0, 0, 2.0) / 2.0  # normalized to [0,1]
print(f"  Scale distribution: mean={correction_scale.mean():.3f}  pct50={np.percentile(correction_scale,50):.3f}  pct90={np.percentile(correction_scale,90):.3f}")

# Apply: w_effective = base_w * (1 + alpha * inflow_scale)
for base_w, alpha in [(0.10, 1.0), (0.10, 2.0), (0.20, 1.0)]:
    w_eff = base_w * (1 + alpha * correction_scale)
    ct3 = oracle_new_t.copy()
    ct3[unseen_mask] = (1 - w_eff) * oracle_new_t[unseen_mask] + w_eff * w30mae_t[unseen_mask]
    ct3 = np.clip(ct3, 0, None)
    du3 = ct3[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  base_w={base_w} alpha={alpha}: seen={ct3[seen_mask].mean():.3f}  unseen={ct3[unseen_mask].mean():.3f}  Δ={du3:+.3f}")

print("\nDone.")
