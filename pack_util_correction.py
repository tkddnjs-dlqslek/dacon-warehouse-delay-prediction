import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

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

# --- Row-level pack_utilization analysis ---
pu_train = train_raw['pack_utilization'].fillna(train_raw['pack_utilization'].median()).values
pu_test  = test_raw['pack_utilization'].fillna(train_raw['pack_utilization'].median()).values
print("="*70)
print("Row-level pack_utilization: residual buckets")
print("="*70)
bins = np.linspace(0, 1, 11)
print(f"\n  {'bin':15s}  {'n_train':>8}  {'mean_resid':>10}  {'n_unseen_test':>13}  {'% unseen_test':>13}")
for lo, hi in zip(bins[:-1], bins[1:]):
    tr_mask = (pu_train >= lo) & (pu_train < hi)
    te_mask = (pu_test[unseen_mask] >= lo) & (pu_test[unseen_mask] < hi)
    if tr_mask.sum() == 0:
        continue
    mr = residuals_train[tr_mask].mean()
    pct = 100*te_mask.sum()/unseen_mask.sum()
    print(f"  [{lo:.1f},{hi:.1f}): {tr_mask.sum():8d}  {mr:10.3f}  {te_mask.sum():13d}  {pct:13.1f}%")

pu_train_max = pu_train.max()
pu_test_u = pu_test[unseen_mask]
print(f"\n  Training max pack_util: {pu_train_max:.4f}")
print(f"  Unseen test OOD (>train max): {(pu_test_u > pu_train_max).sum()} rows / {unseen_mask.sum()} ({100*(pu_test_u > pu_train_max).mean():.1f}%)")

# --- Linear regression at ROW level: residual ~ pack_utilization ---
print("\n" + "="*70)
print("Row-level regression: residual ~ pack_utilization (training data)")
print("="*70)
r_pu, p_pu = pearsonr(pu_train, residuals_train)
print(f"  r(pack_util, residual) = {r_pu:.4f}  (p={p_pu:.4e})")

lr_pu = LinearRegression().fit(pu_train.reshape(-1,1), residuals_train)
print(f"  coef={lr_pu.coef_[0]:.4f}  intercept={lr_pu.intercept_:.4f}")
print(f"  R²={lr_pu.score(pu_train.reshape(-1,1), residuals_train):.4f}")

# --- Apply row-level correction ---
# Predicted correction for each test unseen row
pu_test_unseen = pu_test[unseen_mask]
pred_corr_rowlevel = lr_pu.predict(pu_test_unseen.reshape(-1,1))
print(f"\n  Row-level correction for unseen (row): mean={pred_corr_rowlevel.mean():.4f}  std={pred_corr_rowlevel.std():.4f}")

ct_rowlr = oracle_new_t.copy()
ct_rowlr[unseen_mask] = oracle_new_t[unseen_mask] + pred_corr_rowlevel
ct_rowlr = np.clip(ct_rowlr, 0, None)
du_rowlr = ct_rowlr[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  seen={ct_rowlr[seen_mask].mean():.3f}  unseen={ct_rowlr[unseen_mask].mean():.3f}  Δ={du_rowlr:+.4f}")
fname_rl = "FINAL_NEW_oN_rowLR_packutil_OOF8.3825.csv"
sub_rl = sub_tmpl.copy(); sub_rl['avg_delay_minutes_next_30m'] = ct_rowlr
sub_rl.to_csv(fname_rl, index=False)
print(f"  Saved: {fname_rl}")

# --- Scaled versions: 50%, 100%, 150% of row-level correction ---
print("\n" + "="*70)
print("Scaled row-level pack_util correction")
print("="*70)
for scale in [0.25, 0.50, 0.75, 1.00, 1.50, 2.00]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * pred_corr_rowlevel
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  scale={scale:.2f}: seen={ct_s[seen_mask].mean():.3f}  unseen={ct_s[unseen_mask].mean():.3f}  Δ={du_s:+.4f}")

# --- Threshold correction: different corrections for low vs high pack_util ---
print("\n" + "="*70)
print("Threshold correction: low (<0.60) vs high (>=0.60) pack_util unseen rows")
print("="*70)
# From quintile analysis: high quintile (0.60-0.85) has mean residual = 8.62
# Low quintile (0-0.60) has mean residual ~ 1.6-2.6
lo_mask_tr = pu_train < 0.60
hi_mask_tr = pu_train >= 0.60
mr_lo = residuals_train[lo_mask_tr].mean()
mr_hi = residuals_train[hi_mask_tr].mean()
print(f"  Training: pack_util < 0.60: n={lo_mask_tr.sum()} → mean_resid={mr_lo:.4f}")
print(f"  Training: pack_util >= 0.60: n={hi_mask_tr.sum()} → mean_resid={mr_hi:.4f}")

lo_mask_tu = pu_test[unseen_mask] < 0.60
hi_mask_tu = pu_test[unseen_mask] >= 0.60
print(f"\n  Unseen test: pack_util < 0.60: {lo_mask_tu.sum()} rows ({100*lo_mask_tu.mean():.1f}%)")
print(f"  Unseen test: pack_util >= 0.60: {hi_mask_tu.sum()} rows ({100*hi_mask_tu.mean():.1f}%)")

for (corr_lo, corr_hi), label in [
    ((0, mr_hi), 'hi_full'),
    ((mr_lo, mr_hi), 'both_full'),
    ((mr_lo/2, mr_hi/2), 'both_half'),
    ((mr_lo/4, mr_hi/4), 'both_quarter'),
    ((0, mr_hi/2), 'hi_half'),
    ((0, mr_hi/4), 'hi_quarter'),
]:
    ct_thr = oracle_new_t.copy()
    u_idx = np.where(unseen_mask)[0]
    ct_thr[u_idx[lo_mask_tu]] += corr_lo
    ct_thr[u_idx[hi_mask_tu]] += corr_hi
    ct_thr = np.clip(ct_thr, 0, None)
    du_thr = ct_thr[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  [{label:14s}] corr_lo={corr_lo:6.2f}  corr_hi={corr_hi:6.2f}  Δ={du_thr:+.4f}  unseen={ct_thr[unseen_mask].mean():.3f}")

# Save best threshold correction (hi_half = conservative)
ct_best = oracle_new_t.copy()
u_idx = np.where(unseen_mask)[0]
ct_best[u_idx[hi_mask_tu]] += mr_hi / 2
ct_best = np.clip(ct_best, 0, None)
du_best = ct_best[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname_th = "FINAL_NEW_oN_packutil_hi_half_OOF8.3825.csv"
sub_th = sub_tmpl.copy(); sub_th['avg_delay_minutes_next_30m'] = ct_best
sub_th.to_csv(fname_th, index=False)
print(f"\n  Saved: {fname_th}  Δ={du_best:+.4f}  unseen={ct_best[unseen_mask].mean():.3f}")

# Save full high correction
ct_hi_full = oracle_new_t.copy()
ct_hi_full[u_idx[hi_mask_tu]] += mr_hi
ct_hi_full = np.clip(ct_hi_full, 0, None)
du_hf = ct_hi_full[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname_hf = "FINAL_NEW_oN_packutil_hi_full_OOF8.3825.csv"
sub_hf = sub_tmpl.copy(); sub_hf['avg_delay_minutes_next_30m'] = ct_hi_full
sub_hf.to_csv(fname_hf, index=False)
print(f"  Saved: {fname_hf}  Δ={du_hf:+.4f}  unseen={ct_hi_full[unseen_mask].mean():.3f}")

# Save both_quarter (most conservative mixed)
ct_bq = oracle_new_t.copy()
ct_bq[u_idx[lo_mask_tu]] += mr_lo / 4
ct_bq[u_idx[hi_mask_tu]] += mr_hi / 4
ct_bq = np.clip(ct_bq, 0, None)
du_bq = ct_bq[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname_bq = "FINAL_NEW_oN_packutil_both_quarter_OOF8.3825.csv"
sub_bq = sub_tmpl.copy(); sub_bq['avg_delay_minutes_next_30m'] = ct_bq
sub_bq.to_csv(fname_bq, index=False)
print(f"  Saved: {fname_bq}  Δ={du_bq:+.4f}  unseen={ct_bq[unseen_mask].mean():.3f}")

# ============================================================
# Fine-grained: oracle_NEW OOF residual as a function of pack_util
# Use local polynomial regression fit
# ============================================================
print("\n" + "="*70)
print("Local residual: E[residual | pack_util] via binning")
print("Extrapolate to OOD pack_util > 0.85 range in unseen test")
print("="*70)
bins10 = np.linspace(0, 1.0, 21)
bin_centers = []
bin_resids  = []
for lo, hi in zip(bins10[:-1], bins10[1:]):
    m = (pu_train >= lo) & (pu_train < hi)
    if m.sum() >= 200:
        bin_centers.append((lo + hi) / 2)
        bin_resids.append(residuals_train[m].mean())

bc = np.array(bin_centers)
br = np.array(bin_resids)
print(f"\n  Bin centers: {bc}")
print(f"  Mean residuals: {br}")

# Fit quadratic regression on bins
from numpy.polynomial import polynomial as P
coefs = np.polyfit(bc, br, deg=2)
print(f"\n  Quadratic fit: {coefs[0]:.4f}*x² + {coefs[1]:.4f}*x + {coefs[2]:.4f}")
for pu_val in [0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 1.00]:
    pred_r = np.polyval(coefs, pu_val)
    print(f"    pack_util={pu_val:.2f}: predicted residual = {pred_r:.3f}")

# Apply quadratic correction for unseen only
pu_test_u = pu_test[unseen_mask]
quad_corr = np.polyval(coefs, np.clip(pu_test_u, bc.min(), None))  # no lower cap needed
ct_quad = oracle_new_t.copy()
ct_quad[unseen_mask] = oracle_new_t[unseen_mask] + quad_corr
ct_quad = np.clip(ct_quad, 0, None)
du_quad = ct_quad[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Quadratic correction (full): Δ={du_quad:+.4f}  unseen={ct_quad[unseen_mask].mean():.3f}")

# 50% of quadratic
ct_q50 = oracle_new_t.copy()
ct_q50[unseen_mask] = oracle_new_t[unseen_mask] + 0.5 * quad_corr
ct_q50 = np.clip(ct_q50, 0, None)
du_q50 = ct_q50[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Quadratic correction (50%): Δ={du_q50:+.4f}  unseen={ct_q50[unseen_mask].mean():.3f}")

fname_q = "FINAL_NEW_oN_quadPackutil_OOF8.3825.csv"
sub_q = sub_tmpl.copy(); sub_q['avg_delay_minutes_next_30m'] = ct_quad
sub_q.to_csv(fname_q, index=False)
print(f"  Saved: {fname_q}")

fname_q50 = "FINAL_NEW_oN_quadPackutil_half_OOF8.3825.csv"
sub_q50 = sub_tmpl.copy(); sub_q50['avg_delay_minutes_next_30m'] = ct_q50
sub_q50.to_csv(fname_q50, index=False)
print(f"  Saved: {fname_q50}")

print("\nDone.")
