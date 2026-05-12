import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr

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

inflow_col = 'order_inflow_15m'
train_median_inflow = train_raw[inflow_col].median()
test_median_inflow = test_raw[inflow_col].median()

inflow_tr = train_raw[inflow_col].fillna(train_median_inflow).values
inflow_te = test_raw[inflow_col].fillna(test_median_inflow).values

print("="*70)
print("Row-level Inflow-Quantile Correction")
print("For each unseen row, apply correction based on its inflow value")
print("="*70)

# Inflow quantile buckets from highinflow_only_lgb.py analysis:
# Q0-25  (<35): resid=+1.002
# Q25-50 (35-78): resid=+2.274
# Q50-75 (78-132): resid=+3.553
# Q75-90 (132-200): resid=+5.166
# Q90-100 (>200): resid=+6.102

inflow_bins = [(-np.inf, 35), (35, 78), (78, 132), (132, 200), (200, np.inf)]
inflow_resids = [1.002, 2.274, 3.553, 5.166, 6.102]

print("\n  Inflow bucket residuals:")
for (lo, hi), r in zip(inflow_bins, inflow_resids):
    n_tr = ((inflow_tr >= lo) & (inflow_tr < hi)).sum()
    n_u  = ((inflow_te[unseen_mask] >= lo) & (inflow_te[unseen_mask] < hi)).sum()
    print(f"  [{lo if lo != -np.inf else 0:.0f}, {hi if hi != np.inf else 500:.0f}): "
          f"n_train={n_tr:7d}  n_unseen_test={n_u:6d} ({100*n_u/unseen_mask.sum():.1f}%)  resid={r:.3f}")

# Compute per-row correction for unseen test rows
def get_inflow_correction(inflow_val):
    for (lo, hi), r in zip(inflow_bins, inflow_resids):
        if inflow_val >= lo and inflow_val < hi:
            return r
    return inflow_resids[-1]

inflow_te_unseen = inflow_te[unseen_mask]
row_corrections = np.array([get_inflow_correction(v) for v in inflow_te_unseen])

print(f"\n  Row correction stats for unseen test:")
print(f"    mean={row_corrections.mean():.4f}  std={row_corrections.std():.4f}")
print(f"    min={row_corrections.min():.3f}  max={row_corrections.max():.3f}")

ct_rowq = oracle_new_t.copy()
ct_rowq[unseen_mask] = oracle_new_t[unseen_mask] + row_corrections
ct_rowq = np.clip(ct_rowq, 0, None)
du_rowq = ct_rowq[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Inflow-quantile row correction: Δ={du_rowq:+.4f}  seen={ct_rowq[seen_mask].mean():.3f}  unseen={ct_rowq[unseen_mask].mean():.3f}")

fname_rowq = "FINAL_NEW_oN_inflowQuantile_OOF8.3825.csv"
sub_rowq = sub_tmpl.copy(); sub_rowq['avg_delay_minutes_next_30m'] = ct_rowq
sub_rowq.to_csv(fname_rowq, index=False)
print(f"  Saved: {fname_rowq}")

# Also try linear interpolation between bucket midpoints
print(f"\n--- Linear interpolation version ---")
bucket_mids = [17.5, 56.5, 105, 166, 300]  # midpoints
bucket_resids = [1.002, 2.274, 3.553, 5.166, 6.102]

from scipy.interpolate import interp1d
interp_fn = interp1d(bucket_mids, bucket_resids, kind='linear', bounds_error=False,
                      fill_value=(bucket_resids[0], bucket_resids[-1]))

corr_interp = interp_fn(inflow_te_unseen)
ct_interp = oracle_new_t.copy()
ct_interp[unseen_mask] = oracle_new_t[unseen_mask] + corr_interp
ct_interp = np.clip(ct_interp, 0, None)
du_interp = ct_interp[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Interpolated inflow correction: Δ={du_interp:+.4f}  seen={ct_interp[seen_mask].mean():.3f}  unseen={ct_interp[unseen_mask].mean():.3f}")

fname_interp = "FINAL_NEW_oN_inflowInterp_OOF8.3825.csv"
sub_interp = sub_tmpl.copy(); sub_interp['avg_delay_minutes_next_30m'] = ct_interp
sub_interp.to_csv(fname_interp, index=False)
print(f"  Saved: {fname_interp}")

# ============================================================
# 2D correction: inflow + oracle_NEW prediction value
# For unseen rows: correction = f(inflow, pred)
# ============================================================
print(f"\n--- 2D correction: f(inflow_bucket, pred_bucket) ---")
# For each combination of inflow bucket and pred bucket, compute training residual
pred_bins = [0, 10, 20, 30, 40, 200]
resid_2d = {}
for (inlo, inhi), r_in in zip(inflow_bins, inflow_resids):
    for (plo, phi) in zip(pred_bins[:-1], pred_bins[1:]):
        m = (inflow_tr >= inlo) & (inflow_tr < inhi) & (fw4_oo >= plo) & (fw4_oo < phi)
        if m.sum() >= 100:
            resid_2d[(inlo, inhi, plo, phi)] = residuals_train[m].mean()
        else:
            resid_2d[(inlo, inhi, plo, phi)] = None

print(f"\n  2D residual table (rows=inflow, cols=pred):")
print(f"  {'inflow':25s}", end='')
for plo, phi in zip(pred_bins[:-1], pred_bins[1:]):
    print(f"  [{plo:3d},{phi:3d})", end='')
print()
for (inlo, inhi), r_in in zip(inflow_bins, inflow_resids):
    label = f"[{inlo if inlo!=-np.inf else 0:.0f},{inhi if inhi!=np.inf else 500:.0f})"
    print(f"  {label:25s}", end='')
    for plo, phi in zip(pred_bins[:-1], pred_bins[1:]):
        v = resid_2d.get((inlo, inhi, plo, phi))
        if v is not None:
            print(f"  {v:+7.3f} ", end='')
        else:
            print(f"  {'N/A':>8} ", end='')
    print()

# Apply 2D correction to unseen test rows
pred_te_unseen = oracle_new_t[unseen_mask]
def get_2d_correction(inflow_val, pred_val):
    for (inlo, inhi) in inflow_bins:
        if inflow_val >= inlo and inflow_val < inhi:
            for (plo, phi) in zip(pred_bins[:-1], pred_bins[1:]):
                if pred_val >= plo and pred_val < phi:
                    v = resid_2d.get((inlo, inhi, plo, phi))
                    if v is not None:
                        return v
                    else:
                        # Fall back to inflow-only correction
                        return get_inflow_correction(inflow_val)
    return 3.175  # global mean

corr_2d = np.array([get_2d_correction(inf, pred)
                    for inf, pred in zip(inflow_te_unseen, pred_te_unseen)])
ct_2d = oracle_new_t.copy()
ct_2d[unseen_mask] = oracle_new_t[unseen_mask] + corr_2d
ct_2d = np.clip(ct_2d, 0, None)
du_2d = ct_2d[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  2D correction: Δ={du_2d:+.4f}  seen={ct_2d[seen_mask].mean():.3f}  unseen={ct_2d[unseen_mask].mean():.3f}")

fname_2d = "FINAL_NEW_oN_inflow2D_OOF8.3825.csv"
sub_2d = sub_tmpl.copy(); sub_2d['avg_delay_minutes_next_30m'] = ct_2d
sub_2d.to_csv(fname_2d, index=False)
print(f"  Saved: {fname_2d}")

# ============================================================
# Final table: compare all row-level correction candidates
# ============================================================
print(f"\n" + "="*70)
print("All new correction files: final comparison")
print("="*70)
all_new = [
    'FINAL_NEW_oN_udelta5p5_OOF8.3825.csv',
    'FINAL_NEW_oN_udelta5p77_OOF8.3825.csv',
    'FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv',
    'FINAL_NEW_oN_linearBias_OOF8.3825.csv',
    'FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv',
    'FINAL_NEW_oN_inflowQuantile_OOF8.3825.csv',
    'FINAL_NEW_oN_inflowInterp_OOF8.3825.csv',
    'FINAL_NEW_oN_inflow2D_OOF8.3825.csv',
    'FINAL_NEW_oN_knn10_OOF8.3825.csv',
]
print(f"  {'file':50s}  {'seen':>8}  {'unseen':>8}  {'Δ':>9}")
for fname in all_new:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname[:50]:50s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}")
    except Exception as e:
        print(f"  {fname[:50]:50s}  MISSING")

print("\nDone.")
