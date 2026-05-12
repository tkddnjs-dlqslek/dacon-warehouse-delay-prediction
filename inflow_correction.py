import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t   = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t   = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

fx_orig_oof = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fx_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
oracle_oof = np.clip(0.64*fx_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fx_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

wf=0.72; w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
m_bl = 0.75*mega33_oof+0.25*mega34_oof; m_bl_t = 0.75*mega33_test+0.25*mega34_test
best_base_oof = np.clip((1-0.12)*(wf*(wm_best*m_bl+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_best*r2_oof+w3_best*r3_oof)+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o) + 0.12*cb_oof, 0, None)
best_base_test = np.clip((1-0.12)*(wf*(wm_best*m_bl_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2_best*r2_test+w3_best*r3_test)+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t) + 0.12*cb_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
best_base_v = mae(best_base_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'best_base: OOF={best_base_v:.5f}  test_mean={best_base_test.mean():.3f}')

# Check inflow/pack NaN status
print(f'\nTraining NaN: inflow={train_raw["order_inflow_15m"].isna().sum()}  pack={train_raw["pack_utilization"].isna().sum()}')
print(f'Training inflow: min={train_raw["order_inflow_15m"].min():.1f}  max={train_raw["order_inflow_15m"].max():.1f}  count_valid={train_raw["order_inflow_15m"].notna().sum()}')
print(f'Training total rows: {len(train_raw)}')

# Use valid rows only
valid_mask = train_raw['order_inflow_15m'].notna() & train_raw['pack_utilization'].notna()
n_valid = valid_mask.sum()
print(f'Valid training rows: {n_valid}')

# Check inflow distribution for valid rows
inflow_v = train_raw.loc[valid_mask, 'order_inflow_15m'].values
pack_v = train_raw.loc[valid_mask, 'pack_utilization'].values
y_v = y_true[valid_mask]
oracle_v = oracle_oof[valid_mask]
best_v = best_base_oof[valid_mask]
resid_v = y_v - np.clip(oracle_v, 0, None)

print(f'Inflow stats: p25={np.percentile(inflow_v,25):.1f}  p50={np.percentile(inflow_v,50):.1f}  p75={np.percentile(inflow_v,75):.1f}  p90={np.percentile(inflow_v,90):.1f}  max={inflow_v.max():.1f}')
print()

# Residual by inflow bins
q_vals = np.percentile(inflow_v, [25, 50, 75, 90, 95])
print('=== Oracle residual by inflow bin ===')
for label, lo, hi in [
    ('all', -np.inf, np.inf),
    ('<p25', -np.inf, q_vals[0]),
    ('p25-50', q_vals[0], q_vals[1]),
    ('p50-75', q_vals[1], q_vals[2]),
    ('p75-90', q_vals[2], q_vals[3]),
    ('p90-95', q_vals[3], q_vals[4]),
    ('>p95', q_vals[4], np.inf),
]:
    m = (inflow_v >= lo) & (inflow_v < hi)
    if not m.any():
        continue
    r = resid_v[m]
    print(f'  {label:12s}: n={m.sum():6d}  inflow={inflow_v[m].mean():.1f}  y={y_v[m].mean():.2f}  pred={oracle_v[m].mean():.2f}  resid={r.mean():+.2f}')

# Simple inflow-quantile correction
print('\n=== Simple inflow-quantile correction on test ===')
# Compute residuals per quantile from training data (using OOF — no leakage for training)
train_inflow = train_raw['order_inflow_15m'].fillna(0).values
test_inflow = test_raw['order_inflow_15m'].fillna(0).values

# Binned correction based on quantile bins from training
q25t, q50t, q75t, q90t, q95t = q_vals
bins = [(-np.inf, q25t), (q25t, q50t), (q50t, q75t), (q75t, q90t), (q90t, q95t), (q95t, np.inf)]
bin_resids = []  # mean residual per bin

for lo, hi in bins:
    m = (inflow_v >= lo) & (inflow_v < hi)
    if m.sum() > 0:
        bin_resids.append(resid_v[m].mean())
    else:
        bin_resids.append(0.0)
    print(f'  inflow [{lo:.0f}, {hi:.0f}): n={m.sum()}  mean_resid={bin_resids[-1]:+.2f}')

# Apply correction to test
def apply_bin_correction(preds, inflow, bins, bin_resids, alpha=1.0):
    corrected = preds.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = (inflow >= lo) & (inflow < hi)
        corrected[m] = np.clip(preds[m] + alpha*r, 0, None)
    return corrected

# Test on training (using OOF)
for alpha in [0.3, 0.5, 0.7, 1.0]:
    corr_train = apply_bin_correction(oracle_oof, train_inflow, bins, bin_resids, alpha)
    corr_test = apply_bin_correction(oracle_test, test_inflow, bins, bin_resids, alpha)
    marker = '*' if mae(corr_train) < base_oof else ''
    print(f'  alpha={alpha:.1f}: OOF={mae(corr_train):.5f} ({mae(corr_train)-base_oof:+.6f})  test_mean={corr_test.mean():.3f} (+{corr_test.mean()-oracle_test.mean():+.3f}) {marker}')

# Apply only to unseen test layouts
print('\n=== Apply only to UNSEEN test layouts ===')
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
print(f'Unseen test rows: {unseen_mask.sum()}')

for alpha in [0.3, 0.5, 0.7, 1.0]:
    corr_test_unseen = oracle_test.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
        corr_test_unseen[m] = np.clip(oracle_test[m] + alpha*r, 0, None)
    print(f'  alpha={alpha:.1f}: test_mean={corr_test_unseen.mean():.3f} (+{corr_test_unseen.mean()-oracle_test.mean():+.3f})  unseen_mean={corr_test_unseen[unseen_mask].mean():.3f}')

# Combined: best_base + inflow correction on unseen
print('\n=== best_base + inflow correction on unseen test ===')
for alpha in [0.3, 0.5, 0.7, 1.0]:
    corr_test_bb = best_base_test.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
        corr_test_bb[m] = np.clip(best_base_test[m] + alpha*r, 0, None)
    print(f'  alpha={alpha:.1f}: test_mean={corr_test_bb.mean():.3f} ({corr_test_bb.mean()-oracle_test.mean():+.3f})  OOF_unchanged={best_base_v:.5f}')

# Save best: oracle + inflow correction all rows at alpha=0.5
print('\n--- Saving submissions ---')
sub = pd.read_csv('sample_submission.csv')

# 1. oracle + all-rows inflow correction, alpha=0.5
for alpha in [0.5, 0.7]:
    corr = apply_bin_correction(oracle_test, test_inflow, bins, bin_resids, alpha)
    corr_oof = apply_bin_correction(oracle_oof, train_inflow, bins, bin_resids, alpha)
    sub['avg_delay_minutes_next_30m'] = corr
    fname = f'submission_inflow_corr_a{alpha:.1f}_OOF{mae(corr_oof):.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}  OOF={mae(corr_oof):.5f}  test_mean={corr.mean():.3f}')

# 2. best_base + unseen-only inflow correction, alpha=0.5
for alpha in [0.5]:
    corr_test_bb = best_base_test.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
        corr_test_bb[m] = np.clip(best_base_test[m] + alpha*r, 0, None)
    sub['avg_delay_minutes_next_30m'] = corr_test_bb
    fname = f'submission_best_base_unseen_inflow_a{alpha:.1f}_OOF{best_base_v:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}  OOF={best_base_v:.5f}  test_mean={corr_test_bb.mean():.3f}')

print('\nDone.')
