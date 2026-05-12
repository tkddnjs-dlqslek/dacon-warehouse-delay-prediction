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
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
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

def compute_all(wm, wr, w1, w2, w3, wf=0.64, wxgb=0.12, wlv2=0.16, wrem=0.08):
    fo  = wm*mega33_oof + wr*rank_oof + w1*r1_oof + w2*r2_oof + w3*r3_oof
    ft  = wm*mega33_test + wr*rank_test + w1*r1_test + w2*r2_test + w3*r3_test
    oo  = np.clip(wf*fo + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot  = np.clip(wf*ft + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    return oo, ot

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# Baseline
oof_base, test_base = compute_all(fw['mega33'], fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], fw['iter_r3'])
base_oof = mae(oof_base)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={test_base.mean():.3f}')

# Best FIXED reweight from previous analysis
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3
oof_rw, test_rw = compute_all(wm_best, fw['rank_adj'], fw['iter_r1'], w2_best, w3_best)
print(f'fixed_reweighted: OOF={mae(oof_rw):.5f}  test_mean={test_rw.mean():.3f}')

# Combine: FIXED reweight + higher FIXED contribution (wf>0.64)
print('\n--- Combined: fixed_reweight + wf sweep ---')
print(f'{"wf":>6}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')
for wf in [0.60, 0.62, 0.64, 0.66, 0.68, 0.70, 0.72, 0.75]:
    w_rem = 1.0 - wf
    wxgb = 0.12 * w_rem / 0.36
    wlv2 = 0.16 * w_rem / 0.36
    wrem = 0.08 * w_rem / 0.36
    oo, ot = compute_all(wm_best, fw['rank_adj'], fw['iter_r1'], w2_best, w3_best, wf, wxgb, wlv2, wrem)
    delta = mae(oo) - base_oof
    marker = '*' if mae(oo) < base_oof else ''
    print(f'{wf:>6.2f}  {mae(oo):>9.5f}  {delta:>+9.6f}  {ot.mean():>10.3f} {marker}')

# Check: what about consensus_nogate? We need to see its components
import os
for fname in os.listdir('.'):
    if 'consensus_nogate' in fname and fname.endswith('.csv'):
        print(f'\nFound: {fname}')
        df = pd.read_csv(fname)
        print(f'  test_mean={df["avg_delay_minutes_next_30m"].mean():.3f}')

# Load consensus_nogate oof if it exists
if os.path.exists('results/consensus_nogate_oof.npy'):
    cng_oof = np.load('results/consensus_nogate_oof.npy')
    print(f'\nconsensus_nogate OOF: {mae(cng_oof):.5f}')
    print(f'corr(cng, oracle_NEW): {np.corrcoef(oof_base, cng_oof)[0,1]:.4f}')
    print(f'corr(cng, fixed_rw): {np.corrcoef(oof_rw, cng_oof)[0,1]:.4f}')
    # Blend consensus_nogate with fixed_reweighted
    print('\nBlend cng + fixed_rw:')
    for w in [0.3, 0.5, 0.7]:
        b = (1-w)*cng_oof + w*oof_rw
        print(f'  w_rw={w}: OOF={mae(b):.5f}  delta={mae(b)-base_oof:+.6f}')

# Final best submission candidates
print('\n=== Best submission candidates ===')
candidates = {}

# 1. oracle_NEW
candidates['oracle_NEW'] = (oof_base, test_base)

# 2. fixed_reweighted (dr2=-0.04, dr3=-0.02)
candidates['fixed_reweighted'] = (oof_rw, test_rw)

# 3. "no r2+r3" variant
wm_nr23 = fw['mega33'] + fw['iter_r2'] + fw['iter_r3']
oof_nr23, test_nr23 = compute_all(wm_nr23, fw['rank_adj'], fw['iter_r1'], 0.0, 0.0)
candidates['no_r2r3'] = (oof_nr23, test_nr23)

# 4. Best r2-only transfer (dr2=-0.07)
wm_r2 = fw['mega33'] + 0.07
w2_r2 = fw['iter_r2'] - 0.07
oof_r2, test_r2 = compute_all(wm_r2, fw['rank_adj'], fw['iter_r1'], w2_r2, fw['iter_r3'])
candidates['r2_transfer_07'] = (oof_r2, test_r2)

for name, (oo, ot) in candidates.items():
    print(f'  {name:25s}: OOF={mae(oo):.5f}  delta={mae(oo)-base_oof:+.6f}  test_mean={ot.mean():.3f}  test_delta={ot.mean()-test_base.mean():+.4f}')

# Generate best submission (highest test_mean among those with OOF improvement)
print('\n--- Submitting best candidates ---')
sub = pd.read_csv('sample_submission.csv')

# 1. fixed_reweighted (best OOF + higher test_mean)
sub['avg_delay_minutes_next_30m'] = test_rw
sub.to_csv(f'submission_fixed_reweighted_OOF{mae(oof_rw):.5f}.csv', index=False)
print(f'Saved: submission_fixed_reweighted_OOF{mae(oof_rw):.5f}.csv')

# 2. "no r2+r3"
sub['avg_delay_minutes_next_30m'] = test_nr23
sub.to_csv(f'submission_no_r2r3_OOF{mae(oof_nr23):.5f}.csv', index=False)
print(f'Saved: submission_no_r2r3_OOF{mae(oof_nr23):.5f}.csv')

# 3. r2_transfer_07 — good OOF + decent test_mean
sub['avg_delay_minutes_next_30m'] = test_r2
sub.to_csv(f'submission_r2transfer07_OOF{mae(oof_r2):.5f}.csv', index=False)
print(f'Saved: submission_r2transfer07_OOF{mae(oof_r2):.5f}.csv')

# Fold-level analysis for fixed_reweighted
print('\nFold-level analysis for fixed_reweighted:')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    fo = mean_absolute_error(y_true[vs], np.clip(oof_base[vs],0,None))
    fr = mean_absolute_error(y_true[vs], np.clip(oof_rw[vs],0,None))
    print(f'  Fold {fi+1}: oracle={fo:.5f}  reweighted={fr:.5f}  delta={fr-fo:+.6f}')

print('\nDone.')
