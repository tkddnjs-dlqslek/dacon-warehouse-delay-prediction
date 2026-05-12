import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import itertools

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

# Best FIXED reweight from previous analysis
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

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

fixed_oof  = wm_best*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
fixed_test = wm_best*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# Oracle_NEW baseline (original weights)
base_fixed_oof  = (fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof
                 + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof)
base_fixed_test = (fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test
                 + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test)
oracle_oof  = np.clip(0.64*base_fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*base_fixed_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)
base_oof = mae(oracle_oof)
print(f'oracle_NEW baseline: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

fixed_rw_oof_val = mae(np.clip(0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None))
print(f'fixed_reweighted: OOF={fixed_rw_oof_val:.5f}')

# Sweep oracle_seq weights with best FIXED reweighting
print('\n=== Oracle seq weight sweep (best FIXED reweight applied) ===')
print(f'{"wf":>5} {"wxgb":>6} {"wlv2":>6} {"wrem":>6}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')

results = []
for wf in [0.58, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70]:
    w_total = 1.0 - wf
    # Proportional scaling: default ratio xgb:lv2:rem = 0.12:0.16:0.08 = 3:4:2
    # Try varying lv2/xgb ratio
    for xgb_frac in [0.25, 0.30, 0.33, 0.40, 0.45, 0.50]:
        for lv2_frac in [0.30, 0.35, 0.40, 0.44, 0.50, 0.55]:
            rem_frac = 1.0 - xgb_frac - lv2_frac
            if rem_frac < 0.05 or rem_frac > 0.40:
                continue
            wxgb = w_total * xgb_frac
            wlv2 = w_total * lv2_frac
            wrem = w_total * rem_frac
            oo = np.clip(wf*fixed_oof + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
            ot = np.clip(wf*fixed_test + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
            oof_val = mae(oo)
            if oof_val < base_oof - 0.0003:
                results.append((oof_val, wf, wxgb, wlv2, wrem, ot.mean()))

results.sort()
print(f'Found {len(results)} configs with OOF < baseline - 0.0003')
for r in results[:20]:
    oof_v, wf, wxgb, wlv2, wrem, tm = r
    print(f'  wf={wf:.2f} wxgb={wxgb:.3f} wlv2={wlv2:.3f} wrem={wrem:.3f}  OOF={oof_v:.5f}  delta={oof_v-base_oof:+.6f}  test={tm:.3f}')

# Default ratio sweep (proportional scaling)
print('\n=== wf sweep with proportional oracle_seq scaling ===')
print(f'{"wf":>5} {"wxgb":>6} {"wlv2":>6} {"wrem":>6}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')
for wf in np.arange(0.55, 0.85, 0.02):
    w_rem = 1.0 - wf
    wxgb = 0.12 * w_rem / 0.36
    wlv2 = 0.16 * w_rem / 0.36
    wrem = 0.08 * w_rem / 0.36
    oo = np.clip(wf*fixed_oof + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fixed_test + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    marker = '*' if mae(oo) < base_oof else ''
    print(f'  {wf:.2f}  {wxgb:.3f}  {wlv2:.3f}  {wrem:.3f}  {mae(oo):.5f}  {mae(oo)-base_oof:+.6f}  {ot.mean():.3f} {marker}')

# Best wf=0.62 submission
wf62 = 0.62
w_rem62 = 1.0 - wf62
wxgb62 = 0.12 * w_rem62 / 0.36
wlv2_62 = 0.16 * w_rem62 / 0.36
wrem62 = 0.08 * w_rem62 / 0.36
oo62 = np.clip(wf62*fixed_oof + wxgb62*xgb_o + wlv2_62*lv2_o + wrem62*rem_o, 0, None)
ot62 = np.clip(wf62*fixed_test + wxgb62*xgb_t + wlv2_62*lv2_t + wrem62*rem_t, 0, None)
print(f'\nwf=0.62 submission: OOF={mae(oo62):.5f}  delta={mae(oo62)-base_oof:+.6f}  test_mean={ot62.mean():.3f}')
sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = ot62
fname = f'submission_rw_wf062_OOF{mae(oo62):.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}')

# Best wf=0.70 with best fixed reweight (for LB generalization hypothesis)
wf70 = 0.70
w_rem70 = 1.0 - wf70
wxgb70 = 0.12 * w_rem70 / 0.36
wlv2_70 = 0.16 * w_rem70 / 0.36
wrem70 = 0.08 * w_rem70 / 0.36
oo70 = np.clip(wf70*fixed_oof + wxgb70*xgb_o + wlv2_70*lv2_o + wrem70*rem_o, 0, None)
ot70 = np.clip(wf70*fixed_test + wxgb70*xgb_t + wlv2_70*lv2_t + wrem70*rem_t, 0, None)
print(f'\nwf=0.70 submission: OOF={mae(oo70):.5f}  delta={mae(oo70)-base_oof:+.6f}  test_mean={ot70.mean():.3f}')
sub['avg_delay_minutes_next_30m'] = ot70
fname70 = f'submission_rw_wf070_OOF{mae(oo70):.5f}.csv'
sub.to_csv(fname70, index=False)
print(f'Saved: {fname70}')

# Individual oracle component comparison
print('\n=== Individual oracle_seq component analysis ===')
for name, oo_c, ot_c in [('xgb', xgb_o, xgb_t), ('lv2', lv2_o, lv2_t), ('rem', rem_o, rem_t)]:
    print(f'  {name}: OOF={mae(oo_c):.5f}  test_mean={ot_c.mean():.3f}  corr_with_base={np.corrcoef(oracle_oof,oo_c)[0,1]:.4f}')

# Check xgb vs lv2 as dominant oracle component
print('\n=== xgb-dominant vs lv2-dominant oracle ===')
for wxgb_frac, wlv2_frac in [(0.50, 0.30), (0.45, 0.35), (0.40, 0.40), (0.33, 0.45), (0.25, 0.50), (0.20, 0.55)]:
    wrem_frac = 1.0 - wxgb_frac - wlv2_frac
    wxgb_v = 0.36 * wxgb_frac
    wlv2_v = 0.36 * wlv2_frac
    wrem_v = 0.36 * wrem_frac
    oo = np.clip(0.64*fixed_oof + wxgb_v*xgb_o + wlv2_v*lv2_o + wrem_v*rem_o, 0, None)
    ot = np.clip(0.64*fixed_test + wxgb_v*xgb_t + wlv2_v*lv2_t + wrem_v*rem_t, 0, None)
    marker = '*' if mae(oo) < base_oof else ''
    print(f'  xgb={wxgb_v:.3f} lv2={wlv2_v:.3f} rem={wrem_v:.3f}: OOF={mae(oo):.5f}  delta={mae(oo)-base_oof:+.6f}  test={ot.mean():.3f} {marker}')

print('\nDone.')
