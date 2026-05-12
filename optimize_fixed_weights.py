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

def make_oracle(wm, wr, w1, w2, w3, wseq=0.64, wxgb=0.12, wlv2=0.16, wrem=0.08):
    fixed_oof_  = wm*mega33_oof + wr*rank_oof + w1*r1_oof + w2*r2_oof + w3*r3_oof
    fixed_test_ = wm*mega33_test + wr*rank_test + w1*r1_test + w2*r2_test + w3*r3_test
    o_oof  = np.clip(wseq*fixed_oof_  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    o_test = np.clip(wseq*fixed_test_ + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    return o_oof, o_test

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

oracle_oof, oracle_test = make_oracle(fw['mega33'], fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], fw['iter_r3'])
base_oof = mae(oracle_oof)
print(f'oracle_NEW baseline: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

# Sweep r3 weight change (transfer to mega33)
print('\n--- r3→mega33 transfer sweep ---')
best_oof = base_oof
best_params = None
for dr3 in np.arange(-0.15, 0.05, 0.01):
    wm = fw['mega33'] - dr3
    w3 = fw['iter_r3'] + dr3
    o_oof, o_test = make_oracle(wm, fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], w3)
    delta = mae(o_oof) - base_oof
    if mae(o_oof) < best_oof:
        best_oof = mae(o_oof)
        best_params = (wm, fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], w3)
    marker = '*' if mae(o_oof) < base_oof else ''
    print(f'  dr3={dr3:+.2f}: w_mega33={wm:.4f}  w_r3={w3:.4f}  OOF={mae(o_oof):.5f}  delta={delta:+.6f}  test={o_test.mean():.3f} {marker}')

# Also sweep r2 weight
print('\n--- r2→mega33 transfer sweep ---')
for dr2 in np.arange(-0.10, 0.05, 0.01):
    wm = fw['mega33'] - dr2
    w2 = fw['iter_r2'] + dr2
    o_oof, o_test = make_oracle(wm, fw['rank_adj'], fw['iter_r1'], w2, fw['iter_r3'])
    delta = mae(o_oof) - base_oof
    marker = '*' if mae(o_oof) < base_oof else ''
    print(f'  dr2={dr2:+.2f}: w_mega33={wm:.4f}  w_r2={w2:.4f}  OOF={mae(o_oof):.5f}  delta={delta:+.6f}  test={o_test.mean():.3f} {marker}')

# Joint r2+r3 optimization
print('\n--- Joint r2+r3→mega33 sweep ---')
best_joint = base_oof
for dr2 in [0.0, -0.02, -0.04, -0.06]:
    for dr3 in [0.0, -0.02, -0.04, -0.06, -0.08]:
        wm = fw['mega33'] - dr2 - dr3
        w2 = fw['iter_r2'] + dr2
        w3 = fw['iter_r3'] + dr3
        o_oof, o_test = make_oracle(wm, fw['rank_adj'], fw['iter_r1'], w2, w3)
        if mae(o_oof) < best_joint:
            best_joint = mae(o_oof)
            best_params = (wm, fw['rank_adj'], fw['iter_r1'], w2, w3)
        delta = mae(o_oof) - base_oof
        if mae(o_oof) < base_oof - 0.0001:  # Only show if clearly better
            print(f'  dr2={dr2:+.2f} dr3={dr3:+.2f}: wm={wm:.4f} w2={w2:.4f} w3={w3:.4f}  OOF={mae(o_oof):.5f}  delta={delta:+.6f}  test={o_test.mean():.3f}')

if best_params:
    print(f'\nBest configuration: {best_params}  OOF={best_joint:.5f}  delta={best_joint-base_oof:+.6f}')
    o_oof_best, o_test_best = make_oracle(*best_params)
    # Fold-level analysis
    groups = train_raw['layout_id'].values
    gkf = GroupKFold(n_splits=5)
    print('\nFold-level comparison:')
    for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        orig_fold = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs], 0, None))
        best_fold = mean_absolute_error(y_true[vs], o_oof_best[vs])
        print(f'  Fold {fi+1}: orig={orig_fold:.5f}  best={best_fold:.5f}  delta={best_fold-orig_fold:+.6f}')

    # Generate submission
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = o_test_best
    oof_val = mae(o_oof_best)
    fname = f'submission_fixed_reweighted_OOF{oof_val:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'\nSaved: {fname}  OOF={oof_val:.5f}  test_mean={o_test_best.mean():.3f}')

# Also check: what if we REMOVE r3 entirely (weight=0)?
print('\n--- Remove r3 entirely from FIXED ---')
wm_no_r3 = fw['mega33'] + fw['iter_r3']  # redistribute r3 weight to mega33
o_oof_nr3, o_test_nr3 = make_oracle(wm_no_r3, fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], 0.0)
print(f'  No r3: OOF={mae(o_oof_nr3):.5f}  delta={mae(o_oof_nr3)-base_oof:+.6f}  test={o_test_nr3.mean():.3f}')

wm_no_r23 = fw['mega33'] + fw['iter_r2'] + fw['iter_r3']  # remove r2 and r3
o_oof_nr23, o_test_nr23 = make_oracle(wm_no_r23, fw['rank_adj'], fw['iter_r1'], 0.0, 0.0)
print(f'  No r2+r3: OOF={mae(o_oof_nr23):.5f}  delta={mae(o_oof_nr23)-base_oof:+.6f}  test={o_test_nr23.mean():.3f}')

# What about replacing r2+r3 with r1 (best pseudo-labeling round)?
wm_r1_heavy = fw['mega33']
w2_r1 = fw['iter_r1'] + fw['iter_r2'] + fw['iter_r3']  # give all pseudo weight to r1
o_oof_r1h, o_test_r1h = make_oracle(wm_r1_heavy, fw['rank_adj'], w2_r1, 0.0, 0.0)
print(f'  r1_heavy (r1 absorbs r2+r3): OOF={mae(o_oof_r1h):.5f}  delta={mae(o_oof_r1h)-base_oof:+.6f}  test={o_test_r1h.mean():.3f}')

print('\nDone.')
