import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
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
xgb33_oof  = np.clip(d33['meta_oofs']['xgb'][id2], 0, None)
xgb33_test = np.clip(d33['meta_tests']['xgb'][te_id2], 0, None)

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

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

def make_base(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3
    w2 = fw['iter_r2'] + dr2
    w3 = fw['iter_r3'] + dr3
    fx  = wm*mega + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0 - wf; wxgb = 0.12*w_rem/0.36; wlv2 = 0.16*w_rem/0.36; wrem = 0.08*w_rem/0.36
    oo = np.clip(wf*fx + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    return oo, ot

oracle_oof, oracle_test = make_base(0.0, 0.0, 0.0, 0.64)
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

# Best configs from earlier
best_configs = {
    'rw_wf064': make_base(0.0, -0.04, -0.02, 0.64),
    'rw_wf068': make_base(0.0, -0.04, -0.02, 0.68),
    'rw_wf070': make_base(0.0, -0.04, -0.02, 0.70),
    'm34_rw_wf070': make_base(0.25, -0.04, -0.02, 0.70),
    'm34_rw_wf072': make_base(0.25, -0.04, -0.02, 0.72),
}

# Add CatBoost component blend
print('\n=== CatBoost component addition to best bases ===')
print(f'{"config":35s}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')
print(f'cb standalone: OOF={mae(cb_oof):.5f}  test_mean={cb_test.mean():.3f}')

results = []
for base_name, (base_oof_arr, base_test_arr) in best_configs.items():
    base_v = mae(base_oof_arr)
    for w_cb in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]:
        b_oof = np.clip((1-w_cb)*base_oof_arr + w_cb*cb_oof, 0, None)
        b_test = np.clip((1-w_cb)*base_test_arr + w_cb*cb_test, 0, None)
        name = f'{base_name}+cb{w_cb:.2f}'
        marker = '*' if mae(b_oof) < base_oof else ''
        results.append((mae(b_oof), name, mae(b_oof)-base_oof, b_test.mean(), w_cb, base_name, b_oof, b_test))
    # Also try xgb33
    for w_xgb in [0.05, 0.10]:
        b_oof = np.clip((1-w_xgb)*base_oof_arr + w_xgb*xgb33_oof, 0, None)
        b_test = np.clip((1-w_xgb)*base_test_arr + w_xgb*xgb33_test, 0, None)
        name = f'{base_name}+xgb{w_xgb:.2f}'
        results.append((mae(b_oof), name, mae(b_oof)-base_oof, b_test.mean(), w_xgb, base_name, b_oof, b_test))

results.sort()
print(f'Top 20 (sorted by OOF):')
for oof_v, name, d_oof, tm, _, base_n, _, _ in results[:20]:
    marker = '*' if d_oof < 0 else ''
    print(f'  {name:35s}: OOF={oof_v:.5f}  {d_oof:+.6f}  test={tm:.3f} {marker}')

print(f'\nTop 10 by test_mean (OOF < base_oof):')
top_tm = sorted([r for r in results if r[2] < 0], key=lambda x: -x[3])[:10]
for oof_v, name, d_oof, tm, _, base_n, _, _ in top_tm:
    print(f'  {name:35s}: OOF={oof_v:.5f}  {d_oof:+.6f}  test={tm:.3f}')

# Generate best combo submission
if top_tm:
    oof_v, name, d_oof, tm, w_cb, base_name, b_oof_arr, b_test_arr = top_tm[0]
    print(f'\nBest by test_mean (OOF-improving): {name}')
    print(f'  OOF={oof_v:.5f}  delta={d_oof:+.6f}  test_mean={tm:.3f}')
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = b_test_arr
    fname = f'submission_{name.replace("/","_").replace("+","_")}_OOF{oof_v:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}')

# Fold-level analysis for 3 best configs
print('\n=== Fold-level analysis for key configs ===')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
val_indices = [(fi, np.sort(vi)) for fi, (_, vi) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups))]

check_configs = [
    ('oracle_NEW', oracle_oof),
    ('rw_wf068', best_configs['rw_wf068'][0]),
    ('m34_rw_wf070', best_configs['m34_rw_wf070'][0]),
    ('m34_rw_wf072', best_configs['m34_rw_wf072'][0]),
]

# Add best combo
if top_tm:
    oof_v, name, d_oof, tm, w_cb, base_name, b_oof_arr, b_test_arr = top_tm[0]
    check_configs.append((name[:25], b_oof_arr))

print(f'{"Fold":>5}', end='')
for name, _ in check_configs:
    print(f'  {name[:15]:>15s}', end='')
print()

for fi, vs in val_indices:
    print(f'{fi+1:>5}', end='')
    base_fold = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs],0,None))
    for cname, pred_arr in check_configs:
        fold_v = mean_absolute_error(y_true[vs], pred_arr[vs])
        delta = fold_v - base_fold
        print(f'  {fold_v:>7.5f}({delta:>+7.5f})', end='')
    print()

print(f'{"Global":>5}', end='')
for cname, pred_arr in check_configs:
    print(f'  {mae(pred_arr):>7.5f}({mae(pred_arr)-base_oof:>+7.5f})', end='')
print()

# Summary table for quick decision
print('\n=== FINAL DECISION MATRIX ===')
print(f'{"config":40s}  {"OOF":>9}  {"ΔOOF":>9}  {"test":>8}  {"Δtest":>7}  {"Strategy"}')
print(f'  {"oracle_NEW (best LB=9.7527)":40s}  {base_oof:.5f}  {"±0":>9s}  {oracle_test.mean():.3f}  {"±0":>7s}  baseline')
for name, (oo, ot) in best_configs.items():
    d_oof = mae(oo) - base_oof
    d_test = ot.mean() - oracle_test.mean()
    strat = 'rw' if 'rw' in name and 'm34' not in name else 'm34+rw' if 'm34' in name else 'orig'
    print(f'  {name:40s}  {mae(oo):.5f}  {d_oof:>+9.6f}  {ot.mean():.3f}  {d_test:>+7.4f}  {strat}')

print('\nDone.')
