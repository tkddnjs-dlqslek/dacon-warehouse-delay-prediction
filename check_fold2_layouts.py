import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold

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

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
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
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof, 0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_oof, oracle_test = make_pred(0.0, 0.0, 0.0, 0.64, 0.0)
base_oof = mae(oracle_oof)

# Find fold assignments
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_info = {}
for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    layouts = train_raw.iloc[np.sort(val_idx)]['layout_id'].unique()
    pack_means = train_raw[train_raw['layout_id'].isin(layouts)].groupby('layout_id')['pack_utilization'].mean()
    y_means = train_raw[train_raw['layout_id'].isin(layouts)].groupby('layout_id')['avg_delay_minutes_next_30m'].mean()
    fold_info[fi] = {
        'layouts': sorted(layouts),
        'n_layouts': len(layouts),
        'n_rows': len(val_idx),
        'pack_mean': pack_means.mean(),
        'pack_max': pack_means.max(),
        'y_mean': y_means.mean(),
        'y_max': y_means.max()
    }

print('=== Fold characteristics ===')
print(f'{"Fold":>5}  {"n_lay":>6}  {"n_rows":>6}  {"pack_mean":>10}  {"pack_max":>10}  {"y_mean":>8}  {"y_max":>8}  {"oracle_OOF":>10}')
for fi in range(5):
    info = fold_info[fi]
    val_idx = np.sort([idx for idx in range(len(train_raw)) if train_raw.iloc[idx]['layout_id'] in info['layouts']])
    oracle_fold = np.mean(np.abs(np.clip(oracle_oof[val_idx],0,None) - y_true[val_idx]))
    print(f'{fi+1:>5}  {info["n_layouts"]:>6}  {info["n_rows"]:>6}  {info["pack_mean"]:>10.4f}  {info["pack_max"]:>10.4f}  {info["y_mean"]:>8.2f}  {info["y_max"]:>8.2f}  {oracle_fold:>10.5f}')

# Focus on fold 2 (high OOF fold)
print('\n=== Fold 2 top layouts by pack_mean ===')
f2_layouts = fold_info[1]['layouts']
f2_stats = train_raw[train_raw['layout_id'].isin(f2_layouts)].groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
).reset_index().sort_values('pack_mean', ascending=False)
for _, r in f2_stats.head(15).iterrows():
    print(f'  {r["layout_id"]:12s}: pack={r["pack_mean"]:.4f}  y_mean={r["y_mean"]:.2f}')

# Test layout comparison
print('\n=== Test layouts by pack_mean ===')
test_stats = test_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean')
).reset_index().sort_values('pack_mean', ascending=False)
print(f'  Top 15 test layouts:')
for _, r in test_stats.head(15).iterrows():
    print(f'  {r["layout_id"]:12s}: pack={r["pack_mean"]:.4f}  inflow={r["inflow_mean"]:.1f}')

# Key comparison: fold 2 vs test distributions
print(f'\nFold 2 pack_max: {fold_info[1]["pack_max"]:.4f}')
print(f'Test pack_max: {test_stats["pack_mean"].max():.4f}')
print(f'Fold 2 y_max: {fold_info[1]["y_max"]:.2f}')

# Generate the best candidate submissions not yet saved
sub = pd.read_csv('sample_submission.csv')
new_subs = [
    ('m34_rw_wf070_cb005', 0.25, -0.04, -0.02, 0.70, 0.05),
    ('m34_rw_wf070_cb010', 0.25, -0.04, -0.02, 0.70, 0.10),
    ('m34_rw_wf072_cb010', 0.25, -0.04, -0.02, 0.72, 0.10),
    ('m34_rw_wf072_cb012', 0.25, -0.04, -0.02, 0.72, 0.12),
]
for name, w34, dr2, dr3, wf, w_cb in new_subs:
    oo, ot = make_pred(w34, dr2, dr3, wf, w_cb)
    oof_v = mae(oo)
    sub['avg_delay_minutes_next_30m'] = ot
    fname = f'submission_{name}_OOF{oof_v:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}  OOF={oof_v:.5f}  delta={oof_v-base_oof:+.6f}  test_mean={ot.mean():.3f}')

# Check if rw_wf070 (no m34) was saved
oo70, ot70 = make_pred(0.0, -0.04, -0.02, 0.70, 0.0)
print(f'\nrw_wf070: OOF={mae(oo70):.5f}  test_mean={ot70.mean():.3f}')
sub['avg_delay_minutes_next_30m'] = ot70
sub.to_csv(f'submission_rw_wf070_OOF{mae(oo70):.5f}.csv', index=False)
print(f'Saved: submission_rw_wf070_OOF{mae(oo70):.5f}.csv')

print('\nDone.')
