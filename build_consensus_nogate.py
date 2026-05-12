"""HH5 consensus selection WITHOUT gate — no overfitting"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

sample_sub = pd.read_csv('sample_submission.csv')
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)

KEYS = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
ct = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))

# oracle_NEW for comparison
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*co['mega33']
           + fw['rank_adj']*co['rank_adj']
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_new_oof = 0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
print(f"oracle_NEW OOF: {mae(oracle_new_oof):.5f}")

# HH5 consensus selection (no gate)
stacked_oof = np.stack([np.clip(co[k],0,None) for k in KEYS], axis=1)
stacked_te  = np.stack([np.clip(ct[k],0,None) for k in KEYS], axis=1)
median_oof = np.median(stacked_oof, axis=1, keepdims=True)
median_te  = np.median(stacked_te,  axis=1, keepdims=True)
dist_oof = np.abs(stacked_oof - median_oof)
dist_te  = np.abs(stacked_te  - median_te)
best_comp_oof = np.argmin(dist_oof, axis=1)
best_comp_te  = np.argmin(dist_te,  axis=1)
bo = stacked_oof[np.arange(len(stacked_oof)), best_comp_oof]
bt = stacked_te [np.arange(len(stacked_te)),  best_comp_te]

oof_val = mae(bo)
print(f"consensus_nogate OOF: {oof_val:.5f}")

# Component distribution check
print("\nComponent usage (OOF):")
for i, k in enumerate(KEYS):
    cnt = int(np.sum(best_comp_oof==i))
    print(f"  {k}: {cnt} samples ({100*cnt/len(bo):.1f}%)")

# Save
fname = f'submission_consensus_nogate_OOF{oof_val:.5f}.csv'
sub = np.maximum(0, bt)
df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
df.to_csv(fname, index=False)
print(f"\nSaved: {fname}")
print("Done.")
