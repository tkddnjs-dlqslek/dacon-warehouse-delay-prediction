"""
Quick per-position analysis: for each new oracle that exists,
check per-pos MAE delta vs FIXED. Helps identify which positions benefit.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
y = train_raw['avg_delay_minutes_next_30m'].values
rsc = train_raw['row_in_sc'].values

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_ls = [ls_pos[i] for i in train_raw['ID'].values]
mega = d['meta_avg_oof'][id_to_ls]
rank = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls]
i1 = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
i2 = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
i3 = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]
fw = dict(mega33=0.7637,rank_adj=0.1589,iter_r1=0.0119,iter_r2=0.0346,iter_r3=0.0310)
fixed = fw['mega33']*mega+fw['rank_adj']*rank+fw['iter_r1']*i1+fw['iter_r2']*i2+fw['iter_r3']*i3

oracles = {}
for name, path in [
    ('xgb',      'results/oracle_seq/oof_seqC_xgb.npy'),
    ('lv2',      'results/oracle_seq/oof_seqC_log_v2.npy'),
    ('rf_log',   'results/oracle_seq/oof_seqC_rf_log.npy'),
    ('cumstats', 'results/oracle_seq/oof_seqC_cumstats.npy'),
    ('xgb_log2', 'results/oracle_seq/oof_seqC_xgb_log2.npy'),
    ('et_log',   'results/oracle_seq/oof_seqC_et_log.npy'),
    ('latepos',  'results/oracle_seq/oof_seqC_lgb_latepos.npy'),
]:
    if os.path.exists(path):
        oracles[name] = np.load(path)
        m = np.mean(np.abs(oracles[name] - y))
        print(f"  {name}: OOF MAE={m:.4f}", flush=True)

print(f"FIXED OOF: {np.mean(np.abs(fixed-y)):.4f}\n")

for name, arr in oracles.items():
    print(f"\n--- {name} per-position delta vs FIXED ---")
    for pos in range(25):
        mask = rsc == pos
        f_mae = np.mean(np.abs(fixed[mask] - y[mask]))
        o_mae = np.mean(np.abs(arr[mask] - y[mask]))
        delta = o_mae - f_mae
        marker = ' ***' if delta < -0.02 else (' +' if delta < 0 else '')
        print(f"  pos {pos:2d}: fixed={f_mae:.3f}  oracle={o_mae:.3f}  delta={delta:+.4f}{marker}")
