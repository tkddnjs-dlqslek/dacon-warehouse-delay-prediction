"""
Quick check: do older oracle variants (seqC, seqC_v2, seqC_v3, etc.) add value?
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_ls = [ls_pos[i] for i in train_raw['ID'].values]
mega_oof  = d['meta_avg_oof'][id_to_ls]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
y_true = train_raw['avg_delay_minutes_next_30m'].values
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
res_fixed = y_true - fixed_oof
print(f"FIXED MAE: {fixed_mae:.4f}")

oracle_files = [
    ('seqC',       'results/oracle_seq/oof_seqC.npy'),
    ('seqC_v2',    'results/oracle_seq/oof_seqC_v2.npy'),
    ('seqC_v3',    'results/oracle_seq/oof_seqC_v3.npy'),
    ('seqC_raw',   'results/oracle_seq/oof_seqC_raw.npy'),
    ('seqC_log',   'results/oracle_seq/oof_seqC_log.npy'),
    ('seqC_rlag',  'results/oracle_seq/oof_seqC_ranklag.npy'),
    ('xgb',        'results/oracle_seq/oof_seqC_xgb.npy'),
    ('lv2',        'results/oracle_seq/oof_seqC_log_v2.npy'),
]

print(f"\n{'Name':<12} {'MAE':>8} {'res_corr':>10} {'xgb_corr':>10} {'lv2_corr':>10}")
xgb_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
for name, path in oracle_files:
    if not os.path.exists(path):
        print(f"  {name}: MISSING")
        continue
    arr = np.load(path)
    m = np.mean(np.abs(arr - y_true))
    c = np.corrcoef(res_fixed, arr - fixed_oof)[0,1]
    cx = np.corrcoef(arr, xgb_oof)[0,1]
    cl = np.corrcoef(arr, lv2_oof)[0,1]
    print(f"  {name:<12} {m:>8.4f} {c:>10.4f} {cx:>10.4f} {cl:>10.4f}")
    # Best static blend
    best_m = fixed_mae; best_w = 0
    for w in np.arange(0, 0.61, 0.04):
        bl = (1-w)*fixed_oof + w*arr
        mm = np.mean(np.abs(bl - y_true))
        if mm < best_m: best_m = mm; best_w = w
    print(f"    best_blend_w={best_w:.2f}  blend_MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}")

print("\nDone.")
