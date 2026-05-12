"""
Evaluate all available oracle OOF files and find optimal blend with FIXED.
All arrays aligned to _row_id order (same as train_raw sorted by _row_id).
Run this after any oracle training completes.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_to_pos[i] for i in train_raw['ID'].values]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

fixed2 = fw['mega33']*d['meta_avg_oof'][id2]
fixed2 += fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
fixed2 += fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
fixed2 += fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
fixed2 += fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2]
fixed_mae = np.mean(np.abs(fixed2 - y_true))
print(f"FIXED OOF MAE: {fixed_mae:.4f}")

ORACLE_FILES = {
    'xgb':        'results/oracle_seq/oof_seqC_xgb.npy',
    'lv2':        'results/oracle_seq/oof_seqC_log_v2.npy',
    'remaining':  'results/oracle_seq/oof_seqC_xgb_remaining.npy',
    'lgb_rem':    'results/oracle_seq/oof_seqC_lgb_remaining.npy',
    'lgb_rem_v3': 'results/oracle_seq/oof_seqC_lgb_remaining_v3.npy',
    'layout':     'results/oracle_seq/oof_seqC_xgb_layout.npy',
    'layout_v2':  'results/oracle_seq/oof_seqC_xgb_layout_v2.npy',
    'cumulative': 'results/oracle_seq/oof_seqC_xgb_cumulative.npy',
    'dual':       'results/oracle_seq/oof_seqC_lgb_dual.npy',
    'stack':      'results/oracle_seq/oof_seqC_lgb_stack.npy',
    'residual':   'results/oracle_seq/oof_seqC_xgb_residual.npy',
    'lgb_sc':     'results/oracle_seq/oof_seqC_lgb_sc_only.npy',
    'latepos':    'results/oracle_seq/oof_seqC_lgb_latepos.npy',
    'lgb_log1':   'results/oracle_seq/oof_seqC_lgb_log1.npy',
    'xgb_lag1':   'results/oracle_seq/oof_seqC_xgb_lag1.npy',
    'combined':   'results/oracle_seq/oof_seqC_xgb_combined.npy',
    'xgb_v31':    'results/oracle_seq/oof_seqC_xgb_v31.npy',
    'cb':         'results/oracle_seq/oof_seqC_cb.npy',
}

oracles = {}
for name, path in ORACLE_FILES.items():
    if os.path.exists(path):
        arr = np.load(path)
        mae = np.mean(np.abs(arr - y_true))
        corr_fixed = np.corrcoef(arr, fixed2)[0,1]
        oracles[name] = arr
        print(f"  {name:14s}: OOF={mae:.4f}  corr_fixed={corr_fixed:.4f}")

# Current best (LB 9.7527)
if 'xgb' in oracles and 'lv2' in oracles and 'remaining' in oracles:
    best4 = 0.64*fixed2 + 0.12*oracles['xgb'] + 0.16*oracles['lv2'] + 0.08*oracles['remaining']
    print(f"\nCurrent best (0.64F+0.12X+0.16L+0.08R): {np.mean(np.abs(best4-y_true)):.4f}")
elif 'xgb' in oracles and 'lv2' in oracles:
    best4 = 0.68*fixed2 + 0.12*oracles['xgb'] + 0.20*oracles['lv2']
    print(f"\nbase5 (0.68F+0.12X+0.20L): {np.mean(np.abs(best4-y_true)):.4f}")

# Pairwise correlations between oracle models
print("\n=== Oracle-Oracle correlations (non-lag models) ===")
non_lag = {k:v for k,v in oracles.items() if k not in ('latepos','lgb_log1','xgb_lag1')}
names = list(non_lag.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        c = np.corrcoef(non_lag[names[i]], non_lag[names[j]])[0,1]
        if c < 0.975:
            print(f"  corr({names[i]:14s}, {names[j]:14s}) = {c:.4f}  *** DIVERSE")

print("\n=== Pairwise: FIXED + each oracle ===")
for name, arr in oracles.items():
    best_m = fixed_mae; best_w = 0
    for w in np.arange(0.02, 0.51, 0.02):
        mm = np.mean(np.abs((1-w)*fixed2 + w*arr - y_true))
        if mm < best_m: best_m = mm; best_w = w
    print(f"  FIXED+{name:14s}: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-fixed_mae:+.4f}")

# 3-way from current best
print("\n=== 3-way: best4 + each new oracle ===")
best4_mae = np.mean(np.abs(best4 - y_true))
for name, arr in oracles.items():
    if name in ('xgb','lv2','remaining'): continue
    best_m = best4_mae; best_w = 0
    for w in np.arange(0.02, 0.21, 0.02):
        mm = np.mean(np.abs((1-w)*best4 + w*arr - y_true))
        if mm < best_m: best_m = mm; best_w = w
    delta = best_m - best4_mae
    print(f"  best4+{name:14s}: w={best_w:.2f} MAE={best_m:.4f} delta={delta:+.4f}")

print("\nDone.")
