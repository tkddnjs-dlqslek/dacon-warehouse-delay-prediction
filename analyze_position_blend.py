"""
Position-aware blend analysis: oracle models have more accurate lags at later positions.
Test whether weighting oracle contributions by position improves OOF.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
test_raw['row_in_sc'] = test_raw.groupby(['layout_id','scenario_id']).cumcount()

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[i] for i in test_raw['ID'].values]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)

train_id2 = pd.read_csv('train.csv').copy()
train_id2['_row_id'] = train_id2['ID'].str.replace('TRAIN_','').astype(int)
train_id2 = train_id2.sort_values('_row_id').reset_index(drop=True)
test_id2 = pd.read_csv('test.csv').copy()
test_id2['_row_id'] = test_id2['ID'].str.replace('TEST_','').astype(int)
test_id2 = test_id2.sort_values('_row_id').reset_index(drop=True)

ls_pos2 = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos2 = [ls_pos2[rid] for rid in train_id2['ID'].values]
te_ls_pos2 = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls2 = [te_ls_pos2[rid] for rid in test_id2['ID'].values]

mega_oof  = d['meta_avg_oof'][id_to_lspos2]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos2]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos2]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos2]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos2]
xgb_oof   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
y_true2   = train_id2['avg_delay_minutes_next_30m'].values

fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
fixed_mae = np.mean(np.abs(fixed_oof - y_true2))

# 5-way blend (current best)
blend5_oof = 0.68*fixed_oof + 0.12*xgb_oof + 0.20*lv2_oof
blend5_mae = np.mean(np.abs(blend5_oof - y_true2))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)
print(f"5-way blend OOF MAE: {blend5_mae:.4f}", flush=True)

# row_in_sc for the ID-order training data
# train_id2 is sorted by _row_id, so we need row_in_sc in same order
train_id2_tmp = pd.read_csv('train.csv')
train_id2_tmp['_row_id'] = train_id2_tmp['ID'].str.replace('TRAIN_','').astype(int)
train_id2_tmp = train_id2_tmp.sort_values('_row_id').reset_index(drop=True)
train_id2_tmp['row_in_sc'] = train_id2_tmp.groupby(['layout_id','scenario_id']).cumcount()
row_in_sc = train_id2_tmp['row_in_sc'].values

print("\nPer-position analysis (FIXED vs 5-way):", flush=True)
# For each position 0-24, compute average prediction error
for pos in range(25):
    mask = row_in_sc == pos
    n = mask.sum()
    if n == 0: continue
    mae_f = np.mean(np.abs(fixed_oof[mask] - y_true2[mask]))
    mae_b = np.mean(np.abs(blend5_oof[mask] - y_true2[mask]))
    print(f"  pos={pos:2d}: n={n:5d}  FIXED={mae_f:.4f}  blend5={mae_b:.4f}  delta={mae_b-mae_f:.4f}", flush=True)

# Position-aware blend: vary oracle weight by position
# Higher oracle weight at later positions (more lag history)
print("\nPosition-aware blend optimization...", flush=True)
# Oracle blend: at each position, use a different weight wO
# blend_aware[i] = (1 - w[pos_i]) * fixed_oof[i] + w[pos_i] * oracle_blend[i]
# where oracle_blend = optimal combination of XGB + Lv2

# First find optimal static weights for oracle components
oracle_combined = 0.375*xgb_oof + 0.625*lv2_oof  # ratio from 5-way: 0.12/(0.12+0.20)=0.375

# Position-dependent weight search
best_pos_mae = 9999
best_pos_weights = None
# Try different position-weight functions
# Simple: linear from w_min at pos=0 to w_max at pos=24
for w_min in np.arange(0, 0.25, 0.04):
    for w_max in np.arange(w_min, 0.55, 0.04):
        # Linear interpolation of oracle weight by position
        pos_weights = np.zeros(len(y_true2))
        for pos in range(25):
            mask = row_in_sc == pos
            w_pos = w_min + (w_max - w_min) * pos / 24
            pos_weights[mask] = w_pos
        blend_aware = (1 - pos_weights) * fixed_oof + pos_weights * oracle_combined
        m = np.mean(np.abs(blend_aware - y_true2))
        if m < best_pos_mae:
            best_pos_mae = m
            best_pos_weights = (w_min, w_max)

print(f"Best position-aware blend: w_min={best_pos_weights[0]:.2f}, w_max={best_pos_weights[1]:.2f}, MAE={best_pos_mae:.4f}, delta={best_pos_mae-fixed_mae:.4f}", flush=True)
print(f"vs static blend5: {blend5_mae:.4f} (diff={best_pos_mae-blend5_mae:.4f})", flush=True)

# Fold analysis for position-aware blend
w_min_best, w_max_best = best_pos_weights
gkf = GroupKFold(n_splits=5)
groups_id = train_id2['layout_id'].values
row_in_sc_train = row_in_sc

folds_pa = []
for _, val_idx in gkf.split(np.arange(len(train_id2)), groups=groups_id):
    pos_weights_val = np.zeros(len(val_idx))
    for pos in range(25):
        mask = row_in_sc_train[val_idx] == pos
        w_pos = w_min_best + (w_max_best - w_min_best) * pos / 24
        pos_weights_val[mask] = w_pos
    blend_val = (1 - pos_weights_val) * fixed_oof[val_idx] + pos_weights_val * oracle_combined[val_idx]
    delta = np.mean(np.abs(blend_val - y_true2[val_idx])) - np.mean(np.abs(fixed_oof[val_idx] - y_true2[val_idx]))
    folds_pa.append(delta)
print(f"Fold deltas (position-aware): {[f'{x:.4f}' for x in folds_pa]} ({sum(x<0 for x in folds_pa)}/5 neg)", flush=True)

# If position-aware is better than static, generate submission
mega_test_b  = d['meta_avg_test'][te_id_to_ls2]
rank_test_b  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls2]
iter1_test_b = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls2]
iter2_test_b = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls2]
iter3_test_b = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls2]
xgb_test_b   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test_b   = np.load('results/oracle_seq/test_C_log_v2.npy')
fixed_test_b = (fw['mega33']*mega_test_b + fw['rank_adj']*rank_test_b +
               fw['iter_r1']*iter1_test_b + fw['iter_r2']*iter2_test_b + fw['iter_r3']*iter3_test_b)

CURRENT_BEST = 8.3831
if best_pos_mae < CURRENT_BEST - 0.0003 and sum(x < 0 for x in folds_pa) >= 4:
    oracle_test_combined = 0.375*xgb_test_b + 0.625*lv2_test_b
    pos_weights_test = np.zeros(len(test_id2))
    for pos in range(25):
        mask = test_raw['row_in_sc'].values == pos
        w_pos = w_min_best + (w_max_best - w_min_best) * pos / 24
        pos_weights_test[mask] = w_pos
    blend_test = np.maximum(0, (1 - pos_weights_test) * fixed_test_b + pos_weights_test * oracle_test_combined)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id2['ID'].values, 'avg_delay_minutes_next_30m': blend_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_posaware_OOF{best_pos_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST (position-aware)! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo improvement from position-aware blend ({best_pos_mae:.4f} vs current {CURRENT_BEST:.4f})", flush=True)

print("Done.", flush=True)
