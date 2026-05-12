"""
Analyze RF oracle and find best per-position blend (FIXED + XGB + Lv2 + RF).
Run after train_oracle_rf.py completes.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()

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

mega_oof  = d['meta_avg_oof'][id_to_lspos]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos]
xgb_oof   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rf_oof    = np.load('results/oracle_seq/oof_seqC_rf.npy')
y_true    = train_raw['avg_delay_minutes_next_30m'].values

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)
print(f"oracle-XGB MAE: {np.mean(np.abs(xgb_oof-y_true)):.4f}", flush=True)
print(f"oracle-Lv2 MAE: {np.mean(np.abs(lv2_oof-y_true)):.4f}", flush=True)
print(f"oracle-RF MAE:  {np.mean(np.abs(rf_oof-y_true)):.4f}", flush=True)

res_fixed = y_true - fixed_oof
corr_xgb = np.corrcoef(res_fixed, xgb_oof - fixed_oof)[0,1]
corr_lv2 = np.corrcoef(res_fixed, lv2_oof - fixed_oof)[0,1]
corr_rf  = np.corrcoef(res_fixed, rf_oof - fixed_oof)[0,1]
print(f"\nResidual corr vs FIXED: XGB={corr_xgb:.4f}  Lv2={corr_lv2:.4f}  RF={corr_rf:.4f}", flush=True)
print(f"RF-XGB pred corr: {np.corrcoef(rf_oof, xgb_oof)[0,1]:.4f}", flush=True)
print(f"RF-Lv2 pred corr: {np.corrcoef(rf_oof, lv2_oof)[0,1]:.4f}", flush=True)

# Per-position analysis for RF
row_in_sc = train_raw['row_in_sc'].values
print("\nPer-position RF delta vs FIXED:", flush=True)
rf_pos_deltas = {}
for pos in range(25):
    mask = row_in_sc == pos
    n = mask.sum()
    mae_f = np.mean(np.abs(fixed_oof[mask] - y_true[mask]))
    mae_r = np.mean(np.abs(rf_oof[mask] - y_true[mask]))
    rf_pos_deltas[pos] = mae_r - mae_f
    print(f"  pos={pos:2d}: n={n:4d}  FIXED={mae_f:.4f}  RF={mae_r:.4f}  delta={mae_r-mae_f:+.4f}", flush=True)

# Per-position 3-way grid: optimize (wXGB, wLV2, wRF) per position
print("\nPer-position 3-way optimization (FIXED+XGB+Lv2+RF)...", flush=True)
STEP = 0.04
per_pos_w3 = {}
for pos in range(25):
    mask = row_in_sc == pos
    f_pos = fixed_oof[mask]; x_pos = xgb_oof[mask]; l_pos = lv2_oof[mask]; r_pos = rf_oof[mask]; y_pos = y_true[mask]
    best_m = np.mean(np.abs(f_pos - y_pos)); best_w = (0.0, 0.0, 0.0)
    for wX in np.arange(0, 0.65, STEP):
        for wL in np.arange(0, 0.65-wX, STEP):
            for wR in np.arange(0, 0.65-wX-wL, STEP):
                if wX+wL+wR > 0.60: continue
                bl = (1-wX-wL-wR)*f_pos + wX*x_pos + wL*l_pos + wR*r_pos
                m = np.mean(np.abs(bl - y_pos))
                if m < best_m:
                    best_m = m; best_w = (wX, wL, wR)
    per_pos_w3[pos] = best_w
    print(f"  pos={pos:2d}: wXGB={best_w[0]:.2f} wLv2={best_w[1]:.2f} wRF={best_w[2]:.2f}  delta={best_m-np.mean(np.abs(f_pos-y_pos)):+.4f}", flush=True)

# Compute full OOF for 3-way per-pos
blend3_oof = fixed_oof.copy()
for pos in range(25):
    mask = row_in_sc == pos
    wX, wL, wR = per_pos_w3[pos]
    blend3_oof[mask] = (1-wX-wL-wR)*fixed_oof[mask] + wX*xgb_oof[mask] + wL*lv2_oof[mask] + wR*rf_oof[mask]
blend3_mae = np.mean(np.abs(blend3_oof - y_true))
print(f"\n3-way per-pos OOF MAE: {blend3_mae:.4f}  delta={blend3_mae-fixed_mae:+.4f}", flush=True)

# 5-fold CV for generalization check
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
folds3 = []
for _, val_idx in gkf.split(np.arange(len(train_raw)), groups=groups):
    bl_val = blend3_oof[val_idx]
    f_val  = fixed_oof[val_idx]
    delta  = np.mean(np.abs(bl_val - y_true[val_idx])) - np.mean(np.abs(f_val - y_true[val_idx]))
    folds3.append(delta)
print(f"Fold deltas: {[f'{x:.4f}' for x in folds3]} ({sum(x<0 for x in folds3)}/5 neg)", flush=True)

# Compare with 2D best (no RF)
per_pos_w2 = {
    0:(0.00,0.24), 1:(0.00,0.20), 2:(0.12,0.16), 3:(0.32,0.00),
    4:(0.00,0.00), 5:(0.00,0.00), 6:(0.36,0.00), 7:(0.00,0.00),
    8:(0.20,0.00), 9:(0.24,0.12), 10:(0.00,0.32), 11:(0.40,0.00),
    12:(0.00,0.24), 13:(0.40,0.00), 14:(0.12,0.12), 15:(0.16,0.24),
    16:(0.16,0.32), 17:(0.08,0.52), 18:(0.00,0.00), 19:(0.00,0.52),
    20:(0.00,0.28), 21:(0.48,0.00), 22:(0.00,0.00), 23:(0.48,0.00),
    24:(0.40,0.20)
}
blend2_oof = fixed_oof.copy()
for pos in range(25):
    mask = row_in_sc == pos
    wX, wL = per_pos_w2[pos]
    blend2_oof[mask] = (1-wX-wL)*fixed_oof[mask] + wX*xgb_oof[mask] + wL*lv2_oof[mask]
blend2_mae = np.mean(np.abs(blend2_oof - y_true))
print(f"\n2D per-pos OOF MAE (no RF): {blend2_mae:.4f}", flush=True)
print(f"3D per-pos OOF MAE (with RF): {blend3_mae:.4f}  improvement={blend2_mae-blend3_mae:+.4f}", flush=True)

CURRENT_BEST = 8.3800
if blend3_mae < CURRENT_BEST - 0.0002 and sum(x < 0 for x in folds3) >= 4:
    # Build test predictions
    mega_test  = d['meta_avg_test'][te_id_to_ls]
    rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
    iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
    iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
    iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
    xgb_test   = np.load('results/oracle_seq/test_C_xgb.npy')
    lv2_test   = np.load('results/oracle_seq/test_C_log_v2.npy')
    rf_test    = np.load('results/oracle_seq/test_C_rf.npy')
    fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                  fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)

    test_row_sc = test_raw['row_in_sc'].values
    blend3_test = fixed_test.copy()
    for pos in range(25):
        mask = test_row_sc == pos
        wX, wL, wR = per_pos_w3[pos]
        blend3_test[mask] = (1-wX-wL-wR)*fixed_test[mask] + wX*xgb_test[mask] + wL*lv2_test[mask] + wR*rf_test[mask]
    blend3_test = np.maximum(0, blend3_test)

    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': blend3_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_posaware_3D_OOF{blend3_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo improvement from 3D blend ({blend3_mae:.4f} vs current best {CURRENT_BEST:.4f})", flush=True)

print("Done.", flush=True)
