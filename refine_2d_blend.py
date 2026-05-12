"""
Refine the 2D per-pos blend with STEP=0.01 (finer than original STEP=0.04).
Uses existing xgb and lv2 oracles + current 2D weights as starting point.
Might find small improvement from finer resolution.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

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
id_to_ls = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[i] for i in test_raw['ID'].values]

mega_oof  = d['meta_avg_oof'][id_to_ls]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls]
i1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
i2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
i3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]
fw = dict(mega33=0.7637,rank_adj=0.1589,iter_r1=0.0119,iter_r2=0.0346,iter_r3=0.0310)
fixed = fw['mega33']*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*i1_oof+fw['iter_r2']*i2_oof+fw['iter_r3']*i3_oof
y = train_raw['avg_delay_minutes_next_30m'].values
rsc = train_raw['row_in_sc'].values
fixed_mae = np.mean(np.abs(fixed - y))

xgb = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
xgb_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test = np.load('results/oracle_seq/test_C_log_v2.npy')

mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
i1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
i2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
i3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
fixed_test = fw['mega33']*mega_test+fw['rank_adj']*rank_test+fw['iter_r1']*i1_test+fw['iter_r2']*i2_test+fw['iter_r3']*i3_test

# Coarse best weights from 2D OOF8.3800 submission (original STEP=0.04)
coarse_w = {
    0:(0.00,0.24), 1:(0.00,0.20), 2:(0.12,0.16), 3:(0.32,0.00),
    4:(0.00,0.00), 5:(0.00,0.00), 6:(0.36,0.00), 7:(0.00,0.00),
    8:(0.20,0.00), 9:(0.24,0.12), 10:(0.00,0.32), 11:(0.40,0.00),
    12:(0.00,0.24), 13:(0.40,0.00), 14:(0.12,0.12), 15:(0.16,0.24),
    16:(0.16,0.32), 17:(0.08,0.52), 18:(0.00,0.00), 19:(0.00,0.52),
    20:(0.00,0.28), 21:(0.48,0.00), 22:(0.00,0.00), 23:(0.48,0.00),
    24:(0.40,0.20)
}

STEP = 0.01
SEARCH_RADIUS = 0.06  # search ±0.06 around coarse best

print("Refining 2D per-pos weights (STEP=0.01)...", flush=True)
per_pos_fine = {}
for pos in range(25):
    mask = rsc == pos
    f_pos = fixed[mask]; y_pos = y[mask]
    x_pos = xgb[mask]; l_pos = lv2[mask]
    w0_coarse, w1_coarse = coarse_w[pos]

    best_m = np.mean(np.abs((1-w0_coarse-w1_coarse)*f_pos + w0_coarse*x_pos + w1_coarse*l_pos - y_pos))
    best_w = (w0_coarse, w1_coarse)

    # Fine search around coarse best
    w0_lo = max(0.0, w0_coarse - SEARCH_RADIUS)
    w0_hi = min(0.60, w0_coarse + SEARCH_RADIUS)
    w1_lo = max(0.0, w1_coarse - SEARCH_RADIUS)

    for w0 in np.arange(w0_lo, w0_hi + STEP/2, STEP):
        for w1 in np.arange(w1_lo, min(0.60-w0+STEP/2, w1_coarse+SEARCH_RADIUS+STEP/2), STEP):
            if w0 + w1 > 0.60: continue
            bl = (1-w0-w1)*f_pos + w0*x_pos + w1*l_pos
            m = np.mean(np.abs(bl - y_pos))
            if m < best_m:
                best_m = m; best_w = (w0, w1)

    per_pos_fine[pos] = best_w
    delta = best_m - np.mean(np.abs(f_pos - y_pos))
    w_str = f"wXGB={best_w[0]:.2f} wLv2={best_w[1]:.2f}"
    print(f"  pos {pos:2d}: {w_str}  delta={delta:+.4f}", flush=True)

# Full OOF blend with refined weights
blend_oof = fixed.copy()
for pos in range(25):
    mask = rsc == pos
    w0, w1 = per_pos_fine[pos]
    blend_oof[mask] = (1-w0-w1)*fixed[mask] + w0*xgb[mask] + w1*lv2[mask]
blend_mae = np.mean(np.abs(blend_oof - y))
print(f"\nRefined 2D blend OOF: {blend_mae:.4f}  (vs coarse 8.3800  delta={blend_mae-8.3800:+.4f})", flush=True)

# 5-fold check
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
fold_deltas = []
for _, val_idx in gkf.split(np.arange(len(train_raw)), groups=groups):
    b = blend_oof[val_idx]; f = fixed[val_idx]
    fold_deltas.append(np.mean(np.abs(b-y[val_idx])) - np.mean(np.abs(f-y[val_idx])))
print(f"Fold deltas: {[f'{x:.4f}' for x in fold_deltas]} ({sum(x<0 for x in fold_deltas)}/5 neg)", flush=True)

PREV_BEST = 8.3800
if blend_mae < PREV_BEST - 0.0001 and sum(x < 0 for x in fold_deltas) >= 4:
    # Build test blend
    test_rsc = test_raw['row_in_sc'].values
    blend_test = fixed_test.copy()
    for pos in range(25):
        mask = test_rsc == pos
        w0, w1 = per_pos_fine[pos]
        blend_test[mask] = (1-w0-w1)*fixed_test[mask] + w0*xgb_test[mask] + w1*lv2_test[mask]
    blend_test = np.maximum(0, blend_test)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': blend_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_posaware_2D_refined_OOF{blend_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo significant improvement from refinement (threshold: {PREV_BEST-0.0001:.4f})", flush=True)

print("Done.", flush=True)
