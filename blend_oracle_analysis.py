"""
Blend analysis: oracle-C + FIXED components, all in ID order.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json

print("Loading...", flush=True)
train_id = pd.read_csv('train.csv').copy()
train_id['_row_id'] = train_id['ID'].str.replace('TRAIN_','').astype(int)
train_id = train_id.sort_values('_row_id').reset_index(drop=True)
y_true = train_id['avg_delay_minutes_next_30m'].values

test_id = pd.read_csv('test.csv').copy()
test_id['_row_id'] = test_id['ID'].str.replace('TEST_','').astype(int)
test_id = test_id.sort_values('_row_id').reset_index(drop=True)

# LS order mappings
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_pos[rid] for rid in train_id['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[rid] for rid in test_id['ID'].values]

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

# OOFs in ID order
mega_oof  = d['meta_avg_oof'][id_to_lspos]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos]
oracC_oof = np.load('results/oracle_seq/oof_seqC.npy')  # already in ID order

# Test in ID order
mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
oracC_test = np.load('results/oracle_seq/test_C.npy')   # already in ID order

# Verify OOFs
print(f"mega33   OOF MAE: {np.mean(np.abs(mega_oof - y_true)):.4f}  (exp 8.3989)")
print(f"rank_adj OOF MAE: {np.mean(np.abs(rank_oof - y_true)):.4f}  (exp 8.5041)")
print(f"iter1    OOF MAE: {np.mean(np.abs(iter1_oof - y_true)):.4f}  (exp 8.5332)")
print(f"oracle-C OOF MAE: {np.mean(np.abs(oracC_oof - y_true)):.4f}")

# FIXED weights
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)

fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"\nFIXED OOF MAE: {fixed_mae:.4f}  (exp 8.3935)")

mega_mae = np.mean(np.abs(mega_oof - y_true))
oracC_mae = np.mean(np.abs(oracC_oof - y_true))
corr_C_M = np.corrcoef(oracC_oof - y_true, mega_oof - y_true)[0,1]
corr_C_F = np.corrcoef(oracC_oof - y_true, fixed_oof - y_true)[0,1]

print(f"\noracle-C OOF MAE: {oracC_mae:.4f}  corr(oracle-C, mega33)={corr_C_M:.4f}  corr(oracle-C, FIXED)={corr_C_F:.4f}")
print(f"blend threshold (vs mega33): {mega_mae + 1.25*(1-corr_C_M):.4f}")

# 2-way blend: oracle-C vs mega33
best_m, best_w = 9999, 0
for w in np.arange(0, 1.01, 0.02):
    m = np.mean(np.abs(w*oracC_oof + (1-w)*mega_oof - y_true))
    if m < best_m:
        best_m, best_w = m, w
print(f"\n2-way blend oracle-C + mega33: w={best_w:.2f} MAE={best_m:.4f}  delta={best_m-mega_mae:.4f}")

# 2-way blend: oracle-C vs FIXED
best_m2, best_w2 = 9999, 0
for w in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs(w*oracC_oof + (1-w)*fixed_oof - y_true))
    if m < best_m2:
        best_m2, best_w2 = m, w
print(f"2-way blend oracle-C + FIXED:  w={best_w2:.2f} MAE={best_m2:.4f}  delta={best_m2-fixed_mae:.4f}")

# Full blend: 6-way optimization
from scipy.optimize import minimize
def total_mae(w):
    w = np.array(w)
    w = np.maximum(0, w) / (np.maximum(0, w).sum() + 1e-9)
    blend = (w[0]*mega_oof + w[1]*rank_oof + w[2]*iter1_oof +
             w[3]*iter2_oof + w[4]*iter3_oof + w[5]*oracC_oof)
    return np.mean(np.abs(blend - y_true))

# Grid: just add oracle-C on top of FIXED (scale remaining proportionally)
print("\nGrid: oracle-C added to FIXED (proportional rescale):", flush=True)
best_m6, best_wC = 9999, 0
for wC in np.arange(0, 0.41, 0.02):
    rem = 1 - wC
    blend = (rem*(fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
                  fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof) +
             wC * oracC_oof)
    m = np.mean(np.abs(blend - y_true))
    if m < best_m6:
        best_m6, best_wC = m, wC
print(f"Best oracle-C added to FIXED: wC={best_wC:.2f} MAE={best_m6:.4f}  delta={best_m6-fixed_mae:.4f}")

# Decision and submission
print(f"\n=== DECISION ===")
best_blend_mae = min(best_m, best_m2, best_m6)
print(f"Best OOF achievable: {best_blend_mae:.4f}")
print(f"FIXED OOF: {fixed_mae:.4f}")
print(f"Delta: {best_blend_mae - fixed_mae:.4f}")

if best_blend_mae < fixed_mae - 0.003:
    print(f"→ Significant OOF improvement (>0.003). Generating submission.", flush=True)

    # Determine which blend to use
    if best_m6 <= min(best_m, best_m2):
        # oracle-C added to FIXED
        rem = 1 - best_wC
        test_blend = (rem*(fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                           fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test) +
                      best_wC * oracC_test)
        label = f"oracle_C_added_wC{best_wC:.2f}"
    elif best_m2 <= best_m:
        # oracle-C + FIXED 2-way
        fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                      fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)
        test_blend = best_w2*oracC_test + (1-best_w2)*fixed_test
        label = f"oracle_C_FIXED2way_w{best_w2:.2f}"
    else:
        # oracle-C + mega33 2-way
        test_blend = best_w*oracC_test + (1-best_w)*mega_test
        label = f"oracle_C_mega33_w{best_w:.2f}"

    test_blend = np.maximum(0, test_blend)

    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values,
                           'avg_delay_minutes_next_30m': test_blend})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_C_{label}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"Saved: {fname}")
    print(f"OOF: {best_blend_mae:.4f}  (FIXED: {fixed_mae:.4f}  delta: {best_blend_mae-fixed_mae:.4f})")
elif best_blend_mae < fixed_mae - 0.001:
    print(f"→ Small improvement (0.001-0.003). May be noise. Check fold stability.", flush=True)
    # Check fold-level variability
    folds_delta = []
    gkf_groups = train_id['layout_id'].values
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    for _, val_idx in gkf.split(np.arange(len(train_id)), groups=gkf_groups):
        fixed_val = fixed_oof[val_idx]
        oracC_val = oracC_oof[val_idx]
        y_val = y_true[val_idx]
        blend_val = best_wC*oracC_val + (1-best_wC)*fixed_val
        delta = np.mean(np.abs(blend_val - y_val)) - np.mean(np.abs(fixed_val - y_val))
        folds_delta.append(delta)
    print(f"Fold deltas: {[f'{x:.4f}' for x in folds_delta]}")
    print(f"Std of fold deltas: {np.std(folds_delta):.4f}")
    if np.mean(folds_delta) < -0.001 and max(folds_delta) < 0.005:
        print("→ Consistent improvement across folds. Generating submission.", flush=True)
        # Generate
        rem = 1 - best_wC
        fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                      fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)
        test_blend = np.maximum(0, best_wC*oracC_test + rem*fixed_test)
        sample_sub = pd.read_csv('sample_submission.csv')
        sub_df = pd.DataFrame({'ID': test_id['ID'].values,
                               'avg_delay_minutes_next_30m': test_blend})
        sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
        fname = f'submission_oracle_C_wC{best_wC:.2f}_OOF{best_blend_mae:.4f}.csv'
        sub_df.to_csv(fname, index=False)
        print(f"Saved: {fname}")
    else:
        print("→ Inconsistent improvement. NOT generating submission.")
else:
    print("→ No meaningful improvement. Keeping FIXED as best submission.", flush=True)

with open('results/oracle_seq/blend_summary_v2.json','w') as f:
    json.dump({
        'fixed_mae': float(fixed_mae),
        'oracle_C_mae': float(oracC_mae),
        'corr_C_mega': float(corr_C_M),
        'corr_C_FIXED': float(corr_C_F),
        'best_2way_w': float(best_w2),
        'best_2way_mae': float(best_m2),
        'best_added_wC': float(best_wC),
        'best_added_mae': float(best_m6),
    }, f, indent=2)
print("\nDone.", flush=True)
