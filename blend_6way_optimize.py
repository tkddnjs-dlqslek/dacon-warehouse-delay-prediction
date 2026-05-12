"""
Full 6-way unconstrained weight optimization: mega33, rank_adj, iter1/2/3, oracle-C_v2.
Also try 7-way if oracle-C_v1 adds value.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json
from scipy.optimize import minimize

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

# All OOFs in ID order
mega_oof  = d['meta_avg_oof'][id_to_lspos]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos]
oracC2_oof = np.load('results/oracle_seq/oof_seqC_v2.npy')   # already ID order
oracC1_oof = np.load('results/oracle_seq/oof_seqC.npy')      # already ID order

# All tests in ID order
mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
oracC2_test = np.load('results/oracle_seq/test_C_v2.npy')    # ID order
oracC1_test = np.load('results/oracle_seq/test_C.npy')       # ID order

fw_fixed = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
                iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
                iter_r3=0.031038826035934514)
fixed_oof = (fw_fixed['mega33']*mega_oof + fw_fixed['rank_adj']*rank_oof +
             fw_fixed['iter_r1']*iter1_oof + fw_fixed['iter_r2']*iter2_oof + fw_fixed['iter_r3']*iter3_oof)
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

orafC2_mae = np.mean(np.abs(oracC2_oof - y_true))
print(f"oracle-C_v2 OOF MAE: {orafC2_mae:.4f}", flush=True)

# === 6-way: mega33, rank, iter1, iter2, iter3, oracle-C_v2 ===
oofs_6 = [mega_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof, oracC2_oof]
labels_6 = ['mega33', 'rank_adj', 'iter_r1', 'iter_r2', 'iter_r3', 'oracle-C_v2']

def mae_6(raw_w):
    w = np.maximum(raw_w, 0)
    w = w / (w.sum() + 1e-9)
    blend = sum(wi * oo for wi, oo in zip(w, oofs_6))
    return np.mean(np.abs(blend - y_true))

# Grid: oracle-C_v2 weight from 0 to 0.5, scale remaining proportionally
print("\nGrid: oracle-C_v2 added to FIXED proportionally", flush=True)
best_grid_mae, best_grid_wC2 = 9999, 0
for wC2 in np.arange(0, 0.51, 0.02):
    rem = 1 - wC2
    blend = (rem*(fw_fixed['mega33']*mega_oof + fw_fixed['rank_adj']*rank_oof +
                  fw_fixed['iter_r1']*iter1_oof + fw_fixed['iter_r2']*iter2_oof + fw_fixed['iter_r3']*iter3_oof) +
             wC2*oracC2_oof)
    m = np.mean(np.abs(blend - y_true))
    if m < best_grid_mae:
        best_grid_mae, best_grid_wC2 = m, wC2
print(f"Best grid: wC2={best_grid_wC2:.2f} MAE={best_grid_mae:.4f}  delta={best_grid_mae-fixed_mae:.4f}", flush=True)

# Full unconstrained 6-way optimization
print("\nFull unconstrained 6-way optimization...", flush=True)
w0 = np.array([fw_fixed['mega33'], fw_fixed['rank_adj'], fw_fixed['iter_r1'],
               fw_fixed['iter_r2'], fw_fixed['iter_r3'], best_grid_wC2])
w0 = w0 / w0.sum()

results_6 = minimize(mae_6, w0, method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-6, 'fatol': 1e-6})
w_opt6 = np.maximum(results_6.x, 0)
w_opt6 = w_opt6 / w_opt6.sum()
mae_opt6 = mae_6(w_opt6)
print(f"Unconstrained 6-way MAE: {mae_opt6:.4f}  delta={mae_opt6-fixed_mae:.4f}", flush=True)
for i, (lbl, ww) in enumerate(zip(labels_6, w_opt6)):
    print(f"  {lbl}: {ww:.4f}", flush=True)

# === 7-way: add oracle-C_v1 as well ===
print("\n=== 7-way: mega33, rank, iter1/2/3, C_v1, C_v2 ===", flush=True)
oofs_7 = oofs_6 + [oracC1_oof]
labels_7 = labels_6 + ['oracle-C_v1']

def mae_7(raw_w):
    w = np.maximum(raw_w, 0)
    w = w / (w.sum() + 1e-9)
    blend = sum(wi * oo for wi, oo in zip(w, oofs_7))
    return np.mean(np.abs(blend - y_true))

w0_7 = np.append(w_opt6, 0.0)
w0_7 = w0_7 / w0_7.sum()
results_7 = minimize(mae_7, w0_7, method='Nelder-Mead',
                     options={'maxiter': 50000, 'xatol': 1e-6, 'fatol': 1e-6})
w_opt7 = np.maximum(results_7.x, 0)
w_opt7 = w_opt7 / w_opt7.sum()
mae_opt7 = mae_7(w_opt7)
print(f"7-way MAE: {mae_opt7:.4f}  delta={mae_opt7-fixed_mae:.4f}", flush=True)
for lbl, ww in zip(labels_7, w_opt7):
    print(f"  {lbl}: {ww:.4f}", flush=True)

# Fold check for best blend
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train_id['layout_id'].values

if mae_opt7 < mae_opt6:
    best_overall_mae = mae_opt7
    w_best = w_opt7
    oofs_best = oofs_7
    tests_best = [mega_test, rank_test, iter1_test, iter2_test, iter3_test, oracC2_test, oracC1_test]
    label = '7way'
else:
    best_overall_mae = mae_opt6
    w_best = w_opt6
    oofs_best = oofs_6
    tests_best = [mega_test, rank_test, iter1_test, iter2_test, iter3_test, oracC2_test]
    label = '6way'

print(f"\nBest: {label}  MAE={best_overall_mae:.4f}  delta={best_overall_mae-fixed_mae:.4f}", flush=True)

folds_delta = []
for _, val_idx in gkf.split(np.arange(len(train_id)), groups=groups):
    blend_val = sum(wi * oo[val_idx] for wi, oo in zip(w_best, oofs_best))
    fixed_val = fixed_oof[val_idx]
    delta = np.mean(np.abs(blend_val - y_true[val_idx])) - np.mean(np.abs(fixed_val - y_true[val_idx]))
    folds_delta.append(delta)
print(f"Fold deltas ({label}): {[f'{x:.4f}' for x in folds_delta]}", flush=True)
neg_folds = sum(1 for x in folds_delta if x < 0)

# Compare with grid result
print(f"\nSummary:", flush=True)
print(f"  Grid (proportional): wC2={best_grid_wC2:.2f} MAE={best_grid_mae:.4f} delta={best_grid_mae-fixed_mae:.4f}", flush=True)
print(f"  Unconstrained 6way: MAE={mae_opt6:.4f} delta={mae_opt6-fixed_mae:.4f}", flush=True)
print(f"  Unconstrained 7way: MAE={mae_opt7:.4f} delta={mae_opt7-fixed_mae:.4f}", flush=True)
print(f"  Best overall: {label} neg_folds={neg_folds}/5", flush=True)

# Generate submission if improvement
if best_overall_mae < fixed_mae - 0.003 and neg_folds >= 4 and max(folds_delta) < 0.005:
    test_blend = np.maximum(0, sum(wi * t for wi, t in zip(w_best, tests_best)))
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': test_blend})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_{label}_OOF{best_overall_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"Saved: {fname}", flush=True)
elif best_overall_mae < fixed_mae - 0.001 and neg_folds >= 4:
    print(f"→ Improvement but moderate. Saving anyway.", flush=True)
    test_blend = np.maximum(0, sum(wi * t for wi, t in zip(w_best, tests_best)))
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': test_blend})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_{label}_OOF{best_overall_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"Saved: {fname}", flush=True)
else:
    print(f"→ No meaningful improvement beyond oracle-C_v2 grid. Keeping grid result.", flush=True)

with open('results/oracle_seq/blend_6way_summary.json', 'w') as f:
    json.dump({
        'fixed_mae': float(fixed_mae),
        'grid_wC2': float(best_grid_wC2),
        'grid_mae': float(best_grid_mae),
        'unconstrained_6way_mae': float(mae_opt6),
        'unconstrained_7way_mae': float(mae_opt7),
        'best_label': label,
        'best_mae': float(best_overall_mae),
        'fold_deltas': [float(x) for x in folds_delta],
        'weights_best': {lab: float(ww) for lab, ww in zip(labels_6 if label=='6way' else labels_7, w_best)},
    }, f, indent=2)
print("Done.", flush=True)
