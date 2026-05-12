"""
4-way blend analysis: FIXED + oracle-C2 + oracle-XGB + oracle-log
Try to push beyond -0.0100 OOF.
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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_pos[rid] for rid in train_id['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[rid] for rid in test_id['ID'].values]

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

mega_oof  = d['meta_avg_oof'][id_to_lspos]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos]
oracC2_oof  = np.load('results/oracle_seq/oof_seqC_v2.npy')
oracXGB_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
oracLog_oof = np.load('results/oracle_seq/oof_seqC_log.npy')

mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
oracC2_test  = np.load('results/oracle_seq/test_C_v2.npy')
oracXGB_test = np.load('results/oracle_seq/test_C_xgb.npy')
oracLog_test = np.load('results/oracle_seq/test_C_log.npy')

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
              fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

# Correlations
for name, oof in [('C2', oracC2_oof), ('XGB', oracXGB_oof), ('Log', oracLog_oof)]:
    mae = np.mean(np.abs(oof - y_true))
    corr_F = np.corrcoef(oof-y_true, fixed_oof-y_true)[0,1]
    print(f"  oracle-{name}: MAE={mae:.4f}  corr_FIXED={corr_F:.4f}", flush=True)

corr_C2_XGB = np.corrcoef(oracC2_oof-y_true, oracXGB_oof-y_true)[0,1]
corr_C2_Log = np.corrcoef(oracC2_oof-y_true, oracLog_oof-y_true)[0,1]
corr_XGB_Log = np.corrcoef(oracXGB_oof-y_true, oracLog_oof-y_true)[0,1]
print(f"  corr(C2,XGB)={corr_C2_XGB:.4f}  corr(C2,Log)={corr_C2_Log:.4f}  corr(XGB,Log)={corr_XGB_Log:.4f}", flush=True)

# Grid: 4-way FIXED + C2 + XGB + Log
print("\nGrid: FIXED + C2 + XGB + Log (4-way)...", flush=True)
best_m4, best_wC2, best_wXG, best_wLog = 9999, 0, 0, 0
for wC2 in np.arange(0, 0.41, 0.04):
    for wXG in np.arange(0, 0.41, 0.04):
        for wLog in np.arange(0, 0.41, 0.04):
            if wC2 + wXG + wLog > 0.65:
                continue
            blend = (1-wC2-wXG-wLog)*fixed_oof + wC2*oracC2_oof + wXG*oracXGB_oof + wLog*oracLog_oof
            m = np.mean(np.abs(blend - y_true))
            if m < best_m4:
                best_m4, best_wC2, best_wXG, best_wLog = m, wC2, wXG, wLog
print(f"Best 4-way: wC2={best_wC2:.2f} wXG={best_wXG:.2f} wLog={best_wLog:.2f} MAE={best_m4:.4f}  delta={best_m4-fixed_mae:.4f}", flush=True)

# Unconstrained scipy optimizer for 4-way (8 total components)
all_oofs_4 = [mega_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof, oracC2_oof, oracXGB_oof, oracLog_oof]
all_tests_4 = [mega_test, rank_test, iter1_test, iter2_test, iter3_test, oracC2_test, oracXGB_test, oracLog_test]
labels_4 = ['mega33','rank_adj','iter_r1','iter_r2','iter_r3','C2','XGB','Log']

def mae_unc(raw_w):
    w = np.maximum(raw_w, 0)
    w = w / (w.sum() + 1e-9)
    return np.mean(np.abs(sum(wi*oo for wi,oo in zip(w, all_oofs_4)) - y_true))

w0_unc = np.array([fw['mega33'], fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], fw['iter_r3'],
                   best_wC2, best_wXG, best_wLog])
w0_unc /= w0_unc.sum()
result_unc = minimize(mae_unc, w0_unc, method='Nelder-Mead', options={'maxiter':50000,'xatol':1e-6,'fatol':1e-6})
w_unc = np.maximum(result_unc.x, 0)
w_unc /= w_unc.sum()
mae_unc_val = mae_unc(w_unc)
print(f"\nUnconstrained 8-component: MAE={mae_unc_val:.4f}  delta={mae_unc_val-fixed_mae:.4f}", flush=True)
for lbl, ww in zip(labels_4, w_unc):
    print(f"  {lbl}: {ww:.4f}", flush=True)

# Fold consistency for best
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train_id['layout_id'].values

print(f"\nFold analysis for 4-way grid...", flush=True)
folds_4way = []
for _, val_idx in gkf.split(np.arange(len(train_id)), groups=groups):
    blend_val = ((1-best_wC2-best_wXG-best_wLog)*fixed_oof[val_idx] +
                 best_wC2*oracC2_oof[val_idx] + best_wXG*oracXGB_oof[val_idx] + best_wLog*oracLog_oof[val_idx])
    fixed_val = fixed_oof[val_idx]
    delta = np.mean(np.abs(blend_val - y_true[val_idx])) - np.mean(np.abs(fixed_val - y_true[val_idx]))
    folds_4way.append(delta)
print(f"Fold deltas (4way): {[f'{x:.4f}' for x in folds_4way]}", flush=True)
neg_folds = sum(1 for x in folds_4way if x < 0)

folds_unc = []
for _, val_idx in gkf.split(np.arange(len(train_id)), groups=groups):
    blend_val = sum(wi*oo[val_idx] for wi,oo in zip(w_unc, all_oofs_4))
    fixed_val = fixed_oof[val_idx]
    delta = np.mean(np.abs(blend_val - y_true[val_idx])) - np.mean(np.abs(fixed_val - y_true[val_idx]))
    folds_unc.append(delta)
print(f"Fold deltas (unc): {[f'{x:.4f}' for x in folds_unc]}", flush=True)

# Decision
best_candidate = min(best_m4, mae_unc_val)
use_unc = mae_unc_val < best_m4

if best_candidate < fixed_mae - 0.001:
    if use_unc:
        folds = folds_unc; w_use = w_unc; oofs_use = all_oofs_4; tests_use = all_tests_4; label_use = '8comp_unc'
        if sum(x < 0 for x in folds) < 4: use_unc = False
    if not use_unc:
        folds = folds_4way; label_use = '4way_grid'
        if sum(x < 0 for x in folds) >= 4:
            test_blend = np.maximum(0, ((1-best_wC2-best_wXG-best_wLog)*fixed_test +
                                        best_wC2*oracC2_test + best_wXG*oracXGB_test + best_wLog*oracLog_test))
        else:
            print("Not enough fold consistency. No submission.", flush=True)
            test_blend = None
    else:
        test_blend = np.maximum(0, sum(wi*t for wi,t in zip(w_unc, all_tests_4)))

    if test_blend is not None and sum(x < 0 for x in folds) >= 4:
        fname = f'submission_oracle_{label_use}_OOF{best_candidate:.4f}.csv'
        sample_sub = pd.read_csv('sample_submission.csv')
        sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': test_blend})
        sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
        sub_df.to_csv(fname, index=False)
        print(f"Saved: {fname}", flush=True)
else:
    print("No improvement beyond best triple (FIXED+C2+XGB). No new submission.", flush=True)

with open('results/oracle_seq/blend_4way_summary.json', 'w') as f:
    json.dump({
        'fixed_mae': float(fixed_mae),
        'best_4way_grid': float(best_m4),
        'best_4way_grid_wC2': float(best_wC2), 'best_4way_grid_wXG': float(best_wXG), 'best_4way_grid_wLog': float(best_wLog),
        'unconstrained_mae': float(mae_unc_val),
        'unconstrained_weights': {l: float(w) for l, w in zip(labels_4, w_unc)},
        'fold_deltas_4way': [float(x) for x in folds_4way],
        'fold_deltas_unc': [float(x) for x in folds_unc],
    }, f, indent=2)
print("Done.", flush=True)
