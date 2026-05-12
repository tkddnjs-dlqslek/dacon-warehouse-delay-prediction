"""
Scipy continuous optimization of blend weights.
Also tries adding cascade predictions as additional components.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize, differential_evolution

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

with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)

# All components (row_id order)
components_oof = {
    'mega33':  d['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'iter_r1':  np.load('results/iter_pseudo/round1_oof.npy')[id2],
    'iter_r2':  np.load('results/iter_pseudo/round2_oof.npy')[id2],
    'iter_r3':  np.load('results/iter_pseudo/round3_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
components_test = {
    'mega33':  d['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'iter_r1':  np.load('results/iter_pseudo/round1_test.npy')[te_id2],
    'iter_r2':  np.load('results/iter_pseudo/round2_test.npy')[te_id2],
    'iter_r3':  np.load('results/iter_pseudo/round3_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

# Add cascade specialist as component
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

# Dual gate cascade prediction (refined3 best)
m11 = (clf_oof  > 0.11).astype(float); m11_te = (clf_test > 0.11).astype(float)
m25 = (clf_oof  > 0.25).astype(float); m25_te = (clf_test > 0.25).astype(float)

keys = list(components_oof.keys())
C_oof  = np.array([components_oof[k]  for k in keys]).T   # (n_train, n_comp)
C_test = np.array([components_test[k] for k in keys]).T

print(f"Components: {keys}")
print(f"C_oof shape: {C_oof.shape}")

# Current best weights (fixed component is mega33 × fw_mega33 + rank × fw_rank + ...)
fw_current = np.array([0.7636614598089654*0.64,    # mega33 × 0.64
                        0.1588758398901156*0.64,    # rank_adj × 0.64
                        0.011855567572749024*0.64,  # iter_r1 × 0.64
                        0.034568307*0.64,           # iter_r2 × 0.64
                        0.031038826*0.64,           # iter_r3 × 0.64
                        0.12, 0.16, 0.08])          # oracle xgb, lv2, rem
curr_pred = C_oof @ fw_current
curr_mae = np.mean(np.abs(curr_pred - y_true))
print(f"\nCurrent blend OOF: {curr_mae:.5f}  (weights sum={fw_current.sum():.4f})")

# Verify
fw_fixed = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
               iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = sum(fw_fixed[k]*components_oof[k] for k in fw_fixed)
best_oof = 0.64*fixed_oof + 0.12*components_oof['oracle_xgb'] + 0.16*components_oof['oracle_lv2'] + 0.08*components_oof['oracle_rem']
print(f"Reproduced OOF: {np.mean(np.abs(best_oof-y_true)):.5f}")

def mae(w, C, y):
    pred = np.clip(C @ w, 0, None)
    return np.mean(np.abs(pred - y))

# Constrain weights to sum to 1
constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
bounds = [(0, 1)] * len(keys)

print(f"\n[Search1] L-BFGS-B from current weights", flush=True)
res1 = minimize(mae, fw_current, args=(C_oof, y_true),
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-10})
print(f"  OOF={res1.fun:.5f}  delta={res1.fun-curr_mae:+.5f}")
for k, w in zip(keys, res1.x):
    if w > 0.001: print(f"    {k}: {w:.4f}")

print(f"\n[Search2] Differential evolution (global)", flush=True)
def mae_constrained(w):
    w = np.abs(w); w = w / w.sum()
    return mae(w, C_oof, y_true)
res2 = differential_evolution(mae_constrained, bounds, seed=42,
                               maxiter=300, popsize=15, tol=1e-8,
                               workers=1, polish=True)
w2 = np.abs(res2.x); w2 = w2 / w2.sum()
mae2 = mae(w2, C_oof, y_true)
print(f"  OOF={mae2:.5f}  delta={mae2-curr_mae:+.5f}")
for k, w in zip(keys, w2):
    if w > 0.001: print(f"    {k}: {w:.4f}")

best_w = w2 if mae2 < res1.fun else res1.x
best_mae_opt = min(mae2, res1.fun)
print(f"\nBest scipy OOF: {best_mae_opt:.5f}  delta={best_mae_opt-curr_mae:+.5f}")

# Now apply dual gate cascade on top
best_base_oof  = np.clip(C_oof  @ best_w, 0, None)
best_base_test = np.clip(C_test @ best_w, 0, None)

# Apply cascade gate
casc_oof  = (1-m11*0.03)*best_base_oof  + m11*0.03*lgb_rh_oof
casc_test = (1-m11_te*0.03)*best_base_test + m11_te*0.03*lgb_rh_test
casc_oof  = (1-m25*0.03)*casc_oof  + m25*0.03*lgb_rm_oof
casc_test = (1-m25_te*0.03)*casc_test + m25_te*0.03*lgb_rm_test
casc_mae = np.mean(np.abs(casc_oof - y_true))
print(f"Scipy + cascade gate: {casc_mae:.5f}  delta={casc_mae-curr_mae:+.5f}")

prev_best = 8.37905
final_best_mae = min(best_mae_opt, casc_mae)
if casc_mae < best_mae_opt:
    final_pred_test = casc_test; final_mae = casc_mae
else:
    final_pred_test = best_base_test; final_mae = best_mae_opt

if final_mae < prev_best - 0.00010:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, final_pred_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_scipy_blend_OOF{final_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} ***")
else:
    print(f"\nNo significant improvement over {prev_best:.5f}")
print("Done.")
