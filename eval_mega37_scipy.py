"""
Focused scipy on mega37-centric blend + cascade.
L-BFGS-B on full space found mega37 dominant (0.527) + rank(0.071) + log_v2(0.159) + xgb_combined(0.124) + xgb_v31(0.058) + mega34(0.056).
Refine this with finer DE + cascade gate search.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
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

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)

print("Individual mega MAEs:")
for name, d in [('mega33',d33),('mega34',d34),('mega37',d37)]:
    oof = d['meta_avg_oof'][id2]
    print(f"  {name}: {np.mean(np.abs(oof-y_true)):.5f}")

# Core components
oof_comps = {
    'mega33':  d33['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'mega37':  d37['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb':       np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_log_v2':    np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_rem':   np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':   np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_cb':        np.load('results/oracle_seq/oof_seqC_cb.npy'),
    'oracle_xgb_bestproxy': np.load('results/oracle_seq/oof_seqC_xgb_bestproxy.npy'),
    'oracle_lgb_remaining_v3': np.load('results/oracle_seq/oof_seqC_lgb_remaining_v3.npy'),
}
test_comps = {
    'mega33':  d33['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'mega37':  d37['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb':       np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_log_v2':    np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_rem':   np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':   np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_cb':        np.load('results/oracle_seq/test_C_cb.npy'),
    'oracle_xgb_bestproxy': np.load('results/oracle_seq/test_C_xgb_bestproxy.npy'),
    'oracle_lgb_remaining_v3': np.load('results/oracle_seq/test_C_lgb_remaining_v3.npy'),
}

keys = list(oof_comps.keys())
C_oof  = np.column_stack([oof_comps[k]  for k in keys])
C_test = np.column_stack([test_comps[k] for k in keys])

def mae_fn(w):
    w = np.abs(w); w = w / w.sum()
    return np.mean(np.abs(np.clip(C_oof @ w, 0, None) - y_true))

bounds = [(0, 1)] * len(keys)

# Search 1: L-BFGS-B from mega37-centric init
print(f"\n[Search1] L-BFGS-B from mega37-centric init", flush=True)
w0 = np.zeros(len(keys))
# Approx from full-space L-BFGS-B result
w0[keys.index('mega37')]             = 0.5267
w0[keys.index('mega34')]             = 0.0559
w0[keys.index('rank_adj')]           = 0.0713
w0[keys.index('oracle_log_v2')]      = 0.1591
w0[keys.index('oracle_xgb_v31')]     = 0.0582
w0[keys.index('oracle_xgb_combined')]= 0.1239
w0 = w0 / w0.sum()

res1 = minimize(mae_fn, w0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w1 = np.abs(res1.x); w1 /= w1.sum()
mae1 = np.mean(np.abs(np.clip(C_oof @ w1, 0, None) - y_true))
print(f"  OOF={mae1:.5f}")
for k, w in zip(keys, w1):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 2: L-BFGS-B from different start (equal weights for good components)
print(f"\n[Search2] L-BFGS-B from equal-weight init", flush=True)
w0b = np.ones(len(keys)) / len(keys)
res2 = minimize(mae_fn, w0b, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w2 = np.abs(res2.x); w2 /= w2.sum()
mae2 = np.mean(np.abs(np.clip(C_oof @ w2, 0, None) - y_true))
print(f"  OOF={mae2:.5f}")
for k, w in zip(keys, w2):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 3: Nelder-Mead from best L-BFGS result
best_w_so_far = w1 if mae1 <= mae2 else w2
best_mae_so_far = min(mae1, mae2)
print(f"\n[Search3] Nelder-Mead refinement from best (OOF={best_mae_so_far:.5f})", flush=True)
res3 = minimize(mae_fn, best_w_so_far, method='Nelder-Mead',
                options={'xatol':1e-6, 'fatol':1e-7, 'maxiter':3000, 'adaptive':True})
w3 = np.abs(res3.x); w3 /= w3.sum()
mae3 = np.mean(np.abs(np.clip(C_oof @ w3, 0, None) - y_true))
print(f"  OOF={mae3:.5f}")
for k, w in zip(keys, w3):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 4: DE on reduced component set (top active from L-BFGS-B)
active = [k for k, w in zip(keys, w1) if w > 0.01]
print(f"\n[Search4] DE on active set: {active}", flush=True)
active_idx = [keys.index(k) for k in active]
C_act   = C_oof[:, active_idx]
C_act_t = C_test[:, active_idx]

def mae_act(w):
    w = np.abs(w); w = w / w.sum()
    return np.mean(np.abs(np.clip(C_act @ w, 0, None) - y_true))

res4 = differential_evolution(mae_act, [(0,1)]*len(active), seed=42,
                               maxiter=500, popsize=25, tol=1e-10,
                               workers=1, polish=True)
w4 = np.abs(res4.x); w4 /= w4.sum()
mae4 = np.mean(np.abs(np.clip(C_act @ w4, 0, None) - y_true))
print(f"  OOF={mae4:.5f}")
for k, w in zip(active, w4):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Reconstruct full weight vector for w4
w4_full = np.zeros(len(keys))
for i, k in enumerate(active): w4_full[keys.index(k)] = w4[i]

# Pick overall best base
candidates = [(mae1, w1, C_oof, C_test), (mae2, w2, C_oof, C_test),
              (mae3, w3, C_oof, C_test), (mae4, w4_full, C_oof, C_test)]
candidates.sort(key=lambda x: x[0])
best_base_mae, best_w_base, C_tr_b, C_te_b = candidates[0]
print(f"\n=== Best base: OOF={best_base_mae:.5f} ===")
for k, w in zip(keys, best_w_base):
    if w > 0.005: print(f"  {k}: {w:.4f}")

base_oof_best  = np.clip(C_tr_b @ best_w_base, 0, None)
base_test_best = np.clip(C_te_b @ best_w_base, 0, None)

# Cascade gate search
print(f"\n[Cascade] Fine dual gate search on mega37 base...", flush=True)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best = 8.37851
best_mae = best_base_mae
best_oof_pred = base_oof_best; best_test_pred = base_test_best
best_cfg = 'base_only'

for p1 in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]:
    m1    = (clf_oof  > p1).astype(float)
    m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.010, 0.080, 0.005):
        b1    = (1-m1*w1)*base_oof_best  + m1*w1*rh_oof
        b1_te = (1-m1_te*w1)*base_test_best + m1_te*w1*rh_test
        for p2 in [0.18, 0.20, 0.22, 0.25, 0.28, 0.30]:
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float)
            m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.010, 0.080, 0.005):
                blend = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm; best_cfg = f'p{p1}_w{w1:.3f}+p{p2}_w{w2:.3f}'
                    best_oof_pred = blend
                    best_test_pred = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

print(f"\n=== FINAL: base={best_base_mae:.5f}  +gate={best_mae:.5f} ===")
print(f"  vs prev_best={prev_best:.5f}: delta={best_mae-prev_best:+.5f}")

if best_mae < prev_best - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mega37_scipy_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"No improvement (best_mae={best_mae:.5f})")
print("Done.")
