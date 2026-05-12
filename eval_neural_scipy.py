"""
Add neural models (mlp_deep family) to mega37 scipy blend.
MLP models have OOF MAE 8.60-8.64 — much better than oracle variants individually.
Neural models are in ls-sorted order → need id2/te_id2 indexing.
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

# Load mega pkl files
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)
with open('results/mega33_v31_final.pkl','rb') as f: d33v31 = pickle.load(f)

# Load neural models (ls-sorted → need id2/te_id2)
def load_neural(fp, oof_key, test_key):
    with open(fp,'rb') as f: d = pickle.load(f)
    return d[oof_key][id2], d[test_key][te_id2]

nn = {}
nn['mlp_deep'],    nn['mlp_deep_t']    = load_neural('results/mlp_deep_final.pkl', 'oof', 'test')
nn['mlp_s3'],      nn['mlp_s3_t']      = load_neural('results/mlp_deep_s3_final.pkl', 'oof', 'test')
nn['mlp_s2'],      nn['mlp_s2_t']      = load_neural('results/mlp_deep_s2_final.pkl', 'oof', 'test')
nn['mlp_gelu'],    nn['mlp_gelu_t']    = load_neural('results/mlp_deep_gelu_final.pkl', 'oof', 'test')
nn['mlp'],         nn['mlp_t']         = load_neural('results/mlp_final.pkl', 'mlp_oof', 'mlp_test')
nn['mlp_aug'],     nn['mlp_aug_t']     = load_neural('results/mlp_aug_final.pkl', 'mlp_aug_oof', 'mlp_aug_test')

print("Neural OOF MAEs:")
for k in ['mlp_deep','mlp_s3','mlp_s2','mlp_gelu','mlp','mlp_aug']:
    print(f"  {k}: {np.mean(np.abs(nn[k]-y_true)):.5f}")

# Current best mega37 base
mega37_oof  = d37['meta_avg_oof'][id2]
mega37_test = d37['meta_avg_test'][te_id2]
print(f"\nMega37 base: {np.mean(np.abs(mega37_oof-y_true)):.5f}")

# Build component matrix — include top components from proven base + neural
keys_full = [
    # Proven mega37 base components
    'mega37', 'mega34', 'mega33', 'mega33_v31', 'rank_adj',
    # Best oracle
    'oracle_log_v2', 'oracle_xgb_combined', 'oracle_xgb_v31', 'oracle_xgb',
    'oracle_xgb_rem', 'oracle_cb',
    # Neural models
    'mlp_deep', 'mlp_s3', 'mlp_s2', 'mlp_gelu', 'mlp', 'mlp_aug',
]

oof_vals = {
    'mega37':    d37['meta_avg_oof'][id2],
    'mega34':    d34['meta_avg_oof'][id2],
    'mega33':    d33['meta_avg_oof'][id2],
    'mega33_v31': d33v31['meta_avg_oof'][id2],
    'rank_adj':  np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':      np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':     np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_xgb':         np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_xgb_rem':     np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':          np.load('results/oracle_seq/oof_seqC_cb.npy'),
    'mlp_deep': nn['mlp_deep'], 'mlp_s3': nn['mlp_s3'], 'mlp_s2': nn['mlp_s2'],
    'mlp_gelu': nn['mlp_gelu'], 'mlp': nn['mlp'], 'mlp_aug': nn['mlp_aug'],
}
test_vals = {
    'mega37':    d37['meta_avg_test'][te_id2],
    'mega34':    d34['meta_avg_test'][te_id2],
    'mega33':    d33['meta_avg_test'][te_id2],
    'mega33_v31': d33v31['meta_avg_test'][te_id2],
    'rank_adj':  np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':      np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':     np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_xgb':         np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_xgb_rem':     np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_cb':          np.load('results/oracle_seq/test_C_cb.npy'),
    'mlp_deep': nn['mlp_deep_t'], 'mlp_s3': nn['mlp_s3_t'], 'mlp_s2': nn['mlp_s2_t'],
    'mlp_gelu': nn['mlp_gelu_t'], 'mlp': nn['mlp_t'], 'mlp_aug': nn['mlp_aug_t'],
}

C_oof  = np.column_stack([oof_vals[k]  for k in keys_full])
C_test = np.column_stack([test_vals[k] for k in keys_full])
print(f"\nTotal components: {len(keys_full)} {keys_full}")

# Correlation with mega37
print("\nCorrelations with mega37:")
for k in keys_full:
    if k == 'mega37': continue
    corr = np.corrcoef(oof_vals[k], mega37_oof)[0,1]
    mae  = np.mean(np.abs(oof_vals[k] - y_true))
    print(f"  {k:20s}: corr={corr:.4f}  MAE={mae:.5f}")

def mae_fn(w):
    w = np.abs(w); w = w / w.sum()
    return np.mean(np.abs(np.clip(C_oof @ w, 0, None) - y_true))

bounds = [(0, 1)] * len(keys_full)

# Search 1: L-BFGS-B from mega37-proven init
print(f"\n[Search1] L-BFGS-B from mega37-proven init", flush=True)
w0 = np.zeros(len(keys_full))
w0[keys_full.index('mega37')]             = 0.5269
w0[keys_full.index('mega34')]             = 0.0558
w0[keys_full.index('rank_adj')]           = 0.0712
w0[keys_full.index('oracle_log_v2')]      = 0.1597
w0[keys_full.index('oracle_xgb_combined')]= 0.1237
w0[keys_full.index('oracle_xgb_v31')]     = 0.0583
w0 = w0 / w0.sum()

res1 = minimize(mae_fn, w0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w1 = np.abs(res1.x); w1 /= w1.sum()
mae1 = np.mean(np.abs(np.clip(C_oof @ w1, 0, None) - y_true))
print(f"  OOF={mae1:.5f}  delta_vs_mega37base={mae1-np.mean(np.abs(mega37_oof-y_true)):+.5f}")
for k, w in zip(keys_full, w1):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 2: L-BFGS-B with neural-heavy init
print(f"\n[Search2] L-BFGS-B with neural-heavy init", flush=True)
w0b = np.zeros(len(keys_full))
w0b[keys_full.index('mega37')]     = 0.35
w0b[keys_full.index('mlp_deep')]   = 0.15
w0b[keys_full.index('mlp_s3')]     = 0.10
w0b[keys_full.index('oracle_log_v2')]      = 0.15
w0b[keys_full.index('oracle_xgb_combined')]= 0.10
w0b[keys_full.index('rank_adj')]   = 0.07
w0b[keys_full.index('mega34')]     = 0.04
w0b[keys_full.index('oracle_xgb_v31')]= 0.04
w0b = w0b / w0b.sum()

res2 = minimize(mae_fn, w0b, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w2 = np.abs(res2.x); w2 /= w2.sum()
mae2 = np.mean(np.abs(np.clip(C_oof @ w2, 0, None) - y_true))
print(f"  OOF={mae2:.5f}  delta_vs_mega37base={mae2-np.mean(np.abs(mega37_oof-y_true)):+.5f}")
for k, w in zip(keys_full, w2):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 3: DE on best active set
best_w_search = w1 if mae1 <= mae2 else w2
active_keys = [k for k, w in zip(keys_full, best_w_search) if w > 0.01]
print(f"\n[Search3] DE on active: {active_keys}", flush=True)
act_idx = [keys_full.index(k) for k in active_keys]
C_act = C_oof[:, act_idx]; C_act_te = C_test[:, act_idx]

def mae_act(w):
    w = np.abs(w); w = w/w.sum()
    return np.mean(np.abs(np.clip(C_act @ w, 0, None) - y_true))

res3 = differential_evolution(mae_act, [(0,1)]*len(active_keys), seed=42,
                               maxiter=600, popsize=25, tol=1e-10,
                               workers=1, polish=True)
w3 = np.abs(res3.x); w3 /= w3.sum()
mae3 = np.mean(np.abs(np.clip(C_act @ w3, 0, None) - y_true))
print(f"  OOF={mae3:.5f}")
for k, w in zip(active_keys, w3):
    if w > 0.005: print(f"    {k}: {w:.4f}")
w3_full = np.zeros(len(keys_full))
for i, k in enumerate(active_keys): w3_full[keys_full.index(k)] = w3[i]

# Pick best base
candidates = [(mae1, w1, C_oof, C_test, 'lbfgsb_proven'),
              (mae2, w2, C_oof, C_test, 'lbfgsb_neural'),
              (mae3, w3_full, C_oof, C_test, 'de_active')]
candidates.sort(key=lambda x: x[0])
best_base_mae, best_w, Ctr, Cte, best_name = candidates[0]
print(f"\n=== Best base: {best_name}  OOF={best_base_mae:.5f} ===")
base_oof_b  = np.clip(Ctr @ best_w, 0, None)
base_test_b = np.clip(Cte @ best_w, 0, None)

# Cascade gate search
print(f"\n[Cascade] Dual gate search...", flush=True)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best = 8.37165
best_final_mae = best_base_mae
best_final_oof  = base_oof_b; best_final_test = base_test_b
best_final_cfg  = best_name + '_base_only'

for p1 in np.arange(0.08, 0.16, 0.01):
    m1    = (clf_oof  > p1).astype(float)
    m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.010, 0.060, 0.005):
        b1    = (1-m1*w1)*base_oof_b  + m1*w1*rh_oof
        b1_te = (1-m1_te*w1)*base_test_b + m1_te*w1*rh_test
        for p2 in np.arange(0.17, 0.36, 0.01):
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float)
            m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.010, 0.060, 0.005):
                blend = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_final_mae:
                    best_final_mae = mm
                    best_final_cfg = f'{best_name}+p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    best_final_oof  = blend
                    best_final_test = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                    print(f"★ {best_final_cfg}  OOF={mm:.5f}  delta={mm-prev_best:+.5f}", flush=True)

print(f"\n=== FINAL ===")
print(f"  base={best_base_mae:.5f}  +gate={best_final_mae:.5f}")
print(f"  vs prev_best={prev_best:.5f}: delta={best_final_mae-prev_best:+.5f}")

ref_prev = 8.37851
if best_final_mae < ref_prev - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_final_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_neural_scipy_OOF{best_final_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print("No improvement over ref.")
print("Done.")
