"""
Refine the neural+mega37 blend (best base=8.37119, with gate=8.36787).
Try: more neural models, finer gate, scipy continuous gate.
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

def load_neural(fp, ok, tk):
    with open(fp,'rb') as f: d = pickle.load(f)
    return d[ok][id2], d[tk][te_id2]

# Proven best weights from neural_scipy: mega37×0.4424, rank×0.0747, log_v2×0.1540,
# xgb_combined×0.1406, xgb_v31×0.0589, mlp_deep×0.0506, mlp_s2×0.0539, mlp_gelu×0.0198

# Base with proven weights
base_w_proven = {
    'mega37': 0.4424, 'rank_adj': 0.0747, 'oracle_log_v2': 0.1540,
    'oracle_xgb_combined': 0.1406, 'oracle_xgb_v31': 0.0589,
    'mlp_deep': 0.0506, 'mlp_s2': 0.0539, 'mlp_gelu': 0.0198,
}
total = sum(base_w_proven.values()); base_w_proven = {k:v/total for k,v in base_w_proven.items()}

oof_d = {
    'mega37':  d37['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'mega33':  d33['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_xgb':          np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_cb':           np.load('results/oracle_seq/oof_seqC_cb.npy'),
}
test_d = {
    'mega37':  d37['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'mega33':  d33['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_xgb':          np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_cb':           np.load('results/oracle_seq/test_C_cb.npy'),
}

nn_models = [
    ('mlp_deep', 'results/mlp_deep_final.pkl', 'oof', 'test'),
    ('mlp_s3',   'results/mlp_deep_s3_final.pkl', 'oof', 'test'),
    ('mlp_s2',   'results/mlp_deep_s2_final.pkl', 'oof', 'test'),
    ('mlp_gelu', 'results/mlp_deep_gelu_final.pkl', 'oof', 'test'),
    ('mlp',      'results/mlp_final.pkl', 'mlp_oof', 'mlp_test'),
    ('mlp_aug',  'results/mlp_aug_final.pkl', 'mlp_aug_oof', 'mlp_aug_test'),
    ('mlp2',     'results/mlp2_final.pkl', 'mlp2_oof', 'mlp2_test'),
    ('mlp_wide', 'results/mlp_wide_final.pkl', 'oof', 'test'),
]
for nm, fp, ok, tk in nn_models:
    o, t = load_neural(fp, ok, tk)
    oof_d[nm] = o; test_d[nm] = t

# Reproduce proven base
base_oof  = sum(base_w_proven.get(k,0)*oof_d[k]  for k in oof_d if k in base_w_proven)
base_test = sum(base_w_proven.get(k,0)*test_d[k] for k in test_d if k in base_w_proven)
base_oof  = np.clip(base_oof, 0, None)
base_test = np.clip(base_test, 0, None)
base_mae  = np.mean(np.abs(base_oof - y_true))
print(f"Proven base OOF: {base_mae:.5f}")

# Full extended component matrix
keys_ext = list(oof_d.keys())
C_oof  = np.column_stack([oof_d[k]  for k in keys_ext])
C_test = np.column_stack([test_d[k] for k in keys_ext])

def mae_fn(w):
    w = np.abs(w); w = w / w.sum()
    return np.mean(np.abs(np.clip(C_oof @ w, 0, None) - y_true))

bounds = [(0, 1)] * len(keys_ext)

# Search 1: L-BFGS-B from proven weights extended with zeros
print(f"\n[Search1] L-BFGS-B from proven+extended init", flush=True)
w0 = np.array([base_w_proven.get(k, 0.0) for k in keys_ext])
w0 = w0 / w0.sum()
res1 = minimize(mae_fn, w0, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w1 = np.abs(res1.x); w1 /= w1.sum()
mae1 = np.mean(np.abs(np.clip(C_oof @ w1, 0, None) - y_true))
print(f"  OOF={mae1:.5f}  delta={mae1-base_mae:+.5f}")
for k, w in zip(keys_ext, w1):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 2: Try mega33/mega34 blended with mega37
print(f"\n[Search2] Add mega33/34 to mix", flush=True)
w0b = w1.copy()
w0b[keys_ext.index('mega34')] += 0.05
w0b[keys_ext.index('mega33')] += 0.03
w0b = w0b / w0b.sum()
res2 = minimize(mae_fn, w0b, method='L-BFGS-B', bounds=bounds,
                options={'maxiter':5000, 'ftol':1e-12, 'gtol':1e-9})
w2 = np.abs(res2.x); w2 /= w2.sum()
mae2 = np.mean(np.abs(np.clip(C_oof @ w2, 0, None) - y_true))
print(f"  OOF={mae2:.5f}  delta={mae2-base_mae:+.5f}")
for k, w in zip(keys_ext, w2):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# Search 3: DE on extended set
active = [k for k,w in zip(keys_ext, w1) if w > 0.008]
# Always include mlp_s3 (might help)
if 'mlp_s3' not in active: active.append('mlp_s3')
print(f"\n[Search3] DE on: {active}", flush=True)
act_idx = [keys_ext.index(k) for k in active]
C_act = C_oof[:, act_idx]; C_act_te = C_test[:, act_idx]
def mae_act(w):
    w=np.abs(w); w=w/w.sum()
    return np.mean(np.abs(np.clip(C_act@w,0,None)-y_true))
res3 = differential_evolution(mae_act, [(0,1)]*len(active), seed=42,
                               maxiter=600, popsize=25, tol=1e-10, workers=1, polish=True)
w3 = np.abs(res3.x); w3 /= w3.sum()
mae3 = np.mean(np.abs(np.clip(C_act@w3,0,None)-y_true))
print(f"  OOF={mae3:.5f}")
for k,w in zip(active,w3):
    if w>0.005: print(f"    {k}: {w:.4f}")
w3_full = np.zeros(len(keys_ext))
for i,k in enumerate(active): w3_full[keys_ext.index(k)] = w3[i]

# Best base
candidates = [(mae1,w1,C_oof,C_test,'lbfgsb1'),
              (mae2,w2,C_oof,C_test,'lbfgsb2'),
              (mae3,w3_full,C_oof,C_test,'de_active')]
candidates.sort(key=lambda x:x[0])
best_base_mae, best_w, Ctr, Cte, bname = candidates[0]
print(f"\n=== Best base: {bname}  OOF={best_base_mae:.5f} ===")
base_oof_b  = np.clip(Ctr@best_w, 0, None)
base_test_b = np.clip(Cte@best_w, 0, None)

# Cascade gate — finer search around p0.11/p0.27
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

print(f"\n[Cascade] Finer dual gate (step=0.002)", flush=True)
prev_best = 8.36787
best_mae = best_base_mae
best_oof_p = base_oof_b; best_test_p = base_test_b; best_cfg = bname+'_base'

for p1 in np.arange(0.08, 0.15, 0.01):
    m1    = (clf_oof  > p1).astype(float); m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.005, 0.045, 0.002):
        b1    = (1-m1*w1)*base_oof_b  + m1*w1*rh_oof
        b1_te = (1-m1_te*w1)*base_test_b + m1_te*w1*rh_test
        for p2 in np.arange(0.16, 0.35, 0.01):
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float); m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.005, 0.060, 0.002):
                blend = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'{bname}+p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    best_oof_p = blend
                    best_test_p = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

# Also scipy continuous gate
print(f"\n[Cascade2] Scipy continuous gate", flush=True)
def gate_mae(params):
    p1, w1, p2, w2 = params
    p1=np.clip(p1,0.05,0.20); w1=np.clip(w1,0,0.10)
    p2=np.clip(p2,0.15,0.45); w2=np.clip(w2,0,0.10)
    if p2<=p1: return 9999
    m1=(clf_oof>p1).astype(float); m2=(clf_oof>p2).astype(float)
    b1=(1-m1*w1)*base_oof_b+m1*w1*rh_oof
    bl=(1-m2*w2)*b1+m2*w2*rm_oof
    return np.mean(np.abs(bl-y_true))
for init in [[0.11,0.025,0.27,0.040],[0.11,0.025,0.22,0.035],
             [0.09,0.015,0.24,0.040],[0.12,0.020,0.26,0.035]]:
    res=minimize(gate_mae,init,method='Nelder-Mead',
                 options={'xatol':1e-6,'fatol':1e-7,'maxiter':2000})
    p1c,w1c,p2c,w2c=res.x
    p1c=np.clip(p1c,0.05,0.20); w1c=np.clip(w1c,0,0.10)
    p2c=np.clip(p2c,0.15,0.45); w2c=np.clip(w2c,0,0.10)
    m1=(clf_oof>p1c).astype(float); m2=(clf_oof>p2c).astype(float)
    m1_te=(clf_test>p1c).astype(float); m2_te=(clf_test>p2c).astype(float)
    b1=(1-m1*w1c)*base_oof_b+m1*w1c*rh_oof
    bl=(1-m2*w2c)*b1+m2*w2c*rm_oof
    mm=np.mean(np.abs(bl-y_true))
    print(f"  p1={p1c:.3f} w1={w1c:.4f} p2={p2c:.3f} w2={w2c:.4f}  OOF={mm:.5f}", flush=True)
    if mm<best_mae:
        best_mae=mm; best_cfg=f'scipy_gate_p{p1c:.3f}w{w1c:.4f}+p{p2c:.3f}w{w2c:.4f}'
        b1_te=(1-m1_te*w1c)*base_test_b+m1_te*w1c*rh_test
        best_test_p=(1-m2_te*w2c)*b1_te+m2_te*w2c*rm_test
        print(f"★ NEW BEST: {best_mae:.5f}")

print(f"\n=== FINAL ===")
print(f"  base={best_base_mae:.5f}  +gate={best_mae:.5f}  prev_best={prev_best:.5f}")
print(f"  delta_vs_prev={best_mae-prev_best:+.5f}")

ref_prev = 8.37851
if best_mae < ref_prev - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_p)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_neural_refined_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
print("Done.")
