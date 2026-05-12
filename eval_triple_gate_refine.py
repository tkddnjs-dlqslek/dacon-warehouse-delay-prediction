"""
Refine the triple gate on neural base (best=8.36746).
Try: scipy continuous 3-gate, finer grid, mlp_s3 at top tier,
     mlp_deep vs lgb_rem at tier 2.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

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

with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)
with open('results/mlp_deep_final.pkl','rb') as f: dm = pickle.load(f)
with open('results/mlp_deep_s2_final.pkl','rb') as f: ds2 = pickle.load(f)
with open('results/mlp_deep_gelu_final.pkl','rb') as f: dg = pickle.load(f)
with open('results/mlp_deep_s3_final.pkl','rb') as f: ds3 = pickle.load(f)

# Neural base (proven weights)
w_base = {'mega37':0.4424,'rank_adj':0.0747,'oracle_log_v2':0.1540,
          'oracle_xgb_combined':0.1406,'oracle_xgb_v31':0.0589,
          'mlp_deep':0.0506,'mlp_s2':0.0539,'mlp_gelu':0.0198}
total = sum(w_base.values()); w_base = {k:v/total for k,v in w_base.items()}

oof_c = {
    'mega37':  d37['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'mlp_deep': dm['oof'][id2], 'mlp_s2': ds2['oof'][id2], 'mlp_gelu': dg['oof'][id2],
    'mlp_s3': ds3['oof'][id2],
}
test_c = {
    'mega37':  d37['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'mlp_deep': dm['test'][te_id2], 'mlp_s2': ds2['test'][te_id2], 'mlp_gelu': dg['test'][te_id2],
    'mlp_s3': ds3['test'][te_id2],
}

base_oof  = np.clip(sum(w_base.get(k,0)*oof_c[k]  for k in oof_c), 0, None)
base_test = np.clip(sum(w_base.get(k,0)*test_c[k] for k in test_c), 0, None)
base_mae  = np.mean(np.abs(base_oof - y_true))
print(f"Neural base OOF: {base_mae:.5f}")

# Cascade resources
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]
mlp_deep_oof  = dm['oof'][id2]; mlp_deep_test = dm['test'][te_id2]
mlp_s3_oof    = ds3['oof'][id2]; mlp_s3_test   = ds3['test'][te_id2]
mlp_s2_oof    = ds2['oof'][id2]; mlp_s2_test   = ds2['test'][te_id2]

prev_best = 8.36746
best_mae = prev_best
best_oof_p = None; best_test_p = None; best_cfg = None

# Search 1: Scipy continuous 3-gate with mlp_deep top tier
print(f"\n[Search1] Scipy 3-gate (mlp_deep top)", flush=True)
def triple_mae(params, spec3_oof):
    p1, w1, p2, w2, p3, w3 = params
    p1=np.clip(p1,0.06,0.20); w1=np.clip(w1,0,0.10)
    p2=np.clip(p2,0.15,0.45); w2=np.clip(w2,0,0.10)
    p3=np.clip(p3,0.30,0.70); w3=np.clip(w3,0,0.20)
    if p2<=p1 or p3<=p2: return 9999
    m1=(clf_oof>p1).astype(float); m2=(clf_oof>p2).astype(float); m3=(clf_oof>p3).astype(float)
    b1=(1-m1*w1)*base_oof+m1*w1*rh_oof
    b2=(1-m2*w2)*b1+m2*w2*rm_oof
    b3=(1-m3*w3)*b2+m3*w3*spec3_oof
    return np.mean(np.abs(b3-y_true))

for spec3_name, spec3_oof, spec3_test in [
    ('mlp_deep', mlp_deep_oof, mlp_deep_test),
    ('mlp_s3', mlp_s3_oof, mlp_s3_test),
    ('mlp_s2', mlp_s2_oof, mlp_s2_test),
]:
    inits = [[0.11, 0.025, 0.27, 0.040, 0.45, 0.090],
             [0.11, 0.020, 0.27, 0.035, 0.45, 0.090],
             [0.10, 0.020, 0.27, 0.040, 0.50, 0.080],
             [0.12, 0.025, 0.27, 0.035, 0.40, 0.100]]
    for init in inits:
        res = minimize(lambda p: triple_mae(p, spec3_oof), init, method='Nelder-Mead',
                       options={'xatol':1e-6,'fatol':1e-7,'maxiter':3000})
        p1c,w1c,p2c,w2c,p3c,w3c = res.x
        p1c=np.clip(p1c,0.06,0.20); w1c=np.clip(w1c,0,0.10)
        p2c=np.clip(p2c,0.15,0.45); w2c=np.clip(w2c,0,0.10)
        p3c=np.clip(p3c,0.30,0.70); w3c=np.clip(w3c,0,0.20)
        m1=(clf_oof>p1c).astype(float); m2=(clf_oof>p2c).astype(float); m3=(clf_oof>p3c).astype(float)
        m1t=(clf_test>p1c).astype(float); m2t=(clf_test>p2c).astype(float); m3t=(clf_test>p3c).astype(float)
        b1=(1-m1*w1c)*base_oof+m1*w1c*rh_oof; b1t=(1-m1t*w1c)*base_test+m1t*w1c*rh_test
        b2=(1-m2*w2c)*b1+m2*w2c*rm_oof; b2t=(1-m2t*w2c)*b1t+m2t*w2c*rm_test
        b3=(1-m3*w3c)*b2+m3*w3c*spec3_oof; b3t=(1-m3t*w3c)*b2t+m3t*w3c*spec3_test
        mm=np.mean(np.abs(b3-y_true))
        print(f"  [{spec3_name}] p1={p1c:.3f}w{w1c:.3f}+p2={p2c:.3f}w{w2c:.3f}+p3={p3c:.3f}w{w3c:.3f}  OOF={mm:.5f}", flush=True)
        if mm<best_mae:
            best_mae=mm; best_cfg=f'scipy3_{spec3_name}_{mm:.5f}'
            best_oof_p=b3; best_test_p=b3t
            print(f"★ NEW BEST: {best_mae:.5f}")

# Search 2: Fine grid around best triple gate params
print(f"\n[Search2] Fine grid around p0.11/p0.27/p0.45 (step=0.001)", flush=True)
for p1 in np.arange(0.10, 0.14, 0.005):
    m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
    for w1 in np.arange(0.018, 0.035, 0.002):
        b1 = (1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_test
        for p2 in np.arange(0.24, 0.32, 0.005):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.030, 0.055, 0.003):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof; b2t=(1-m2t*w2)*b1t+m2t*w2*rm_test
                for p3 in [0.40, 0.43, 0.45, 0.47, 0.50, 0.55]:
                    if p3<=p2: continue
                    m3=(clf_oof>p3).astype(float); m3t=(clf_test>p3).astype(float)
                    for w3 in np.arange(0.060, 0.130, 0.010):
                        b3=(1-m3*w3)*b2+m3*w3*mlp_deep_oof
                        mm=np.mean(np.abs(b3-y_true))
                        if mm<best_mae:
                            best_mae=mm; best_cfg=f'p{p1:.3f}w{w1:.3f}+p{p2:.3f}w{w2:.3f}+p{p3:.2f}w{w3:.3f}'
                            best_oof_p=b3
                            b3t=(1-m3t*w3)*b2t+m3t*w3*mlp_deep_test
                            best_test_p=b3t
                            print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

print(f"\n=== FINAL ===")
print(f"  prev={prev_best:.5f}  best={best_mae:.5f}  ({best_cfg})")

ref_prev = 8.37851
if best_mae < ref_prev - 0.00005 and best_test_p is not None:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_p)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_triple_gate_refine_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print("No new improvement (best is existing file)")
print("Done.")
