"""
Neural-free blend: mega37 + oracle_combined/v31/log_v2 + rank_adj + cascade gate.
No mlp_deep/s2/gelu — they hurt LB on unseen layouts.
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

with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

oof_parts = {
    'mega37':   d37['meta_avg_oof'][id2],
    'mega34':   d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_log_v2':   np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb':      np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_cb':       np.load('results/oracle_seq/oof_seqC_cb.npy'),
}
test_parts = {
    'mega37':   d37['meta_avg_test'][te_id2],
    'mega34':   d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_log_v2':   np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb':      np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_cb':       np.load('results/oracle_seq/test_C_cb.npy'),
}

keys = list(oof_parts.keys())
oofs = np.stack([oof_parts[k] for k in keys], axis=1)
tests = np.stack([test_parts[k] for k in keys], axis=1)

# Individual MAEs
for k in keys:
    mae = np.mean(np.abs(oof_parts[k] - y_true))
    print(f"  {k}: {mae:.5f}")

# Scipy L-BFGS-B blend optimization
def mae_blend(w):
    w = np.clip(w, 0, 1); w = w / w.sum()
    pred = np.clip((oofs * w).sum(axis=1), 0, None)
    return np.mean(np.abs(pred - y_true))

best_mae = 999; best_w = None
for _ in range(10):
    w0 = np.random.dirichlet(np.ones(len(keys)))
    res = minimize(mae_blend, w0, method='L-BFGS-B',
                   bounds=[(0,1)]*len(keys), options={'maxiter':2000,'ftol':1e-9})
    if res.fun < best_mae:
        best_mae = res.fun; best_w = res.x

best_w = np.clip(best_w, 0, 1); best_w /= best_w.sum()
print(f"\nOptimized blend OOF: {best_mae:.5f}")
for k, w in zip(keys, best_w):
    if w > 0.005: print(f"  {k}: {w:.4f}")

blend_oof  = np.clip((oofs  * best_w).sum(axis=1), 0, None)
blend_test = np.clip((tests * best_w).sum(axis=1), 0, None)

# Add cascade gate (no neural at tier-3)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

print(f"\n[Cascade gate search (no neural)]", flush=True)
best_gate_mae = best_mae; best_gate_oof = blend_oof; best_gate_test = blend_test; best_cfg = 'no_gate'

for p1 in np.arange(0.08, 0.18, 0.01):
    m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
    for w1 in np.arange(0.010, 0.060, 0.005):
        b1 = (1-m1*w1)*blend_oof + m1*w1*rh_oof
        b1t= (1-m1t*w1)*blend_test + m1t*w1*rh_te
        for p2 in np.arange(0.20, 0.40, 0.02):
            if p2 <= p1: continue
            m2 = (clf_oof>p2).astype(float); m2t = (clf_test>p2).astype(float)
            for w2 in np.arange(0.020, 0.080, 0.005):
                b2 = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(b2 - y_true))
                if mm < best_gate_mae:
                    best_gate_mae = mm; best_cfg = f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    best_gate_oof = b2
                    b2t = (1-m2t*w2)*b1t + m2t*w2*rm_te
                    best_gate_test = b2t
                    print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

print(f"\nFinal OOF: {best_gate_mae:.5f}  ({best_cfg})")

# Save
sample_sub = pd.read_csv('sample_submission.csv')
sub = np.maximum(0, best_gate_test)
sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
fname = f'submission_no_neural_OOF{best_gate_mae:.5f}.csv'
sub_df.to_csv(fname, index=False)
print(f"*** SAVED: {fname} ***")
print("Done.")
