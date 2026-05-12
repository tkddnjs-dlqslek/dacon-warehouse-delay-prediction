"""
Refine cascade gate on mega37 base (OOF=8.37459).
Best so far: p0.11_w0.025+p0.22_w0.025 → 8.37165.
Try: finer step, scipy continuous, include mega33 component.
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

# Proven mega37 base weights
base_keys = ['mega34', 'mega37', 'rank_adj', 'oracle_log_v2', 'oracle_xgb_combined', 'oracle_xgb_v31']
base_w    = np.array([0.0558, 0.5269, 0.0712, 0.1597, 0.1237, 0.0583])
base_w   /= base_w.sum()

base_oof_parts = {
    'mega33':  d33['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'mega37':  d37['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':      np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':     np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_xgb':         np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_xgb_rem':     np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':          np.load('results/oracle_seq/oof_seqC_cb.npy'),
}
base_test_parts = {
    'mega33':  d33['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'mega37':  d37['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':      np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':     np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_xgb':         np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_xgb_rem':     np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_cb':          np.load('results/oracle_seq/test_C_cb.npy'),
}

base_oof  = np.clip(sum(base_w[i]*base_oof_parts[k]  for i,k in enumerate(base_keys)), 0, None)
base_test = np.clip(sum(base_w[i]*base_test_parts[k] for i,k in enumerate(base_keys)), 0, None)
base_mae  = np.mean(np.abs(base_oof - y_true))
print(f"Mega37 base OOF: {base_mae:.5f}")

# Cascade resources
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best = 8.37165
best_mae = prev_best
best_oof_pred = None; best_test_pred = None; best_cfg = None

# Search 1: Fine grid around best found (p0.11/p0.22)
print(f"\n[Search1] Fine grid around p0.11/p0.22 + wider p2 range", flush=True)
for p1 in np.arange(0.08, 0.16, 0.01):
    m1    = (clf_oof  > p1).astype(float)
    m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.005, 0.060, 0.002):
        b1    = (1-m1*w1)*base_oof  + m1*w1*rh_oof
        b1_te = (1-m1_te*w1)*base_test + m1_te*w1*rh_test
        for p2 in np.arange(0.16, 0.36, 0.01):
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float)
            m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.005, 0.060, 0.002):
                blend = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'p{p1:.2f}_w{w1:.3f}+p{p2:.2f}_w{w2:.3f}'
                    best_oof_pred = blend
                    best_test_pred = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

print(f"\n[Search2] Scipy continuous gate optimization", flush=True)
def gate_mae(params):
    p1, w1, p2, w2 = params
    p1 = np.clip(p1, 0.05, 0.20); w1 = np.clip(w1, 0, 0.15)
    p2 = np.clip(p2, 0.15, 0.50); w2 = np.clip(w2, 0, 0.15)
    if p2 <= p1: return 9999
    m1 = (clf_oof > p1).astype(float); m2 = (clf_oof > p2).astype(float)
    b1 = (1-m1*w1)*base_oof + m1*w1*rh_oof
    blend = (1-m2*w2)*b1 + m2*w2*rm_oof
    return np.mean(np.abs(blend - y_true))

for init in [[0.11, 0.025, 0.22, 0.025], [0.09, 0.015, 0.20, 0.030],
             [0.12, 0.020, 0.25, 0.025], [0.10, 0.020, 0.18, 0.020]]:
    res = minimize(gate_mae, init, method='Nelder-Mead',
                   options={'xatol':1e-5, 'fatol':1e-6, 'maxiter':2000})
    p1c, w1c, p2c, w2c = res.x
    p1c = np.clip(p1c,0.05,0.20); w1c=np.clip(w1c,0,0.15)
    p2c = np.clip(p2c,0.15,0.50); w2c=np.clip(w2c,0,0.15)
    m1 = (clf_oof > p1c).astype(float); m2 = (clf_oof > p2c).astype(float)
    m1_te = (clf_test > p1c).astype(float); m2_te = (clf_test > p2c).astype(float)
    b1 = (1-m1*w1c)*base_oof + m1*w1c*rh_oof
    blend = (1-m2*w2c)*b1 + m2*w2c*rm_oof
    mm = np.mean(np.abs(blend - y_true))
    print(f"  Nelder-Mead: p1={p1c:.3f} w1={w1c:.4f} p2={p2c:.3f} w2={w2c:.4f}  OOF={mm:.5f}", flush=True)
    if mm < best_mae:
        best_mae = mm; best_cfg = f'scipy_gate p{p1c:.3f}w{w1c:.4f}+p{p2c:.3f}w{w2c:.4f}'
        best_oof_pred = blend
        b1_te = (1-m1_te*w1c)*base_test + m1_te*w1c*rh_test
        best_test_pred = (1-m2_te*w2c)*b1_te + m2_te*w2c*rm_test
        print(f"★ NEW BEST: {best_mae:.5f}", flush=True)

print(f"\n[Search3] Also try mega33 blended in with mega37 base", flush=True)
# Try adding mega33 as a small component (might add diversity)
for m33_w in [0.05, 0.08, 0.10, 0.12, 0.15]:
    # Rescale base_w to accommodate mega33
    scale = 1 - m33_w
    b_oof  = scale*base_oof  + m33_w*base_oof_parts['mega33']
    b_test = scale*base_test + m33_w*base_test_parts['mega33']
    b_mae  = np.mean(np.abs(b_oof - y_true))
    # Apply best cascade gate found
    if best_cfg:
        # Reproduce best cascade
        m1 = (clf_oof > 0.11).astype(float); m2 = (clf_oof > 0.22).astype(float)
        m1_te = (clf_test > 0.11).astype(float); m2_te = (clf_test > 0.22).astype(float)
        bl = (1-m1*0.025)*b_oof + m1*0.025*rh_oof
        bl = (1-m2*0.025)*bl    + m2*0.025*rm_oof
        bl_te = (1-m1_te*0.025)*b_test + m1_te*0.025*rh_test
        bl_te = (1-m2_te*0.025)*bl_te  + m2_te*0.025*rm_test
        mm = np.mean(np.abs(bl - y_true))
        print(f"  m33_blend={m33_w}: base={b_mae:.5f}  +gate={mm:.5f}", flush=True)
        if mm < best_mae:
            best_mae = mm; best_cfg = f'm33_blend{m33_w}+gate'
            best_oof_pred = bl; best_test_pred = bl_te
            print(f"★ NEW BEST: {best_mae:.5f}", flush=True)

print(f"\n=== FINAL ===")
print(f"  Prev best OOF: {prev_best:.5f}")
print(f"  This run best: {best_mae:.5f}  ({best_cfg})")

ref_prev = 8.37851
if best_mae < ref_prev - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred if best_test_pred is not None else base_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mega37_refined_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print("No further improvement needed (existing file is best).")
print("Done.")
