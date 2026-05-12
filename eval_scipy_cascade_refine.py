"""
Refine cascade gate on scipy-optimized base (OOF=8.38110).
Prev best with gate: 8.37851 (dual p0.11/p0.25 from old base).
Now try same gate search on new scipy base.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
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

with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)

# Scipy-optimized base weights (L-BFGS-B result)
w_scipy = np.array([0.4997, 0.1019, 0.0, 0.0, 0.0, 0.1338, 0.1741, 0.0991])
w_scipy = w_scipy / w_scipy.sum()

keys = ['mega33','rank_adj','iter_r1','iter_r2','iter_r3','oracle_xgb','oracle_lv2','oracle_rem']
C_oof = np.column_stack([
    d['meta_avg_oof'][id2],
    np.load('results/ranking/rank_adj_oof.npy')[id2],
    np.load('results/iter_pseudo/round1_oof.npy')[id2],
    np.load('results/iter_pseudo/round2_oof.npy')[id2],
    np.load('results/iter_pseudo/round3_oof.npy')[id2],
    np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
])
C_test = np.column_stack([
    d['meta_avg_test'][te_id2],
    np.load('results/ranking/rank_adj_test.npy')[te_id2],
    np.load('results/iter_pseudo/round1_test.npy')[te_id2],
    np.load('results/iter_pseudo/round2_test.npy')[te_id2],
    np.load('results/iter_pseudo/round3_test.npy')[te_id2],
    np.load('results/oracle_seq/test_C_xgb.npy'),
    np.load('results/oracle_seq/test_C_log_v2.npy'),
    np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
])

base_oof  = np.clip(C_oof  @ w_scipy, 0, None)
base_test = np.clip(C_test @ w_scipy, 0, None)
base_mae = np.mean(np.abs(base_oof - y_true))
print(f"Scipy base OOF: {base_mae:.5f}")

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best = 8.37851
best_mae = prev_best
best_cfg = None; best_oof_pred = None; best_test_pred = None

p_vals = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15,
          0.18, 0.20, 0.22, 0.25, 0.28, 0.30]
masks    = {p: (clf_oof  > p).astype(float) for p in p_vals}
masks_te = {p: (clf_test > p).astype(float) for p in p_vals}

print(f"\n[Search1] Fine dual gate on scipy base", flush=True)
for p1 in [0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13]:
    for w1 in np.arange(0.010, 0.070, 0.005):
        b1    = (1-masks[p1]*w1)*base_oof  + masks[p1]*w1*lgb_rh_oof
        b1_te = (1-masks_te[p1]*w1)*base_test + masks_te[p1]*w1*lgb_rh_test
        for p2 in [0.20, 0.22, 0.25, 0.28, 0.30]:
            if p2 <= p1: continue
            for w2 in np.arange(0.010, 0.070, 0.005):
                blend = (1-masks[p2]*w2)*b1 + masks[p2]*w2*lgb_rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'dual p{p1}_h_w{w1:.3f}+p{p2}_m_w{w2:.3f}'
                    best_oof_pred  = blend
                    best_test_pred = (1-masks_te[p2]*w2)*b1_te + masks_te[p2]*w2*lgb_rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-base_mae:+.5f}", flush=True)

print(f"\n[Search2] Scipy joint optimize: base weights + cascade gate simultaneously", flush=True)
# Fix gate structure, optimize w1/w2 continuously with scipy
def dual_gate_mae(params):
    w1, w2 = params
    w1 = np.clip(w1, 0, 0.3); w2 = np.clip(w2, 0, 0.3)
    b1 = (1-masks[0.11]*w1)*base_oof + masks[0.11]*w1*lgb_rh_oof
    blend = (1-masks[0.25]*w2)*b1 + masks[0.25]*w2*lgb_rm_oof
    return np.mean(np.abs(blend - y_true))

res = minimize(dual_gate_mae, [0.03, 0.03], method='Nelder-Mead',
               options={'xatol':1e-5, 'fatol':1e-6, 'maxiter':1000})
w1_opt, w2_opt = np.clip(res.x, 0, 0.5)
b1 = (1-masks[0.11]*w1_opt)*base_oof + masks[0.11]*w1_opt*lgb_rh_oof
b1_te = (1-masks_te[0.11]*w1_opt)*base_test + masks_te[0.11]*w1_opt*lgb_rh_test
blend_opt = (1-masks[0.25]*w2_opt)*b1 + masks[0.25]*w2_opt*lgb_rm_oof
blend_te_opt = (1-masks_te[0.25]*w2_opt)*b1_te + masks_te[0.25]*w2_opt*lgb_rm_test
mm_opt = np.mean(np.abs(blend_opt - y_true))
print(f"  Nelder-Mead: w1={w1_opt:.4f} w2={w2_opt:.4f}  OOF={mm_opt:.5f}  delta={mm_opt-base_mae:+.5f}", flush=True)
if mm_opt < best_mae:
    best_mae = mm_opt; best_cfg = f'scipy_gate w1={w1_opt:.4f} w2={w2_opt:.4f}'
    best_oof_pred = blend_opt; best_test_pred = blend_te_opt

if best_cfg:
    print(f"\n=== IMPROVED: {best_mae:.5f}  delta={best_mae-base_mae:+.5f} vs prev={prev_best:.5f} ===")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_scipy_refined_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"\nNo improvement over {prev_best:.5f}")
print("Done.")
