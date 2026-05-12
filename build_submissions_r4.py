"""
Round 4: Cross-round ensemble + systematic gate experiments.
Key observations so far:
  - Best OOF: H/M (mega33+34, OOF=8.37789)
  - oracle_v31 adds marginal OOF gain (N: 8.37818 vs A: 8.37854)
  - No gate (U) will test if gate helps/hurts LB
  - Geometric mean (S) tests robustness
Strategy:
  X1: Ensemble of top-3 OOF submissions (H+M+N test preds)
  X2: mega33 + oracle_xgb + oracle_rem only (no lv2, simpler oracle set)
  X3: cascade_refined3 with gate, but using oracle_rem instead of oracle_lv2
  X4: Weighted avg of R1+R2 best (H, N, O, A) with scipy on OOF
  X5: mega34 only + gate (is mega34 standalone better than mega33 standalone?)
  X6: mega33 + mega34 equal weight (0.5/0.5) then oracle on top
  X7: oracle_cb standalone + gate (any standalone oracle value?)
  X8: 4-way blend H+N+O+M (all R1+R2 top4 by OOF)
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

sample_sub = pd.read_csv('sample_submission.csv')

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'mega34':     d34['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/oof_seqC_cb.npy'),
    'oracle_v31': np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
}
ct = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'mega34':     d34['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/test_C_cb.npy'),
    'oracle_v31': np.load('results/oracle_seq/test_C_xgb_v31.npy'),
}

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

saved = []
np.random.seed(789)

def mae(pred): return np.mean(np.abs(pred - y_true))

def gate_search(base_oof, base_test, label, p1_range=None, p2_range=None):
    if p1_range is None: p1_range = np.arange(0.06, 0.18, 0.01)
    if p2_range is None: p2_range = np.arange(0.15, 0.45, 0.02)
    best = mae(base_oof); best_oof = base_oof; best_te = base_test; best_cfg = 'no_gate'
    for p1 in p1_range:
        m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
        for w1 in np.arange(0.010, 0.070, 0.005):
            b1 = (1-m1*w1)*base_oof + m1*w1*rh_oof
            b1t = (1-m1t*w1)*base_test + m1t*w1*rh_te
            for p2 in p2_range:
                if p2 <= p1: continue
                m2 = (clf_oof>p2).astype(float); m2t = (clf_test>p2).astype(float)
                for w2 in np.arange(0.010, 0.080, 0.005):
                    b2 = (1-m2*w2)*b1 + m2*w2*rm_oof
                    mm = mae(b2)
                    if mm < best:
                        best = mm; best_cfg = f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        best_oof = b2; best_te = (1-m2t*w2)*b1t + m2t*w2*rm_te
    print(f"  [{label}] gate={best_cfg}  OOF={best:.5f}")
    return best_oof, best_te, best

def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

def scipy_blend(keys, n_trials=15, bounds=None):
    oofs_ = np.stack([co[k] for k in keys], axis=1)
    tes_  = np.stack([ct[k] for k in keys], axis=1)
    if bounds is None: bounds = [(0,1)]*len(keys)
    def obj(w):
        w = np.clip(w,0,1); w /= w.sum()
        return np.mean(np.abs(np.clip(oofs_@w,0,None) - y_true))
    best = 999; best_w = None
    for _ in range(n_trials):
        w0 = np.random.dirichlet(np.ones(len(keys)))
        r = minimize(obj, w0, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter':3000,'ftol':1e-10})
        if r.fun < best: best = r.fun; best_w = r.x
    best_w = np.clip(best_w,0,1); best_w /= best_w.sum()
    oof_ = np.clip(oofs_@best_w, 0, None)
    te_  = np.clip(tes_@best_w, 0, None)
    for k,w in zip(keys, best_w):
        if w > 0.005: print(f"    {k}: {w:.4f}")
    return oof_, te_, best, best_w

# ─────────────────────────────────────────────────────────────
# Precompute gated OOF/test for top previous models to enable meta-blending
# ─────────────────────────────────────────────────────────────
print("Precomputing base model OOFs for meta-blend...")

# H = mega33+34 optimized
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_H = np.stack([co[k] for k in keys_H], axis=1)
tes_H  = np.stack([ct[k] for k in keys_H], axis=1)
def obj_H(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_H@w,0,None)-y_true))
best_H=999; best_wH=None
for _ in range(20):
    w0=np.random.dirichlet(np.ones(6))
    r=minimize(obj_H,w0,method='L-BFGS-B',bounds=[(0,1)]*6,options={'maxiter':3000,'ftol':1e-10})
    if r.fun<best_H: best_H=r.fun; best_wH=r.x
best_wH=np.clip(best_wH,0,1); best_wH/=best_wH.sum()
bo_H=np.clip(oofs_H@best_wH,0,None); bt_H=np.clip(tes_H@best_wH,0,None)
bo_Hg, bt_Hg, _ = gate_search(bo_H, bt_H, 'H_precomp')

# N = oracle_v31 included
keys_N = ['mega33','rank_adj','oracle_xgb','oracle_v31','oracle_lv2']
bo_N, bt_N, _, _ = scipy_blend(keys_N, n_trials=20)
bo_Ng, bt_Ng, _ = gate_search(bo_N, bt_N, 'N_precomp')

# A = cascade_refined3 base
w_A = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_A/=w_A.sum()
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_A = np.stack([co[k] for k in keys_A], axis=1)
tes_A  = np.stack([ct[k] for k in keys_A], axis=1)
bo_A0=np.clip(oofs_A@w_A,0,None); bt_A0=np.clip(tes_A@w_A,0,None)
bo_Ag, bt_Ag, _ = gate_search(bo_A0, bt_A0, 'A_precomp')

print("Precompute done.\n")

# ─────────────────────────────────────────────────────────────
# MODEL X1: Ensemble of top-3 by OOF (H+N+A gated predictions)
# ─────────────────────────────────────────────────────────────
print("=== MODEL X1: Top-3 OOF meta-blend (H+N+A) ===")
meta_oofs  = np.stack([bo_Hg, bo_Ng, bo_Ag], axis=1)
meta_tests = np.stack([bt_Hg, bt_Ng, bt_Ag], axis=1)
def obj_X1(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(meta_oofs@w - y_true))
best_X1=999; best_wX1=None
for _ in range(10):
    w0=np.random.dirichlet(np.ones(3))
    r=minimize(obj_X1,w0,method='L-BFGS-B',bounds=[(0,1)]*3,options={'maxiter':1000})
    if r.fun<best_X1: best_X1=r.fun; best_wX1=r.x
best_wX1=np.clip(best_wX1,0,1); best_wX1/=best_wX1.sum()
x1_oof=meta_oofs@best_wX1; x1_te=meta_tests@best_wX1
print(f"  X1 OOF: {best_X1:.5f}  w: H={best_wX1[0]:.3f} N={best_wX1[1]:.3f} A={best_wX1[2]:.3f}")
save_sub(x1_te, best_X1, 'X1_HNA_blend')

# ─────────────────────────────────────────────────────────────
# MODEL X2: mega33 + oracle_xgb + oracle_rem (no lv2, simpler)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL X2: mega33+oracle_xgb+oracle_rem (no lv2) ===")
keys_X2 = ['mega33','rank_adj','oracle_xgb','oracle_rem']
bo_X2, bt_X2, bm_X2, _ = scipy_blend(keys_X2, n_trials=20)
print(f"  blend OOF: {bm_X2:.5f}")
bo_X2g, bt_X2g, bm_X2g = gate_search(bo_X2, bt_X2, 'X2_gate')
save_sub(bt_X2g, bm_X2g, 'X2_no_lv2')

# ─────────────────────────────────────────────────────────────
# MODEL X3: cascade_refined3 weights but oracle_rem replaces oracle_lv2
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL X3: ref3 weights, oracle_rem instead of oracle_lv2 ===")
# Original ref3: mega33=0.4997, rank_adj=0.1019, oracle_xgb=0.1338, oracle_lv2=0.1741, oracle_rem=0.0991
# Swap: oracle_lv2 → oracle_rem weight, oracle_rem gets lv2 weight
keys_X3a = ['mega33','rank_adj','oracle_xgb','oracle_rem']
w_X3a = np.array([0.4997,0.1019,0.1338,0.1741+0.0991]); w_X3a/=w_X3a.sum()
oofs_X3a = np.stack([co[k] for k in keys_X3a], axis=1)
tes_X3a  = np.stack([ct[k] for k in keys_X3a], axis=1)
bo_X3a=np.clip(oofs_X3a@w_X3a,0,None); bt_X3a=np.clip(tes_X3a@w_X3a,0,None)
bm_X3a=mae(bo_X3a)
print(f"  fixed-weight OOF: {bm_X3a:.5f}")
# Also try scipy on same 4 components
bo_X3b, bt_X3b, bm_X3b, _ = scipy_blend(keys_X3a, n_trials=15)
print(f"  scipy OOF: {bm_X3b:.5f}")
# Use whichever is better
if bm_X3b <= bm_X3a:
    bo_X3, bt_X3, bm_X3 = bo_X3b, bt_X3b, bm_X3b
else:
    bo_X3, bt_X3, bm_X3 = bo_X3a, bt_X3a, bm_X3a
bo_X3g, bt_X3g, bm_X3g = gate_search(bo_X3, bt_X3, 'X3_gate')
save_sub(bt_X3g, bm_X3g, 'X3_rem_dominant')

# ─────────────────────────────────────────────────────────────
# MODEL X4: mega34-only standalone + gate
# (Test: is mega34 standalone stronger than mega33 standalone?)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL X4: mega34-only standalone ===")
bo_X4 = co['mega34'].copy(); bt_X4 = ct['mega34'].copy()
bm_X4 = mae(bo_X4)
print(f"  mega34 standalone OOF: {bm_X4:.5f}")
bo_X4g, bt_X4g, bm_X4g = gate_search(bo_X4, bt_X4, 'X4_gate')
save_sub(bt_X4g, bm_X4g, 'X4_mega34only')

# ─────────────────────────────────────────────────────────────
# MODEL X5: mega33+34 equal (0.5/0.5) then oracle blend on top
# Different from scipy-optimized H — forces equal mega weight
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL X5: mega33+34 equal, oracle on top ===")
mega_avg_oof  = (co['mega33'] + co['mega34']) / 2
mega_avg_test = (ct['mega33'] + ct['mega34']) / 2
# Now blend mega_avg + oracle components via scipy
# Treat mega_avg as one component
keys_X5 = ['oracle_xgb','oracle_lv2','oracle_rem']
oofs_X5_oracle = np.stack([co[k] for k in keys_X5], axis=1)
tes_X5_oracle  = np.stack([ct[k] for k in keys_X5], axis=1)
def obj_X5(params):
    # params[0]=w_mega, params[1:]=oracle weights (sum to 1-w_mega)
    wm = np.clip(params[0], 0.3, 0.8)
    wo = np.clip(params[1:],0,1); wo /= (wo.sum() + 1e-9); wo *= (1-wm)
    pred = np.clip(wm*mega_avg_oof + oofs_X5_oracle@wo, 0, None)
    return np.mean(np.abs(pred - y_true))
best_X5=999; best_parX5=None
for _ in range(20):
    wm0 = np.random.uniform(0.3, 0.7)
    wo0 = np.random.dirichlet(np.ones(3)) * (1-wm0)
    p0 = np.array([wm0] + list(wo0))
    r = minimize(obj_X5, p0, method='L-BFGS-B',
                 bounds=[(0.3,0.8),(0,0.4),(0,0.4),(0,0.4)],
                 options={'maxiter':2000})
    if r.fun < best_X5: best_X5=r.fun; best_parX5=r.x
wm5 = np.clip(best_parX5[0],0.3,0.8)
wo5 = np.clip(best_parX5[1:],0,1); wo5 /= (wo5.sum()+1e-9); wo5 *= (1-wm5)
bo_X5 = np.clip(wm5*mega_avg_oof + oofs_X5_oracle@wo5, 0, None)
bt_X5 = np.clip(wm5*mega_avg_test + tes_X5_oracle@wo5, 0, None)
print(f"  X5 OOF: {best_X5:.5f}  w_mega={wm5:.3f}  oracle_w: xgb={wo5[0]:.3f} lv2={wo5[1]:.3f} rem={wo5[2]:.3f}")
bo_X5g, bt_X5g, bm_X5g = gate_search(bo_X5, bt_X5, 'X5_gate')
save_sub(bt_X5g, bm_X5g, 'X5_mega_eq_oracle')

# ─────────────────────────────────────────────────────────────
# MODEL X6: 4-way blend of H+N+A+M gated OOFs
# (All round 1+2 top by OOF)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL X6: 4-way blend H+N+A+M ===")
# M = same base as H with same gate, so bo_Hg ≈ bo_Mg
# But recompute M with same methodology
bo_Mg = bo_Hg  # M and H share same scipy optimized base — use H gated
bt_Mg = bt_Hg

meta4_oofs  = np.stack([bo_Hg, bo_Ng, bo_Ag, bo_Mg], axis=1)
meta4_tests = np.stack([bt_Hg, bt_Ng, bt_Ag, bt_Mg], axis=1)
def obj_X6(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(meta4_oofs@w - y_true))
best_X6=999; best_wX6=None
for _ in range(10):
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_X6,w0,method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':1000})
    if r.fun<best_X6: best_X6=r.fun; best_wX6=r.x
best_wX6=np.clip(best_wX6,0,1); best_wX6/=best_wX6.sum()
x6_oof=meta4_oofs@best_wX6; x6_te=meta4_tests@best_wX6
print(f"  X6 OOF: {best_X6:.5f}  w: {best_wX6}")
save_sub(x6_te, best_X6, 'X6_4way_blend')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 4 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 4 Total: {len(saved)} new files")
