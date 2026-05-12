"""
Round 2: More diverse submissions.
Key insight from round 1: seqA/B → weight≈0, oracle_cb ≈ oracle_xgb in blend.
New angles:
  I: Constrained blend (force oracle_cb >= 0.15, force diversity)
  J: oracle-only blend (no mega, no rank_adj)
  K: mega33-heavy (0.7+) + minimal oracle
  L: Wider gate (less aggressive, fewer activations)
  M: H (mega33+34) cascade-refined further with wider search
  N: oracle_xgb + oracle_cb + oracle_lv2 only (no mega, no rank)
  O: Scipy with penalty for correlated components
  P: Blend of different-OOF predictions (H + A_nogate)
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
np.random.seed(123)

def mae(pred): return np.mean(np.abs(pred - y_true))

def blend_fixed(weights_dict):
    """blend with explicit dict of {key: weight}"""
    w_arr = np.array(list(weights_dict.values()))
    w_arr /= w_arr.sum()
    keys = list(weights_dict.keys())
    oof_  = sum(w_arr[i]*co[k] for i,k in enumerate(keys))
    test_ = sum(w_arr[i]*ct[k] for i,k in enumerate(keys))
    return np.clip(oof_,0,None), np.clip(test_,0,None)

def gate_search(base_oof, base_test, label, p1_range=None, p2_range=None):
    if p1_range is None: p1_range = np.arange(0.06,0.18,0.01)
    if p2_range is None: p2_range = np.arange(0.15,0.45,0.02)
    best=mae(base_oof); best_oof=base_oof; best_te=base_test; best_cfg='no_gate'
    for p1 in p1_range:
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.010,0.070,0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof
            b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in p2_range:
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.010,0.080,0.005):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; best_cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        best_oof=b2; best_te=(1-m2t*w2)*b1t+m2t*w2*rm_te
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

# ─────────────────────────────────────────────────────────────
# MODEL I: Constrained blend – force oracle_cb ≥ 0.15
# mega33 + rank_adj + oracle_xgb + oracle_cb(≥0.15) + oracle_lv2
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL I: Constrained oracle_cb ≥ 0.15 ===")
keys_I = ['mega33','rank_adj','oracle_xgb','oracle_cb','oracle_lv2']
oofs_I = np.stack([co[k] for k in keys_I], axis=1)
tes_I  = np.stack([ct[k] for k in keys_I], axis=1)
# idx 3 = oracle_cb, force >= 0.15
def obj_I(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_I@w,0,None) - y_true))
best_I=999; best_wI=None
for _ in range(20):
    w0=np.random.dirichlet(np.ones(5))
    w0[3]=max(w0[3], 0.15)  # bias oracle_cb
    r=minimize(obj_I,w0,method='L-BFGS-B',
               bounds=[(0,1),(0,1),(0,1),(0.10,0.50),(0,1)],
               options={'maxiter':3000,'ftol':1e-10})
    if r.fun<best_I: best_I=r.fun; best_wI=r.x
best_wI=np.clip(best_wI,0,1); best_wI/=best_wI.sum()
bo_I=np.clip(oofs_I@best_wI,0,None); bt_I=np.clip(tes_I@best_wI,0,None)
print(f"  blend OOF: {best_I:.5f}")
for k,w in zip(keys_I,best_wI): print(f"    {k}: {w:.4f}")
bo_Ig, bt_Ig, bm_Ig = gate_search(bo_I, bt_I, 'I_gate')
save_sub(bt_Ig, bm_Ig, 'I_cb_constrained')

# ─────────────────────────────────────────────────────────────
# MODEL J: Oracle-only blend (no mega33, no rank_adj)
# oracle_xgb + oracle_lv2 + oracle_rem + oracle_cb
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL J: Oracle-only (no mega33/rank_adj) ===")
keys_J = ['oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
oofs_J = np.stack([co[k] for k in keys_J], axis=1)
tes_J  = np.stack([ct[k] for k in keys_J], axis=1)
def obj_J(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_J@w,0,None)-y_true))
best_J=999; best_wJ=None
for _ in range(15):
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_J,w0,method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':2000})
    if r.fun<best_J: best_J=r.fun; best_wJ=r.x
best_wJ=np.clip(best_wJ,0,1); best_wJ/=best_wJ.sum()
bo_J=np.clip(oofs_J@best_wJ,0,None); bt_J=np.clip(tes_J@best_wJ,0,None)
print(f"  blend OOF: {best_J:.5f}")
for k,w in zip(keys_J,best_wJ): print(f"    {k}: {w:.4f}")
bo_Jg, bt_Jg, bm_Jg = gate_search(bo_J, bt_J, 'J_gate')
save_sub(bt_Jg, bm_Jg, 'J_oracle_only')

# ─────────────────────────────────────────────────────────────
# MODEL K: mega33-heavy (force mega33 ≥ 0.65) + minimal oracle
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL K: mega33-heavy (≥0.65) ===")
keys_K = ['mega33','rank_adj','oracle_xgb','oracle_lv2']
oofs_K = np.stack([co[k] for k in keys_K], axis=1)
tes_K  = np.stack([ct[k] for k in keys_K], axis=1)
def obj_K(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_K@w,0,None)-y_true))
best_K=999; best_wK=None
for _ in range(20):
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_K,w0,method='L-BFGS-B',
               bounds=[(0.55,0.85),(0,0.20),(0,0.25),(0,0.25)],
               options={'maxiter':3000,'ftol':1e-10})
    if r.fun<best_K: best_K=r.fun; best_wK=r.x
best_wK=np.clip(best_wK,0,1); best_wK/=best_wK.sum()
bo_K=np.clip(oofs_K@best_wK,0,None); bt_K=np.clip(tes_K@best_wK,0,None)
print(f"  blend OOF: {best_K:.5f}")
for k,w in zip(keys_K,best_wK): print(f"    {k}: {w:.4f}")
bo_Kg, bt_Kg, bm_Kg = gate_search(bo_K, bt_K, 'K_gate')
save_sub(bt_Kg, bm_Kg, 'K_mega33_heavy')

# ─────────────────────────────────────────────────────────────
# MODEL L: Narrow gate (p1>0.20, only very high confidence)
# Base A blend + narrow gate (fewer samples activated)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL L: Narrow-gate (p1≥0.20, conservative activation) ===")
# Reload A blend
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_A = np.stack([co[k] for k in keys_A], axis=1)
tes_A  = np.stack([ct[k] for k in keys_A], axis=1)
w_A = np.array([0.4997, 0.1019, 0.1338, 0.1741, 0.0991])  # cascade_refined3 weights
w_A /= w_A.sum()
bo_L = np.clip(oofs_A@w_A, 0, None); bt_L = np.clip(tes_A@w_A, 0, None)
# Search gate with high threshold only (p1 ≥ 0.20)
bo_Lg, bt_Lg, bm_Lg = gate_search(bo_L, bt_L, 'L_narrow',
    p1_range=np.arange(0.18, 0.35, 0.02),
    p2_range=np.arange(0.30, 0.55, 0.02))
save_sub(bt_Lg, bm_Lg, 'L_narrow_gate')

# ─────────────────────────────────────────────────────────────
# MODEL M: H (mega33+34) with finer gate search
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL M: mega33+mega34, full gate search ===")
keys_M = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_M = np.stack([co[k] for k in keys_M], axis=1)
tes_M  = np.stack([ct[k] for k in keys_M], axis=1)
def obj_M(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_M@w,0,None)-y_true))
best_M=999; best_wM=None
for _ in range(20):
    w0=np.random.dirichlet(np.ones(6))
    r=minimize(obj_M,w0,method='L-BFGS-B',bounds=[(0,1)]*6,options={'maxiter':3000,'ftol':1e-10})
    if r.fun<best_M: best_M=r.fun; best_wM=r.x
best_wM=np.clip(best_wM,0,1); best_wM/=best_wM.sum()
bo_M=np.clip(oofs_M@best_wM,0,None); bt_M=np.clip(tes_M@best_wM,0,None)
print(f"  blend OOF: {best_M:.5f}")
for k,w in zip(keys_M,best_wM): print(f"    {k}: {w:.4f}")
# Triple gate (3 tiers)
print("  [M_triple] searching 3-tier gate...", flush=True)
best_triple=best_M; bo_Mt=bo_M; bt_Mt=bt_M; cfg_t='no_gate'
for p1 in np.arange(0.08,0.16,0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.015,0.055,0.005):
        b1=(1-m1*w1)*bo_M+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_M+m1t*w1*rh_te
        for p2 in np.arange(0.20,0.38,0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.015,0.065,0.005):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_triple:
                    best_triple=mm; cfg_t=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    bo_Mt=b2
                    b2t=(1-m2t*w2)*b1t+m2t*w2*rm_te; bt_Mt=b2t
print(f"  [M_gate] gate={cfg_t}  OOF={best_triple:.5f}")
save_sub(bt_Mt, best_triple, 'M_mega34_gate')

# ─────────────────────────────────────────────────────────────
# MODEL N: oracle_v31 included (another oracle variant)
# mega33 + rank_adj + oracle_xgb + oracle_v31 + oracle_lv2
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL N: oracle_v31 included ===")
keys_N = ['mega33','rank_adj','oracle_xgb','oracle_v31','oracle_lv2']
oofs_N = np.stack([co[k] for k in keys_N], axis=1)
tes_N  = np.stack([ct[k] for k in keys_N], axis=1)
def obj_N(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_N@w,0,None)-y_true))
best_N=999; best_wN=None
for _ in range(20):
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_N,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000,'ftol':1e-10})
    if r.fun<best_N: best_N=r.fun; best_wN=r.x
best_wN=np.clip(best_wN,0,1); best_wN/=best_wN.sum()
bo_N=np.clip(oofs_N@best_wN,0,None); bt_N=np.clip(tes_N@best_wN,0,None)
print(f"  blend OOF: {best_N:.5f}")
for k,w in zip(keys_N,best_wN): print(f"    {k}: {w:.4f}")
bo_Ng, bt_Ng, bm_Ng = gate_search(bo_N, bt_N, 'N_gate')
save_sub(bt_Ng, bm_Ng, 'N_oracle_v31')

# ─────────────────────────────────────────────────────────────
# Compute A (cascade_refined3) gated blend for MODEL O
# ─────────────────────────────────────────────────────────────
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_A = np.stack([co[k] for k in keys_A], axis=1)
tes_A  = np.stack([ct[k] for k in keys_A], axis=1)
w_A = np.array([0.4997, 0.1019, 0.1338, 0.1741, 0.0991])
w_A /= w_A.sum()
bo_A0 = np.clip(oofs_A@w_A, 0, None); bt_A0 = np.clip(tes_A@w_A, 0, None)
bo_Ag, bt_Ag, _ = gate_search(bo_A0, bt_A0, 'A_forO')

# ─────────────────────────────────────────────────────────────
# MODEL O: Blend previous best LB submissions
# A_refined3 + J_oracle_only + I_cb_constrained (3-way OOF optimize)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL O: 3-way blend of A+I+J test predictions ===")
oofs_O  = np.stack([bo_Ag, bo_Ig, bo_Jg], axis=1)
tests_O = np.stack([bt_Ag, bt_Ig, bt_Jg], axis=1)
def obj_O(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(oofs_O@w - y_true))
best_O=999; best_wO=None
for _ in range(10):
    w0=np.random.dirichlet(np.ones(3))
    r=minimize(obj_O,w0,method='L-BFGS-B',bounds=[(0,1)]*3,options={'maxiter':1000})
    if r.fun<best_O: best_O=r.fun; best_wO=r.x
best_wO=np.clip(best_wO,0,1); best_wO/=best_wO.sum()
o_oof=oofs_O@best_wO; o_test=tests_O@best_wO
print(f"  O 3-way OOF: {best_O:.5f}  w: A={best_wO[0]:.3f} I={best_wO[1]:.3f} J={best_wO[2]:.3f}")
save_sub(o_test, best_O, 'O_AIJ_blend')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 2 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:22s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 2 Total: {len(saved)} new files")
