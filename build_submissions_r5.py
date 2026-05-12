"""
Round 5: Surgical experiments based on R1-R4 findings.
Key findings so far:
  - Best OOF candidates: H/M (8.37789), N (8.37818), O/X1 blends (~8.378xx)
  - Gate is essential (U no-gate = 8.38205, worse)
  - rank_adj marginally helps (Q no-rankadj = 8.37976)
  - mega34 adds marginal benefit over mega33 alone
  - oracle components: lv2 > xgb > rem by individual MAE
  - Standalone oracle (J/V) = terrible, standalone mega33 (P) = terrible

New angles for R5:
  Y1: Systematic gate param search on N (best blend after H) — fine-grained
  Y2: mega33 + oracle_lv2 only (strongest oracle paired with mega33)
  Y3: H blend but rank_adj dropped (test rank_adj in mega33+34 context)
  Y4: cascade_refined3 with oracle_cb instead of oracle_rem
  Y5: mega33 + oracle_xgb + oracle_lv2 (drop oracle_rem, keep lv2)
  Y6: Very tight gate (p1=0.08-0.12, very conservative)
  Y7: scipy on all 8 components with hard floor (each >= 0.01)
  Y8: Blend of H and A (two very close but different gated predictions)
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
np.random.seed(101)

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

def gate_search_fine(base_oof, base_test, label):
    """Finer-grained gate search around the known-good p0.11/p0.26 range."""
    best = mae(base_oof); best_oof = base_oof; best_te = base_test; best_cfg = 'no_gate'
    for p1 in np.arange(0.08, 0.16, 0.005):
        m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
        for w1 in np.arange(0.015, 0.060, 0.003):
            b1 = (1-m1*w1)*base_oof + m1*w1*rh_oof
            b1t = (1-m1t*w1)*base_test + m1t*w1*rh_te
            for p2 in np.arange(0.20, 0.38, 0.01):
                if p2 <= p1: continue
                m2 = (clf_oof>p2).astype(float); m2t = (clf_test>p2).astype(float)
                for w2 in np.arange(0.020, 0.065, 0.003):
                    b2 = (1-m2*w2)*b1 + m2*w2*rm_oof
                    mm = mae(b2)
                    if mm < best:
                        best = mm; best_cfg = f'p{p1:.3f}w{w1:.3f}+p{p2:.3f}w{w2:.3f}'
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
# MODEL Y1: Fine-grained gate search on N-blend (oracle_v31 included)
# N gave best OOF=8.37818 — fine gate might squeeze more
# ─────────────────────────────────────────────────────────────
print("=== MODEL Y1: Fine gate search on N (oracle_v31) ===")
keys_N = ['mega33','rank_adj','oracle_xgb','oracle_v31','oracle_lv2']
bo_N, bt_N, _, _ = scipy_blend(keys_N, n_trials=25)
print(f"  N blend OOF: {_:.5f}" if False else f"  N blend computed")
bo_N_fine, bt_N_fine, bm_Y1 = gate_search_fine(bo_N, bt_N, 'Y1_fine')
save_sub(bt_N_fine, bm_Y1, 'Y1_N_finegate')

# ─────────────────────────────────────────────────────────────
# MODEL Y2: mega33 + oracle_lv2 only (strongest oracle + mega33)
# lv2 is the strongest individual oracle — what if it's all we need?
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y2: mega33 + oracle_lv2 only ===")
keys_Y2 = ['mega33','oracle_lv2']
bo_Y2, bt_Y2, bm_Y2, _ = scipy_blend(keys_Y2, n_trials=15)
print(f"  2-comp OOF: {bm_Y2:.5f}")
bo_Y2g, bt_Y2g, bm_Y2g = gate_search(bo_Y2, bt_Y2, 'Y2_gate')
save_sub(bt_Y2g, bm_Y2g, 'Y2_mega33_lv2')

# ─────────────────────────────────────────────────────────────
# MODEL Y3: H (mega33+34 blend) without rank_adj
# Does rank_adj help in the context of 2 megas?
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y3: mega33+34 without rank_adj ===")
keys_Y3 = ['mega33','mega34','oracle_xgb','oracle_lv2','oracle_rem']
bo_Y3, bt_Y3, bm_Y3, _ = scipy_blend(keys_Y3, n_trials=20)
print(f"  blend OOF: {bm_Y3:.5f}")
bo_Y3g, bt_Y3g, bm_Y3g = gate_search(bo_Y3, bt_Y3, 'Y3_gate')
save_sub(bt_Y3g, bm_Y3g, 'Y3_2mega_norank')

# ─────────────────────────────────────────────────────────────
# MODEL Y4: cascade_refined3 with oracle_cb instead of oracle_rem
# oracle_cb is similar to oracle_xgb in behavior — test swap
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y4: ref3 with oracle_cb instead of oracle_rem ===")
keys_Y4 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_cb']
bo_Y4, bt_Y4, bm_Y4, _ = scipy_blend(keys_Y4, n_trials=20)
print(f"  blend OOF: {bm_Y4:.5f}")
bo_Y4g, bt_Y4g, bm_Y4g = gate_search(bo_Y4, bt_Y4, 'Y4_gate')
save_sub(bt_Y4g, bm_Y4g, 'Y4_cb_swap')

# ─────────────────────────────────────────────────────────────
# MODEL Y5: mega33 + oracle_xgb + oracle_lv2 (drop oracle_rem)
# oracle_rem might be adding overfit noise
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y5: mega33 + oracle_xgb + oracle_lv2 (no rem) ===")
keys_Y5 = ['mega33','rank_adj','oracle_xgb','oracle_lv2']
bo_Y5, bt_Y5, bm_Y5, _ = scipy_blend(keys_Y5, n_trials=20)
print(f"  blend OOF: {bm_Y5:.5f}")
bo_Y5g, bt_Y5g, bm_Y5g = gate_search(bo_Y5, bt_Y5, 'Y5_gate')
save_sub(bt_Y5g, bm_Y5g, 'Y5_no_rem')

# ─────────────────────────────────────────────────────────────
# MODEL Y6: H blend, fine gate search (best component combo, refined gate)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y6: H blend + fine gate search ===")
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_H, bt_H, _, _ = scipy_blend(keys_H, n_trials=25)
print(f"  H blend computed")
bo_Y6, bt_Y6, bm_Y6 = gate_search_fine(bo_H, bt_H, 'Y6_H_fine')
save_sub(bt_Y6, bm_Y6, 'Y6_H_finegate')

# ─────────────────────────────────────────────────────────────
# MODEL Y7: All 8 components, each with floor=0.01 (force diversity)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y7: All 8 components, floor=0.01 ===")
keys_Y7 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb','oracle_v31']
bo_Y7, bt_Y7, bm_Y7, _ = scipy_blend(keys_Y7, n_trials=25,
    bounds=[(0.01,1)]*8)
print(f"  8-comp OOF: {bm_Y7:.5f}")
bo_Y7g, bt_Y7g, bm_Y7g = gate_search(bo_Y7, bt_Y7, 'Y7_gate')
save_sub(bt_Y7g, bm_Y7g, 'Y7_all8_floor')

# ─────────────────────────────────────────────────────────────
# MODEL Y8: Blend of H_gated + A_gated test predictions
# Two nearby predictions with different base composition
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Y8: H_gated + A_gated prediction blend ===")
# Compute A gated
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
w_A = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_A/=w_A.sum()
oofs_A = np.stack([co[k] for k in keys_A], axis=1)
tes_A  = np.stack([ct[k] for k in keys_A], axis=1)
bo_A0=np.clip(oofs_A@w_A,0,None); bt_A0=np.clip(tes_A@w_A,0,None)
bo_Ag, bt_Ag, _ = gate_search(bo_A0, bt_A0, 'A_for_Y8')
bo_Hg, bt_Hg, _ = gate_search(bo_H, bt_H, 'H_for_Y8')

# Blend H + A gated
ha_oofs  = np.stack([bo_Hg, bo_Ag], axis=1)
ha_tests = np.stack([bt_Hg, bt_Ag], axis=1)
def obj_Y8(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(ha_oofs@w - y_true))
best_Y8=999; best_wY8=None
for _ in range(10):
    w0=np.random.dirichlet(np.ones(2))
    r=minimize(obj_Y8,w0,method='L-BFGS-B',bounds=[(0,1)]*2,options={'maxiter':500})
    if r.fun<best_Y8: best_Y8=r.fun; best_wY8=r.x
best_wY8=np.clip(best_wY8,0,1); best_wY8/=best_wY8.sum()
y8_oof=ha_oofs@best_wY8; y8_te=ha_tests@best_wY8
print(f"  Y8 OOF: {best_Y8:.5f}  w: H={best_wY8[0]:.3f} A={best_wY8[1]:.3f}")
save_sub(y8_te, best_Y8, 'Y8_HA_blend')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 5 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 5 Total: {len(saved)} new files")
