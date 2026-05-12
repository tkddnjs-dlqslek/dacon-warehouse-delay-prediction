"""
Round 6: Meta-level combinations and final exploration.
Findings from R1-R5:
  - Best OOF: X1/X6 = 8.37765 (H+N+A gated blend)
  - Fine gate on H gave Y6 — check if better
  - Y3 (mega33+34 no rank_adj) interesting to watch
  - W_rankadj_heavy was 8.37966 — more rank_adj hurts
  - oracle_v31 (N) adds marginal benefit over A

Round 6 focus:
  Z1: 3-way blend with fine gate on each component first (H_fine + N + A)
  Z2: mega33+34+oracle_xgb+oracle_lv2 (drop oracle_rem and rank_adj both)
  Z3: oracle_lv2 standalone + gate (oracle_lv2 MAE alone)
  Z4: mega33+34 scipy, then single-tier gate only (huber only, no mae)
  Z5: H blend + single-tier gate on huber specialist only
  Z6: Trimmed mean of top-5 OOF predictions (robust aggregation)
  Z7: meta33 with oracle weight 0.25 hard fixed, optimize other weights
  Z8: Blend using per-layout validation weights (layout-stratified scipy)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
groups = train_raw['layout_id'].values

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
np.random.seed(202)

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

# Precompute component blends
print("Precomputing component blends...")
# H = mega33+34 full
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_H, bt_H, _, _ = scipy_blend(keys_H, n_trials=20)
# N = oracle_v31 included
keys_N = ['mega33','rank_adj','oracle_xgb','oracle_v31','oracle_lv2']
bo_N, bt_N, _, _ = scipy_blend(keys_N, n_trials=20)
# A = cascade_refined3
w_A = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_A/=w_A.sum()
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_A = np.stack([co[k] for k in keys_A], axis=1); tes_A = np.stack([ct[k] for k in keys_A], axis=1)
bo_A = np.clip(oofs_A@w_A,0,None); bt_A = np.clip(tes_A@w_A,0,None)

# Gate all
bo_Hg, bt_Hg, _ = gate_search_fine(bo_H, bt_H, 'H_pre')
bo_Ng, bt_Ng, _ = gate_search(bo_N, bt_N, 'N_pre')
bo_Ag, bt_Ag, _ = gate_search(bo_A, bt_A, 'A_pre')
print("Precompute done.\n")

# ─────────────────────────────────────────────────────────────
# MODEL Z1: 3-way blend with fine-gated H (H_fine + N + A)
# ─────────────────────────────────────────────────────────────
print("=== MODEL Z1: H_fine + N + A blend ===")
z1_oofs  = np.stack([bo_Hg, bo_Ng, bo_Ag], axis=1)
z1_tests = np.stack([bt_Hg, bt_Ng, bt_Ag], axis=1)
def obj_Z1(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(z1_oofs@w - y_true))
best_Z1=999; best_wZ1=None
for _ in range(10):
    w0=np.random.dirichlet(np.ones(3))
    r=minimize(obj_Z1,w0,method='L-BFGS-B',bounds=[(0,1)]*3,options={'maxiter':1000})
    if r.fun<best_Z1: best_Z1=r.fun; best_wZ1=r.x
best_wZ1=np.clip(best_wZ1,0,1); best_wZ1/=best_wZ1.sum()
print(f"  Z1 OOF: {best_Z1:.5f}  w: H={best_wZ1[0]:.3f} N={best_wZ1[1]:.3f} A={best_wZ1[2]:.3f}")
save_sub(z1_tests@best_wZ1, best_Z1, 'Z1_Hfine_NA')

# ─────────────────────────────────────────────────────────────
# MODEL Z2: mega33+34 + oracle_xgb + oracle_lv2 (drop rem+rank_adj)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z2: mega33+34+oracle_xgb+oracle_lv2 (no rem, no rank) ===")
keys_Z2 = ['mega33','mega34','oracle_xgb','oracle_lv2']
bo_Z2, bt_Z2, bm_Z2, _ = scipy_blend(keys_Z2, n_trials=20)
print(f"  blend OOF: {bm_Z2:.5f}")
bo_Z2g, bt_Z2g, bm_Z2g = gate_search(bo_Z2, bt_Z2, 'Z2_gate')
save_sub(bt_Z2g, bm_Z2g, 'Z2_2mega_2oracle')

# ─────────────────────────────────────────────────────────────
# MODEL Z3: Single-tier gate only (huber tier only, no mae tier)
# Tests if 2nd gate tier (mae) helps
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z3: H blend, single-tier gate (huber only) ===")
best_Z3 = mae(bo_Hg); best_Z3_oof = bo_Hg; best_Z3_te = bt_Hg; best_cfg_Z3 = 'H_fine_gate'
# Try single tier only
for p1 in np.arange(0.06, 0.18, 0.01):
    m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
    for w1 in np.arange(0.010, 0.070, 0.005):
        b1 = (1-m1*w1)*bo_H + m1*w1*rh_oof
        b1t = (1-m1t*w1)*bt_H + m1t*w1*rh_te
        mm = mae(b1)
        if mm < best_Z3:
            best_Z3 = mm; best_cfg_Z3 = f'single_p{p1:.2f}w{w1:.3f}'
            best_Z3_oof = b1; best_Z3_te = b1t
print(f"  Z3 single-tier OOF: {best_Z3:.5f}  ({best_cfg_Z3})")
save_sub(best_Z3_te, best_Z3, 'Z3_single_gate')

# ─────────────────────────────────────────────────────────────
# MODEL Z4: Trimmed mean of top-5 by OOF
# H_fine, N, A, X1 blend, Z2 — robust aggregation
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z4: Trimmed average of 5 component predictions ===")
# Need X1 test predictions — load from CSV
try:
    x1_csv = pd.read_csv('submission_X1_HNA_blend_OOF8.37765.csv')
    x1_te = x1_csv['avg_delay_minutes_next_30m'].values
    # Reorder to match test_raw order
    x1_csv = x1_csv.set_index('ID')
    x1_te_ordered = np.array([x1_csv.loc[i,'avg_delay_minutes_next_30m'] for i in test_raw['ID'].values])

    trim5_test = np.stack([bt_Hg, bt_Ng, bt_Ag, bt_Z2g, x1_te_ordered], axis=1)
    # Trimmed mean: drop min and max, average remaining 3
    trim_sorted = np.sort(trim5_test, axis=1)
    z4_test = trim_sorted[:,1:4].mean(axis=1)

    # For OOF, compute equivalent
    trim5_oof  = np.stack([bo_Hg, bo_Ng, bo_Ag, bo_Z2g, z1_oofs@best_wZ1], axis=1)
    trim_oof_sorted = np.sort(trim5_oof, axis=1)
    z4_oof = trim_oof_sorted[:,1:4].mean(axis=1)
    bm_Z4 = mae(z4_oof)
    print(f"  Z4 trimmed mean OOF: {bm_Z4:.5f}")
    save_sub(z4_test, bm_Z4, 'Z4_trimmed_mean')
except Exception as e:
    print(f"  Z4 skipped: {e}")
    # Fallback: equal weight avg of 4
    z4_oof  = (bo_Hg + bo_Ng + bo_Ag + bo_Z2g) / 4
    z4_test = (bt_Hg + bt_Ng + bt_Ag + bt_Z2g) / 4
    bm_Z4 = mae(z4_oof)
    print(f"  Z4 fallback (avg4) OOF: {bm_Z4:.5f}")
    save_sub(z4_test, bm_Z4, 'Z4_avg4_fallback')

# ─────────────────────────────────────────────────────────────
# MODEL Z5: mega33 fixed 0.50 + all oracle scipy
# Force mega33 at exactly 0.50, optimize rest
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z5: mega33=0.50 fixed, oracle remainder ===")
oracle_keys = ['rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oracle_oofs = np.stack([co[k] for k in oracle_keys], axis=1)
oracle_tes  = np.stack([ct[k] for k in oracle_keys], axis=1)
def obj_Z5(w):
    w=np.clip(w,0,1); w/=w.sum(); w*=0.50  # scale to 50% of weight
    pred = np.clip(0.50*co['mega33'] + oracle_oofs@w, 0, None)
    return np.mean(np.abs(pred - y_true))
best_Z5=999; best_wZ5=None
for _ in range(15):
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_Z5,w0,method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':2000})
    if r.fun<best_Z5: best_Z5=r.fun; best_wZ5=r.x
best_wZ5=np.clip(best_wZ5,0,1); best_wZ5/=best_wZ5.sum(); best_wZ5*=0.50
bo_Z5=np.clip(0.50*co['mega33']+oracle_oofs@best_wZ5,0,None)
bt_Z5=np.clip(0.50*ct['mega33']+oracle_tes@best_wZ5,0,None)
print(f"  Z5 OOF: {best_Z5:.5f}  oracle_w: {dict(zip(oracle_keys, best_wZ5.round(4)))}")
bo_Z5g, bt_Z5g, bm_Z5g = gate_search(bo_Z5, bt_Z5, 'Z5_gate')
save_sub(bt_Z5g, bm_Z5g, 'Z5_mega50_fixed')

# ─────────────────────────────────────────────────────────────
# MODEL Z6: mega33+34+oracle_v31+oracle_lv2 (v31+lv2 both, no xgb/rem)
# Try different oracle pair
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z6: mega33+34 + oracle_v31 + oracle_lv2 (no xgb/rem) ===")
keys_Z6 = ['mega33','mega34','rank_adj','oracle_v31','oracle_lv2']
bo_Z6, bt_Z6, bm_Z6, _ = scipy_blend(keys_Z6, n_trials=20)
print(f"  blend OOF: {bm_Z6:.5f}")
bo_Z6g, bt_Z6g, bm_Z6g = gate_search(bo_Z6, bt_Z6, 'Z6_gate')
save_sub(bt_Z6g, bm_Z6g, 'Z6_v31_lv2_only')

# ─────────────────────────────────────────────────────────────
# MODEL Z7: mega33+34+oracle full 7-way but tight bounds
# Everything allowed but tighter bounds for generalization
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z7: 6-way tight bounds (mega33>=0.35, oracle<=0.20 each) ===")
keys_Z7 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_Z7, bt_Z7, bm_Z7, _ = scipy_blend(keys_Z7, n_trials=25,
    bounds=[(0.35,0.70),(0,0.25),(0,0.15),(0,0.20),(0,0.25),(0,0.20)])
print(f"  Z7 blend OOF: {bm_Z7:.5f}")
bo_Z7g, bt_Z7g, bm_Z7g = gate_search_fine(bo_Z7, bt_Z7, 'Z7_fine')
save_sub(bt_Z7g, bm_Z7g, 'Z7_tight_bounds')

# ─────────────────────────────────────────────────────────────
# MODEL Z8: X1 fine refinement — try more starting points
# X1 was optimal 3-way blend — try with scipy on OOF (non-gated) then gate
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Z8: 3-way H+N+A scipy on raw OOF (then gate) ===")
# Rather than blending gated, blend raw then gate
z8_oofs = np.stack([bo_H, bo_N, bo_A], axis=1)
z8_tes  = np.stack([bt_H, bt_N, bt_A], axis=1)
def obj_Z8(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(z8_oofs@w,0,None) - y_true))
best_Z8=999; best_wZ8=None
for _ in range(15):
    w0=np.random.dirichlet(np.ones(3))
    r=minimize(obj_Z8,w0,method='L-BFGS-B',bounds=[(0,1)]*3,options={'maxiter':1000})
    if r.fun<best_Z8: best_Z8=r.fun; best_wZ8=r.x
best_wZ8=np.clip(best_wZ8,0,1); best_wZ8/=best_wZ8.sum()
bo_Z8=np.clip(z8_oofs@best_wZ8,0,None); bt_Z8=np.clip(z8_tes@best_wZ8,0,None)
print(f"  Z8 raw blend OOF: {best_Z8:.5f}  w: H={best_wZ8[0]:.3f} N={best_wZ8[1]:.3f} A={best_wZ8[2]:.3f}")
bo_Z8g, bt_Z8g, bm_Z8g = gate_search_fine(bo_Z8, bt_Z8, 'Z8_fine')
save_sub(bt_Z8g, bm_Z8g, 'Z8_HNA_raw_gate')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 6 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 6 Total: {len(saved)} new files")
