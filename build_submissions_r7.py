"""
Round 7: v31-free variants (safe bets) + further exploration.
Key insight: oracle_v31 was flagged as LB-hurting in memory.
Best OOF so far: Y7=8.37731 (all8 incl v31), Z6=8.37747 (incl v31)
Best v31-FREE OOF: Y6/Z3/Z7 = 8.37783 (H fine gate)

R7 focus on v31-free variants to find best safe submission:
  AA1: Y7-style all 7 (no v31) with 0.01 floor
  AA2: H + oracle_cb (7 components, no v31)
  AA3: mega33>=0.55, oracle_xgb+oracle_lv2 only (ultra-conservative)
  AA4: Z6 concept but no v31 — mega33+34+oracle_lv2+oracle_cb
  AA5: cascade_refined3 + oracle_cb added (baseline + extra oracle)
  AA6: 5-way no-v31 blend (H+A+Y6+Z7+AA1 gated predictions)
  AA7: mega33+rank_adj+oracle_xgb+oracle_lv2+oracle_rem+oracle_cb (6 comp, no mega34, no v31)
  AA8: H with very fine gate search on all parameters
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

# SAFE components (no oracle_v31, no neural, no mega37)
co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'mega34':     d34['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/oof_seqC_cb.npy'),
    # oracle_v31 explicitly excluded from safe-set runs
    'oracle_v31': np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),  # only for AA1 test
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
np.random.seed(303)

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

# ─────────────────────────────────────────────────────────────
# MODEL AA1: Y7-style all 7 NO-v31 (floor=0.01)
# 7 safe components: mega33, mega34, rank_adj, oracle_xgb, oracle_lv2, oracle_rem, oracle_cb
# ─────────────────────────────────────────────────────────────
print("=== MODEL AA1: 7-comp NO-v31 with 0.01 floor ===")
keys_AA1 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_AA1, bt_AA1, bm_AA1, wAA1 = scipy_blend(keys_AA1, n_trials=30,
    bounds=[(0.01,1)]*7)
print(f"  blend OOF: {bm_AA1:.5f}")
bo_AA1g, bt_AA1g, bm_AA1g = gate_search_fine(bo_AA1, bt_AA1, 'AA1_fine')
save_sub(bt_AA1g, bm_AA1g, 'AA1_7comp_nov31')

# ─────────────────────────────────────────────────────────────
# MODEL AA2: H + oracle_cb (7 components including cb, no v31)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA2: H + oracle_cb (mega33+34+rank+xgb+lv2+rem+cb) ===")
keys_AA2 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_AA2, bt_AA2, bm_AA2, _ = scipy_blend(keys_AA2, n_trials=25)
print(f"  blend OOF: {bm_AA2:.5f}")
bo_AA2g, bt_AA2g, bm_AA2g = gate_search(bo_AA2, bt_AA2, 'AA2_gate')
save_sub(bt_AA2g, bm_AA2g, 'AA2_H_plus_cb')

# ─────────────────────────────────────────────────────────────
# MODEL AA3: Ultra-conservative (mega33>=0.60, oracle_xgb+oracle_lv2)
# Maximally simple, safest generalization
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA3: Ultra-conservative (mega33>=0.60, xgb+lv2) ===")
keys_AA3 = ['mega33','oracle_xgb','oracle_lv2']
bo_AA3, bt_AA3, bm_AA3, _ = scipy_blend(keys_AA3, n_trials=20,
    bounds=[(0.55,0.85),(0,0.25),(0,0.25)])
print(f"  blend OOF: {bm_AA3:.5f}")
bo_AA3g, bt_AA3g, bm_AA3g = gate_search(bo_AA3, bt_AA3, 'AA3_gate')
save_sub(bt_AA3g, bm_AA3g, 'AA3_ultraconserv')

# ─────────────────────────────────────────────────────────────
# MODEL AA4: mega33+34 + oracle_lv2 + oracle_cb (4 comp, no v31/xgb/rem)
# lv2 and cb as the oracle pair
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA4: mega33+34 + oracle_lv2 + oracle_cb only ===")
keys_AA4 = ['mega33','mega34','rank_adj','oracle_lv2','oracle_cb']
bo_AA4, bt_AA4, bm_AA4, _ = scipy_blend(keys_AA4, n_trials=20)
print(f"  blend OOF: {bm_AA4:.5f}")
bo_AA4g, bt_AA4g, bm_AA4g = gate_search(bo_AA4, bt_AA4, 'AA4_gate')
save_sub(bt_AA4g, bm_AA4g, 'AA4_2mega_lv2_cb')

# ─────────────────────────────────────────────────────────────
# MODEL AA5: cascade_refined3 + oracle_cb added (6 comp)
# A_refined3 extended with CatBoost oracle
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA5: cascade_refined3 + oracle_cb added ===")
keys_AA5 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_AA5, bt_AA5, bm_AA5, _ = scipy_blend(keys_AA5, n_trials=25)
print(f"  blend OOF: {bm_AA5:.5f}")
bo_AA5g, bt_AA5g, bm_AA5g = gate_search_fine(bo_AA5, bt_AA5, 'AA5_fine')
save_sub(bt_AA5g, bm_AA5g, 'AA5_ref3_plus_cb')

# ─────────────────────────────────────────────────────────────
# MODEL AA6: 5-way no-v31 blend (H+A+AA1+AA2+AA5 gated predictions)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA6: 5-way no-v31 meta blend ===")
# Precompute H and A gated
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_H, bt_H, _, _ = scipy_blend(keys_H, n_trials=20)
bo_Hg, bt_Hg, _ = gate_search_fine(bo_H, bt_H, 'H_for_AA6')

w_A = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_A/=w_A.sum()
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_A = np.stack([co[k] for k in keys_A], axis=1); tes_A = np.stack([ct[k] for k in keys_A], axis=1)
bo_A=np.clip(oofs_A@w_A,0,None); bt_A=np.clip(tes_A@w_A,0,None)
bo_Ag, bt_Ag, _ = gate_search(bo_A, bt_A, 'A_for_AA6')

aa6_oofs  = np.stack([bo_Hg, bo_Ag, bo_AA1g, bo_AA2g, bo_AA5g], axis=1)
aa6_tests = np.stack([bt_Hg, bt_Ag, bt_AA1g, bt_AA2g, bt_AA5g], axis=1)
def obj_AA6(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(aa6_oofs@w - y_true))
best_AA6=999; best_wAA6=None
for _ in range(12):
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_AA6,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':1000})
    if r.fun<best_AA6: best_AA6=r.fun; best_wAA6=r.x
best_wAA6=np.clip(best_wAA6,0,1); best_wAA6/=best_wAA6.sum()
aa6_oof=aa6_oofs@best_wAA6; aa6_te=aa6_tests@best_wAA6
print(f"  AA6 OOF: {best_AA6:.5f}  w: {best_wAA6.round(3)}")
save_sub(aa6_te, best_AA6, 'AA6_5way_nov31')

# ─────────────────────────────────────────────────────────────
# MODEL AA7: Y7 with 0.03 floor (tighter floor = more forced diversity)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA7: all 8 with 0.03 floor (tighter) ===")
keys_AA7 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb','oracle_v31']
bo_AA7, bt_AA7, bm_AA7, _ = scipy_blend(keys_AA7, n_trials=25,
    bounds=[(0.03,1)]*8)
print(f"  blend OOF: {bm_AA7:.5f}")
bo_AA7g, bt_AA7g, bm_AA7g = gate_search(bo_AA7, bt_AA7, 'AA7_gate')
save_sub(bt_AA7g, bm_AA7g, 'AA7_8comp_floor03')

# ─────────────────────────────────────────────────────────────
# MODEL AA8: Fine gate on AA1 (best safe no-v31 from this round)
# AA1 + fine gate
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL AA8: AA1 blend with standard gate (double-check) ===")
# Standard gate on AA1 base (already have AA1g, do wider search)
best_AA8 = bm_AA1g; bo_AA8 = bo_AA1g; bt_AA8 = bt_AA1g
# Try wider gate parameters
for p1 in np.arange(0.04, 0.20, 0.01):
    m1 = (clf_oof>p1).astype(float); m1t = (clf_test>p1).astype(float)
    for w1 in np.arange(0.005, 0.080, 0.005):
        b1 = (1-m1*w1)*bo_AA1 + m1*w1*rh_oof
        b1t = (1-m1t*w1)*bt_AA1 + m1t*w1*rh_te
        for p2 in np.arange(0.10, 0.55, 0.02):
            if p2 <= p1: continue
            m2 = (clf_oof>p2).astype(float); m2t = (clf_test>p2).astype(float)
            for w2 in np.arange(0.005, 0.090, 0.005):
                b2 = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = mae(b2)
                if mm < best_AA8:
                    best_AA8 = mm
                    bo_AA8 = b2; bt_AA8 = (1-m2t*w2)*b1t + m2t*w2*rm_te
print(f"  AA8 wide gate OOF: {best_AA8:.5f}")
save_sub(bt_AA8, best_AA8, 'AA8_AA1_widegate')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 7 SUMMARY (NO oracle_v31 focus)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 7 Total: {len(saved)} new files")
