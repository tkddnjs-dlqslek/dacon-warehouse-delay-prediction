"""
Round 3: Generalization-focused variants.
Key insight: LB gap = 50 unseen layouts. We want max generalization.
Strategies:
  P: mega33-ONLY (ultra-simple, no oracle) + gate — pure mega33 baseline
  Q: Remove rank_adj entirely (test if it hurts generalization)
  R: Hard-clip blend (max component = 0.60) — diversity-forced scipy
  S: Quantile-averaged blend (robust to outlier layouts)
  T: mega33 + oracle_xgb only (2-component, ultra-minimal)
  U: cascade_refined3 with NO gate at all (check if gate helps LB)
  V: Triple oracle ensemble (oracle_xgb + oracle_lv2 + oracle_rem equally)
  W: mega33 + rank_adj heavy (rank_adj >= 0.20, test rank-signal strength)
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
np.random.seed(456)

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
# MODEL P: mega33-ONLY (ultra-simple, no oracle components)
# Pure mega33 base, test if it has any standalone LB value
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL P: mega33-only (pure mega33 + gate) ===")
bo_P = co['mega33'].copy(); bt_P = ct['mega33'].copy()
p_mae = mae(bo_P)
print(f"  mega33 standalone OOF: {p_mae:.5f}")
bo_Pg, bt_Pg, bm_Pg = gate_search(bo_P, bt_P, 'P_gate')
save_sub(bt_Pg, bm_Pg, 'P_mega33only')

# ─────────────────────────────────────────────────────────────
# MODEL Q: No rank_adj — test if rank_adj helps or hurts LB
# mega33 + oracle_xgb + oracle_lv2 + oracle_rem (cascade_refined3 minus rank_adj)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL Q: No rank_adj ===")
keys_Q = ['mega33','oracle_xgb','oracle_lv2','oracle_rem']
bo_Q, bt_Q, bm_Q, wQ = scipy_blend(keys_Q, n_trials=20)
print(f"  blend OOF: {bm_Q:.5f}")
bo_Qg, bt_Qg, bm_Qg = gate_search(bo_Q, bt_Q, 'Q_gate')
save_sub(bt_Qg, bm_Qg, 'Q_no_rankadj')

# ─────────────────────────────────────────────────────────────
# MODEL R: Hard-cap blend (max any component = 0.55)
# Forced diversity — no single component dominates
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL R: Hard-cap blend (max=0.55) ===")
keys_R = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_R, bt_R, bm_R, wR = scipy_blend(keys_R, n_trials=25,
    bounds=[(0,0.55),(0,0.20),(0,0.40),(0,0.40),(0,0.40)])
print(f"  blend OOF: {bm_R:.5f}")
bo_Rg, bt_Rg, bm_Rg = gate_search(bo_R, bt_R, 'R_gate')
save_sub(bt_Rg, bm_Rg, 'R_hardcap_blend')

# ─────────────────────────────────────────────────────────────
# MODEL S: Geometric mean of top-2 components (mega33 + best oracle)
# Geometric mean is more robust to layout-distribution shift
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL S: Geometric mean (mega33 × oracle_xgb) ===")
eps = 0.1
# Geometric mean = exp(avg of logs) — robust to extreme values
geo_oof = np.sqrt(np.maximum(eps, co['mega33']) * np.maximum(eps, co['oracle_xgb']))
geo_te  = np.sqrt(np.maximum(eps, ct['mega33']) * np.maximum(eps, ct['oracle_xgb']))
bm_S = mae(geo_oof)
print(f"  geometric mean OOF: {bm_S:.5f}")
bo_Sg, bt_Sg, bm_Sg = gate_search(geo_oof, geo_te, 'S_gate')
save_sub(bt_Sg, bm_Sg, 'S_geometric_mean')

# ─────────────────────────────────────────────────────────────
# MODEL T: 2-component ultra-minimal (mega33 + oracle_xgb only)
# Absolute minimum complexity — best generalization candidate
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL T: 2-component minimal (mega33 + oracle_xgb) ===")
keys_T = ['mega33','oracle_xgb']
bo_T, bt_T, bm_T, wT = scipy_blend(keys_T, n_trials=20)
print(f"  2-comp blend OOF: {bm_T:.5f}")
bo_Tg, bt_Tg, bm_Tg = gate_search(bo_T, bt_T, 'T_gate')
save_sub(bt_Tg, bm_Tg, 'T_2comp_minimal')

# ─────────────────────────────────────────────────────────────
# MODEL U: cascade_refined3 with NO gate (pure blend only)
# Test if the cascade gate helps or hurts LB
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL U: cascade_refined3 NO gate ===")
keys_U = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
w_U = np.array([0.4997, 0.1019, 0.1338, 0.1741, 0.0991])
w_U /= w_U.sum()
oofs_U = np.stack([co[k] for k in keys_U], axis=1)
tes_U  = np.stack([ct[k] for k in keys_U], axis=1)
bo_U = np.clip(oofs_U@w_U, 0, None); bt_U = np.clip(tes_U@w_U, 0, None)
bm_U = mae(bo_U)
print(f"  cascade_refined3 NO gate OOF: {bm_U:.5f}")
save_sub(bt_U, bm_U, 'U_nogate_ref3')

# ─────────────────────────────────────────────────────────────
# MODEL V: Triple oracle equal weight (oracle_xgb=oracle_lv2=oracle_rem=1/3)
# Pure oracle signal, no mega/rank influence
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL V: Triple oracle equal weight ===")
bo_V = (co['oracle_xgb'] + co['oracle_lv2'] + co['oracle_rem']) / 3
bt_V = (ct['oracle_xgb'] + ct['oracle_lv2'] + ct['oracle_rem']) / 3
bm_V = mae(bo_V)
print(f"  triple oracle OOF: {bm_V:.5f}")
bo_Vg, bt_Vg, bm_Vg = gate_search(bo_V, bt_V, 'V_gate')
save_sub(bt_Vg, bm_Vg, 'V_triple_oracle')

# ─────────────────────────────────────────────────────────────
# MODEL W: mega33 + rank_adj heavy (rank_adj >= 0.20)
# Hypothesis: rank signal is more robust to unseen layouts
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL W: rank_adj heavy (>=0.20) ===")
keys_W = ['mega33','rank_adj','oracle_xgb','oracle_lv2']
bo_W, bt_W, bm_W, wW = scipy_blend(keys_W, n_trials=20,
    bounds=[(0,0.70),(0.15,0.35),(0,0.30),(0,0.30)])
print(f"  rank-heavy blend OOF: {bm_W:.5f}")
bo_Wg, bt_Wg, bm_Wg = gate_search(bo_W, bt_W, 'W_gate')
save_sub(bt_Wg, bm_Wg, 'W_rankadj_heavy')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 3 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 3 Total: {len(saved)} new files")
