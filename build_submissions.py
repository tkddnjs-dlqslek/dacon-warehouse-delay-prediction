"""
Build diverse submission files.
Anchor: cascade_refined3 (mega33+oracle_xgb+oracle_lv2+oracle_rem+rank_adj+gate) = LB 9.7527

Strategy:
  A: cascade_refined3 exact recreation + variations (gate off, tighter grid)
  B: oracle_cb (CatBoost oracle) as primary oracle
  C: seqB diversity injection
  D: A+B hybrid blend
  E: No cascade gate baseline (pure blend)
  F: oracle_cb + oracle_xgb both (두 oracle 같이)
  G: Scipy re-optimize with fresh random starts
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

# ── Data ──────────────────────────────────────────────────────────────────────
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

# ── Load models ───────────────────────────────────────────────────────────────
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

components_oof = {
    'mega33':     d33['meta_avg_oof'][id2],
    'mega34':     d34['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/oof_seqC_cb.npy'),
    'seqA':       np.load('results/oracle_seq/oof_seqA.npy'),
    'seqB':       np.load('results/oracle_seq/oof_seqB.npy'),
}
components_test = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'mega34':     d34['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/test_C_cb.npy'),
    'seqA':       np.load('results/oracle_seq/test_A.npy'),
    'seqB':       np.load('results/oracle_seq/test_B.npy'),
}

# Cascade resources (CLF v1 only — v2 AUC worse)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

saved = []

def mae(pred): return np.mean(np.abs(pred - y_true))

def scipy_blend(keys, n_starts=15):
    oofs  = np.stack([components_oof[k]  for k in keys], axis=1)
    tests = np.stack([components_test[k] for k in keys], axis=1)
    def obj(w):
        w = np.clip(w,0,1); w /= w.sum()
        return np.mean(np.abs(np.clip(oofs@w,0,None) - y_true))
    best_mae = 999; best_w = None
    for _ in range(n_starts):
        w0 = np.random.dirichlet(np.ones(len(keys)))
        r = minimize(obj, w0, method='L-BFGS-B',
                     bounds=[(0,1)]*len(keys), options={'maxiter':3000,'ftol':1e-10})
        if r.fun < best_mae: best_mae=r.fun; best_w=r.x
    best_w = np.clip(best_w,0,1); best_w /= best_w.sum()
    blend_oof  = np.clip(oofs@best_w,  0, None)
    blend_test = np.clip(tests@best_w, 0, None)
    return blend_oof, blend_test, best_w, best_mae

def gate_search(base_oof, base_test, label):
    best = mae(base_oof); best_oof = base_oof; best_te = base_test; best_cfg = 'no_gate'
    for p1 in np.arange(0.07, 0.18, 0.01):
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.010, 0.065, 0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof
            b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in np.arange(0.18, 0.42, 0.02):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.015, 0.075, 0.005):
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

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A: cascade_refined3 exact recreation
# Keys: mega33, rank_adj, oracle_xgb, oracle_lv2, oracle_rem
# Weights: mega33=0.4997, rank_adj=0.1019, oracle_xgb=0.1338, lv2=0.1741, rem=0.0991
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL A: cascade_refined3 exact (mega33 + oracle_xgb/lv2/rem)")
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_A, bt_A, w_A, bm_A = scipy_blend(keys_A, n_starts=20)
print(f"  A blend OOF: {bm_A:.5f}")
for k,w in zip(keys_A,w_A):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Ag, bt_Ag, bm_Ag = gate_search(bo_A, bt_A, 'A_gate')
save_sub(bt_Ag, bm_Ag, 'A_refined3')

# A-nogate: same blend, no cascade gate (test if gate helps on unseen layouts)
save_sub(bt_A, bm_A, 'A_nogate')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B: oracle_cb (CatBoost oracle) replaces oracle_xgb
# Keys: mega33, rank_adj, oracle_cb, oracle_lv2, oracle_rem
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL B: oracle_cb (CatBoost) as primary oracle")
keys_B = ['mega33','rank_adj','oracle_cb','oracle_lv2','oracle_rem']
bo_B, bt_B, w_B, bm_B = scipy_blend(keys_B, n_starts=20)
print(f"  B blend OOF: {bm_B:.5f}")
for k,w in zip(keys_B,w_B):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Bg, bt_Bg, bm_Bg = gate_search(bo_B, bt_B, 'B_gate')
save_sub(bt_Bg, bm_Bg, 'B_oracle_cb')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C: seqB diversity — add seqB (sequential model, different arch)
# Keys: mega33, rank_adj, oracle_xgb, oracle_lv2, oracle_rem, seqB
# seqB is weaker individually (MAE 9.19) but different architecture
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL C: seqB diversity (different architecture)")
keys_C = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','seqB']
bo_C, bt_C, w_C, bm_C = scipy_blend(keys_C, n_starts=20)
print(f"  C blend OOF: {bm_C:.5f}")
for k,w in zip(keys_C,w_C):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Cg, bt_Cg, bm_Cg = gate_search(bo_C, bt_C, 'C_gate')
save_sub(bt_Cg, bm_Cg, 'C_seqB_div')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL D: A+B test prediction blend (50/50 hybrid)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL D: A+B hybrid (50/50 test blend)")
hybrid_test = 0.5*bt_Ag + 0.5*bt_Bg
hybrid_oof  = 0.5*bo_Ag + 0.5*bo_Bg
hybrid_mae  = mae(hybrid_oof)
print(f"  D hybrid OOF: {hybrid_mae:.5f}")
save_sub(hybrid_test, hybrid_mae, 'D_AB_hybrid')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL E: oracle_xgb + oracle_cb both (두 oracle 앙상블)
# mega33 + rank_adj + oracle_xgb + oracle_cb + oracle_lv2
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL E: oracle_xgb + oracle_cb both")
keys_E = ['mega33','rank_adj','oracle_xgb','oracle_cb','oracle_lv2','oracle_rem']
bo_E, bt_E, w_E, bm_E = scipy_blend(keys_E, n_starts=20)
print(f"  E blend OOF: {bm_E:.5f}")
for k,w in zip(keys_E,w_E):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Eg, bt_Eg, bm_Eg = gate_search(bo_E, bt_E, 'E_gate')
save_sub(bt_Eg, bm_Eg, 'E_xgb_cb_both')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL F: seqA+seqB diversity (both sequential)
# mega33 + rank_adj + oracle_xgb + oracle_lv2 + seqA + seqB
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL F: seqA+seqB (both sequential architectures)")
keys_F = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','seqA','seqB']
bo_F, bt_F, w_F, bm_F = scipy_blend(keys_F, n_starts=20)
print(f"  F blend OOF: {bm_F:.5f}")
for k,w in zip(keys_F,w_F):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Fg, bt_Fg, bm_Fg = gate_search(bo_F, bt_F, 'F_gate')
save_sub(bt_Fg, bm_Fg, 'F_seqAB_div')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL G: A+C+B 3-way blend (A/B/C test predictions blended)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL G: A+B+C 3-way blend")
oofs_g  = np.stack([bo_Ag, bo_Bg, bo_Cg], axis=1)
tests_g = np.stack([bt_Ag, bt_Bg, bt_Cg], axis=1)
def obj_g(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(oofs_g@w - y_true))
best_g = 999; best_wg = None
for _ in range(10):
    w0 = np.random.dirichlet(np.ones(3))
    r = minimize(obj_g, w0, method='L-BFGS-B', bounds=[(0,1)]*3,
                 options={'maxiter':1000})
    if r.fun < best_g: best_g=r.fun; best_wg=r.x
best_wg = np.clip(best_wg,0,1); best_wg/=best_wg.sum()
g_oof  = oofs_g@best_wg; g_test = tests_g@best_wg
g_mae = mae(g_oof)
print(f"  G 3-way blend OOF: {g_mae:.5f}  weights: A={best_wg[0]:.3f} B={best_wg[1]:.3f} C={best_wg[2]:.3f}")
save_sub(g_test, g_mae, 'G_ABC_blend')

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL H: mega33+mega34 mix + oracle_xgb (test mega34 contribution)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("MODEL H: mega33+mega34 mix")
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_H, bt_H, w_H, bm_H = scipy_blend(keys_H, n_starts=20)
print(f"  H blend OOF: {bm_H:.5f}")
for k,w in zip(keys_H,w_H):
    if w>0.005: print(f"    {k}: {w:.4f}")
bo_Hg, bt_Hg, bm_Hg = gate_search(bo_H, bt_H, 'H_gate')
save_sub(bt_Hg, bm_Hg, 'H_mega33_34')

# ═══════════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:20s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} submission files ready.")
