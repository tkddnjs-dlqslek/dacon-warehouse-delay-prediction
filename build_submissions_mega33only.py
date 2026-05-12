"""
Mega33-ONLY series: LB confirmed safe components only.
Confirmed bad for LB: mega37, mega34, oracle_combined/v31, neural, meta-residual.
Confirmed safe: mega33, oracle_xgb, oracle_lv2, oracle_rem, rank_adj, gate.
oracle_cb: untested on LB — worth experimenting.

This script generates ONLY mega33-based submissions (no mega34 anywhere).
Models:
  S1: cascade_refined3 exact recreation (baseline)
  S2: ref3 + oracle_cb added (6 comp)
  S3: oracle_cb replaces oracle_xgb (ref3 variant)
  S4: oracle_cb replaces oracle_rem (ref3 variant)
  S5: scipy optimize ref3 weights (free optimization around same components)
  S6: mega33 + oracle_lv2 + oracle_rem only (minimal 3-comp)
  S7: mega33 + rank_adj + oracle_xgb + oracle_cb (4-comp, no lv2/rem)
  S8: mega33 + rank_adj + oracle_lv2 + oracle_cb (4-comp, no xgb/rem)
  S9: 4-oracle blend mega33 + xgb + lv2 + rem + cb (all oracle types)
  S10: S2 with fine gate search
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

# SAFE components only — NO mega34, NO mega37, NO v31
co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/oof_seqC_cb.npy'),
}
ct = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_cb':  np.load('results/oracle_seq/test_C_cb.npy'),
}

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

saved = []
np.random.seed(999)

def mae(pred): return np.mean(np.abs(pred - y_true))

def gate_search(base_oof, base_test, label):
    best = mae(base_oof); best_oof = base_oof; best_te = base_test; best_cfg = 'no_gate'
    for p1 in np.arange(0.06, 0.18, 0.01):
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.010, 0.070, 0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in np.arange(0.15, 0.45, 0.02):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.010, 0.080, 0.005):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; best_cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        best_oof=b2; best_te=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={best_cfg}  OOF={best:.5f}")
    return best_oof, best_te, best

def gate_search_fine(base_oof, base_test, label):
    best = mae(base_oof); best_oof = base_oof; best_te = base_test; best_cfg = 'no_gate'
    for p1 in np.arange(0.08, 0.16, 0.005):
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.015, 0.060, 0.003):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in np.arange(0.20, 0.38, 0.01):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.020, 0.065, 0.003):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; best_cfg=f'p{p1:.3f}w{w1:.3f}+p{p2:.3f}w{w2:.3f}'
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

def scipy_blend(keys, n_trials=20, bounds=None):
    oofs_ = np.stack([co[k] for k in keys], axis=1)
    tes_  = np.stack([ct[k] for k in keys], axis=1)
    if bounds is None: bounds = [(0,1)]*len(keys)
    def obj(w):
        w=np.clip(w,0,1); w/=w.sum()
        return np.mean(np.abs(np.clip(oofs_@w,0,None)-y_true))
    best=999; best_w=None
    for _ in range(n_trials):
        w0=np.random.dirichlet(np.ones(len(keys)))
        r=minimize(obj,w0,method='L-BFGS-B',bounds=bounds,options={'maxiter':3000,'ftol':1e-10})
        if r.fun<best: best=r.fun; best_w=r.x
    best_w=np.clip(best_w,0,1); best_w/=best_w.sum()
    for k,w in zip(keys,best_w):
        if w>0.005: print(f"    {k}: {w:.4f}")
    return np.clip(oofs_@best_w,0,None), np.clip(tes_@best_w,0,None), best, best_w

# ─────────────────────────────────────────────────────────────
# S1: cascade_refined3 exact (baseline verify)
# ─────────────────────────────────────────────────────────────
print("=== S1: cascade_refined3 exact ===")
keys_A = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
w_A = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_A/=w_A.sum()
oofs_A = np.stack([co[k] for k in keys_A],axis=1); tes_A = np.stack([ct[k] for k in keys_A],axis=1)
bo_S1=np.clip(oofs_A@w_A,0,None); bt_S1=np.clip(tes_A@w_A,0,None)
bo_S1g, bt_S1g, bm_S1 = gate_search(bo_S1, bt_S1, 'S1')
save_sub(bt_S1g, bm_S1, 'S1_ref3_verify')

# ─────────────────────────────────────────────────────────────
# S2: ref3 + oracle_cb added (6 components)
# ─────────────────────────────────────────────────────────────
print("\n=== S2: ref3 + oracle_cb added ===")
keys_S2 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_S2, bt_S2, bm_S2_base, _ = scipy_blend(keys_S2)
print(f"  blend OOF: {bm_S2_base:.5f}")
bo_S2g, bt_S2g, bm_S2g = gate_search(bo_S2, bt_S2, 'S2')
save_sub(bt_S2g, bm_S2g, 'S2_ref3_plus_cb')

# ─────────────────────────────────────────────────────────────
# S3: oracle_cb replaces oracle_xgb
# ─────────────────────────────────────────────────────────────
print("\n=== S3: oracle_cb replaces oracle_xgb ===")
keys_S3 = ['mega33','rank_adj','oracle_cb','oracle_lv2','oracle_rem']
bo_S3, bt_S3, bm_S3_base, _ = scipy_blend(keys_S3)
print(f"  blend OOF: {bm_S3_base:.5f}")
bo_S3g, bt_S3g, bm_S3g = gate_search(bo_S3, bt_S3, 'S3')
save_sub(bt_S3g, bm_S3g, 'S3_cb_replaces_xgb')

# ─────────────────────────────────────────────────────────────
# S4: oracle_cb replaces oracle_rem
# ─────────────────────────────────────────────────────────────
print("\n=== S4: oracle_cb replaces oracle_rem ===")
keys_S4 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_cb']
bo_S4, bt_S4, bm_S4_base, _ = scipy_blend(keys_S4)
print(f"  blend OOF: {bm_S4_base:.5f}")
bo_S4g, bt_S4g, bm_S4g = gate_search(bo_S4, bt_S4, 'S4')
save_sub(bt_S4g, bm_S4g, 'S4_cb_replaces_rem')

# ─────────────────────────────────────────────────────────────
# S5: scipy free optimize ref3 weights (same 5 components)
# ─────────────────────────────────────────────────────────────
print("\n=== S5: scipy free optimize ref3 weights ===")
keys_S5 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_S5, bt_S5, bm_S5_base, _ = scipy_blend(keys_S5, n_trials=30)
print(f"  optimized OOF: {bm_S5_base:.5f}")
bo_S5g, bt_S5g, bm_S5g = gate_search_fine(bo_S5, bt_S5, 'S5_fine')
save_sub(bt_S5g, bm_S5g, 'S5_ref3_scipy')

# ─────────────────────────────────────────────────────────────
# S6: mega33 + oracle_lv2 + oracle_rem (minimal 3-comp + rank_adj)
# ─────────────────────────────────────────────────────────────
print("\n=== S6: mega33 + rank_adj + oracle_lv2 + oracle_rem ===")
keys_S6 = ['mega33','rank_adj','oracle_lv2','oracle_rem']
bo_S6, bt_S6, bm_S6_base, _ = scipy_blend(keys_S6)
print(f"  blend OOF: {bm_S6_base:.5f}")
bo_S6g, bt_S6g, bm_S6g = gate_search(bo_S6, bt_S6, 'S6')
save_sub(bt_S6g, bm_S6g, 'S6_lv2_rem_only')

# ─────────────────────────────────────────────────────────────
# S7: 4-oracle all-in (mega33 + xgb + lv2 + rem + cb)
# ─────────────────────────────────────────────────────────────
print("\n=== S7: mega33 + rank_adj + all 4 oracles ===")
keys_S7 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_S7, bt_S7, bm_S7_base, _ = scipy_blend(keys_S7, n_trials=30,
    bounds=[(0.35,0.75),(0,0.15),(0,0.20),(0,0.25),(0,0.20),(0,0.20)])
print(f"  blend OOF: {bm_S7_base:.5f}")
bo_S7g, bt_S7g, bm_S7g = gate_search_fine(bo_S7, bt_S7, 'S7_fine')
save_sub(bt_S7g, bm_S7g, 'S7_4oracle')

# ─────────────────────────────────────────────────────────────
# S8: S2 with fine gate search
# ─────────────────────────────────────────────────────────────
print("\n=== S8: S2 blend (ref3+cb) with fine gate ===")
bo_S8g, bt_S8g, bm_S8g = gate_search_fine(bo_S2, bt_S2, 'S8_fine')
save_sub(bt_S8g, bm_S8g, 'S8_ref3cb_finegate')

# ─────────────────────────────────────────────────────────────
# S9: mega33 heavy (>=0.55) + all 4 oracles equal
# ─────────────────────────────────────────────────────────────
print("\n=== S9: mega33>=0.55, rank_adj+4oracle equal ===")
keys_S9 = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb']
bo_S9, bt_S9, bm_S9_base, _ = scipy_blend(keys_S9, n_trials=25,
    bounds=[(0.55,0.80),(0,0.12),(0,0.15),(0,0.18),(0,0.15),(0,0.15)])
print(f"  blend OOF: {bm_S9_base:.5f}")
bo_S9g, bt_S9g, bm_S9g = gate_search(bo_S9, bt_S9, 'S9')
save_sub(bt_S9g, bm_S9g, 'S9_mega33heavy_4o')

# ─────────────────────────────────────────────────────────────
# S10: inverse-OOF weighted avg of S1~S9 gated predictions
# ─────────────────────────────────────────────────────────────
print("\n=== S10: OOF-weighted avg of S-series ===")
s_preds_oof  = [bo_S1g, bo_S2g, bo_S3g, bo_S4g, bo_S5g, bo_S7g, bo_S8g, bo_S9g]
s_preds_test = [bt_S1g, bt_S2g, bt_S3g, bt_S4g, bt_S5g, bt_S7g, bt_S8g, bt_S9g]
def obj_S10(w):
    w=np.clip(w,0,1); w/=w.sum()
    oof_blend = sum(w[i]*s_preds_oof[i] for i in range(len(w)))
    return np.mean(np.abs(oof_blend - y_true))
best_S10=999; best_wS10=None
for _ in range(12):
    w0=np.random.dirichlet(np.ones(8))
    r=minimize(obj_S10,w0,method='L-BFGS-B',bounds=[(0,1)]*8,options={'maxiter':500})
    if r.fun<best_S10: best_S10=r.fun; best_wS10=r.x
best_wS10=np.clip(best_wS10,0,1); best_wS10/=best_wS10.sum()
s10_oof  = sum(best_wS10[i]*s_preds_oof[i]  for i in range(8))
s10_test = sum(best_wS10[i]*s_preds_test[i] for i in range(8))
print(f"  S10 OOF: {best_S10:.5f}  w: {best_wS10.round(3)}")
save_sub(s10_test, best_S10, 'S10_S_series_blend')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("MEGA33-ONLY SERIES SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
