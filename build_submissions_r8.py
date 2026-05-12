"""
Round 8: Continued exploration — systematic oracle pair testing + gate variants.
Insights from R7 (v31-free focus):
  - AA1 (7comp no-v31): benchmark for v31 comparison
  - AA5 (ref3+cb): cascade_refined3 + oracle_cb
  - AA6 (5-way no-v31 blend): meta

R8 new directions:
  BB1: mega33 + oracle_xgb + oracle_rem (no lv2 — test lv2 necessity)
  BB2: mega33+34 + rank_adj + oracle_lv2 only (oracle_lv2 dominates)
  BB3: random gate threshold search (p1 from 0.03–0.25, wider space)
  BB4: Symmetric blend: mega_avg * 0.55 + oracle_avg * 0.45
  BB5: H blend + triple gate (3 cascade tiers)
  BB6: cascade_refined3 weights (fixed) on mega34 instead of mega33
  BB7: OOF-weighted test avg of top-5 files by OOF
  BB8: AA1 + N blend (safe 7comp + v31 component test)
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
np.random.seed(404)

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
# MODEL BB1: mega33 + oracle_xgb + oracle_rem (no lv2)
# ─────────────────────────────────────────────────────────────
print("=== MODEL BB1: mega33+rank+oracle_xgb+oracle_rem (no lv2) ===")
keys_BB1 = ['mega33','rank_adj','oracle_xgb','oracle_rem']
bo_BB1, bt_BB1, bm_BB1, _ = scipy_blend(keys_BB1, n_trials=20)
print(f"  blend OOF: {bm_BB1:.5f}")
bo_BB1g, bt_BB1g, bm_BB1g = gate_search(bo_BB1, bt_BB1, 'BB1_gate')
save_sub(bt_BB1g, bm_BB1g, 'BB1_no_lv2_v2')

# ─────────────────────────────────────────────────────────────
# MODEL BB2: mega33+34 + rank_adj + oracle_lv2 only
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB2: mega33+34+rank+oracle_lv2 only ===")
keys_BB2 = ['mega33','mega34','rank_adj','oracle_lv2']
bo_BB2, bt_BB2, bm_BB2, _ = scipy_blend(keys_BB2, n_trials=20)
print(f"  blend OOF: {bm_BB2:.5f}")
bo_BB2g, bt_BB2g, bm_BB2g = gate_search(bo_BB2, bt_BB2, 'BB2_gate')
save_sub(bt_BB2g, bm_BB2g, 'BB2_2mega_lv2only')

# ─────────────────────────────────────────────────────────────
# MODEL BB3: Wide gate search (p1 0.03–0.25, wider space)
# Uses H blend base, explores outside normal gate range
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB3: H blend + wide gate (p1=0.03-0.25) ===")
keys_H = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
bo_H, bt_H, _, _ = scipy_blend(keys_H, n_trials=20)
best_BB3 = mae(bo_H); bo_BB3 = bo_H; bt_BB3 = bt_H; cfg_BB3 = 'no_gate'
for p1 in np.arange(0.03, 0.25, 0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005, 0.080, 0.005):
        b1=(1-m1*w1)*bo_H+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_H+m1t*w1*rh_te
        for p2 in np.arange(0.10, 0.60, 0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005, 0.090, 0.005):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_BB3:
                    best_BB3=mm; cfg_BB3=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    bo_BB3=b2; bt_BB3=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [BB3] gate={cfg_BB3}  OOF={best_BB3:.5f}")
save_sub(bt_BB3, best_BB3, 'BB3_widegate')

# ─────────────────────────────────────────────────────────────
# MODEL BB4: Symmetric blend — mega_avg=0.55, oracle_avg=0.45
# mega_avg = (mega33+mega34)/2, oracle_avg = (xgb+lv2+rem)/3
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB4: Symmetric mega(0.55) + oracle(0.45) ===")
mega_oof  = (co['mega33'] + co['mega34']) / 2
mega_test = (ct['mega33'] + ct['mega34']) / 2
oracle_oof  = (co['oracle_xgb'] + co['oracle_lv2'] + co['oracle_rem']) / 3
oracle_test = (ct['oracle_xgb'] + ct['oracle_lv2'] + ct['oracle_rem']) / 3
def obj_BB4(w):
    wm = np.clip(w[0], 0.3, 0.8)
    pred = np.clip(wm*mega_oof + (1-wm)*oracle_oof, 0, None)
    return np.mean(np.abs(pred - y_true))
res_BB4 = minimize(obj_BB4, [0.55], method='L-BFGS-B', bounds=[(0.3,0.8)])
wm_BB4 = np.clip(res_BB4.x[0], 0.3, 0.8)
bo_BB4 = np.clip(wm_BB4*mega_oof + (1-wm_BB4)*oracle_oof, 0, None)
bt_BB4 = np.clip(wm_BB4*mega_test + (1-wm_BB4)*oracle_test, 0, None)
print(f"  optimal w_mega={wm_BB4:.3f}  OOF={mae(bo_BB4):.5f}")
bo_BB4g, bt_BB4g, bm_BB4g = gate_search(bo_BB4, bt_BB4, 'BB4_gate')
save_sub(bt_BB4g, bm_BB4g, 'BB4_symmetric')

# ─────────────────────────────────────────────────────────────
# MODEL BB5: H blend + triple gate (3 cascade tiers)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB5: H blend + triple gate (3 tiers) ===")
best_BB5 = mae(bo_H); bo_BB5 = bo_H; bt_BB5 = bt_H; cfg_BB5 = 'no_gate'
for p1 in np.arange(0.07, 0.15, 0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.015, 0.055, 0.005):
        b1=(1-m1*w1)*bo_H+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_H+m1t*w1*rh_te
        for p2 in np.arange(0.18, 0.38, 0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.015, 0.065, 0.005):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof; b2t=(1-m2t*w2)*b1t+m2t*w2*rm_te
                for p3 in np.arange(0.45, 0.70, 0.05):
                    if p3<=p2: continue
                    m3=(clf_oof>p3).astype(float); m3t=(clf_test>p3).astype(float)
                    for w3 in np.arange(0.015, 0.060, 0.010):
                        # 3rd tier uses huber again at very high confidence
                        b3=(1-m3*w3)*b2+m3*w3*rh_oof
                        mm=mae(b3)
                        if mm<best_BB5:
                            best_BB5=mm; cfg_BB5=f'p{p1:.2f}+p{p2:.2f}+p{p3:.2f}'
                            bo_BB5=b3; bt_BB5=(1-m3t*w3)*b2t+m3t*w3*rh_te
print(f"  [BB5] gate={cfg_BB5}  OOF={best_BB5:.5f}")
save_sub(bt_BB5, best_BB5, 'BB5_triple_gate')

# ─────────────────────────────────────────────────────────────
# MODEL BB6: cascade_refined3 weights on mega34 (swap mega33→mega34)
# What if mega34 is actually better for unseen layouts?
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB6: ref3 weights but mega34 instead of mega33 ===")
w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
keys_ref3_34 = ['mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_ref3_34 = np.stack([co[k] for k in keys_ref3_34], axis=1)
tes_ref3_34  = np.stack([ct[k] for k in keys_ref3_34], axis=1)
bo_BB6 = np.clip(oofs_ref3_34@w_ref3, 0, None)
bt_BB6 = np.clip(tes_ref3_34@w_ref3, 0, None)
bm_BB6_base = mae(bo_BB6)
print(f"  mega34-ref3 fixed OOF: {bm_BB6_base:.5f}")
# Also try scipy optimize same keys
bo_BB6s, bt_BB6s, bm_BB6s, _ = scipy_blend(keys_ref3_34, n_trials=15)
print(f"  mega34-ref3 scipy OOF: {bm_BB6s:.5f}")
if bm_BB6s <= bm_BB6_base:
    bo_BB6, bt_BB6 = bo_BB6s, bt_BB6s
bo_BB6g, bt_BB6g, bm_BB6g = gate_search(bo_BB6, bt_BB6, 'BB6_gate')
save_sub(bt_BB6g, bm_BB6g, 'BB6_mega34_ref3')

# ─────────────────────────────────────────────────────────────
# MODEL BB7: OOF-inverse-weighted avg of top 5 test CSVs
# Weight inversely by OOF (better OOF = more weight)
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB7: Inverse-OOF weighted avg of top 5 files ===")
top5_files = [
    ('submission_Y7_all8_floor_OOF8.37731.csv', 8.37731),
    ('submission_Z6_v31_lv2_only_OOF8.37747.csv', 8.37747),
    ('submission_Z1_Hfine_NA_OOF8.37758.csv', 8.37758),
    ('submission_Z8_HNA_raw_gate_OOF8.37759.csv', 8.37759),
    ('submission_X1_HNA_blend_OOF8.37765.csv', 8.37765),
]
try:
    preds = []
    oofs_vals = []
    for fname, oofval in top5_files:
        df = pd.read_csv(fname).set_index('ID')
        preds.append(np.array([df.loc[i,'avg_delay_minutes_next_30m'] for i in test_raw['ID'].values]))
        oofs_vals.append(oofval)
    # Inverse-OOF weights
    inv_w = np.array([1.0/(o - 8.37) for o in oofs_vals])
    inv_w /= inv_w.sum()
    bb7_test = sum(w*p for w,p in zip(inv_w, preds))
    # OOF estimate (weighted avg of OOF values)
    bb7_oof_est = sum(w*o for w,o in zip(inv_w, oofs_vals))
    print(f"  BB7 inv-weighted OOF estimate: {bb7_oof_est:.5f}")
    print(f"  weights: {inv_w.round(3)}")
    save_sub(bb7_test, bb7_oof_est, 'BB7_invoof_avg5')
except Exception as e:
    print(f"  BB7 skipped: {e}")

# ─────────────────────────────────────────────────────────────
# MODEL BB8: AA1-style (7comp no-v31) + oracle_v31 added at minimum weight
# Compare: what weight does scipy assign v31 if we let it in?
# ─────────────────────────────────────────────────────────────
print("\n=== MODEL BB8: 7comp + v31 with v31 capped at 0.10 ===")
keys_BB8 = ['mega33','mega34','rank_adj','oracle_xgb','oracle_lv2','oracle_rem','oracle_cb','oracle_v31']
bo_BB8, bt_BB8, bm_BB8, wBB8 = scipy_blend(keys_BB8, n_trials=25,
    bounds=[(0.01,1),(0.01,1),(0.01,1),(0.01,1),(0.01,1),(0.01,1),(0.01,1),(0,0.10)])
print(f"  BB8 v31-capped OOF: {bm_BB8:.5f}  v31_w={wBB8[-1]:.4f}")
bo_BB8g, bt_BB8g, bm_BB8g = gate_search(bo_BB8, bt_BB8, 'BB8_gate')
save_sub(bt_BB8g, bm_BB8g, 'BB8_v31_capped')

# ─────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("ROUND 8 SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nRound 8 Total: {len(saved)} new files")
