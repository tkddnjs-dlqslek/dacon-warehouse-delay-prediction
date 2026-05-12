"""
DD Round: ref3-only 고급 gate 최적화.
All models: ONLY mega33, rank_adj, oracle_xgb, oracle_lv2, oracle_rem.

DD1: Scipy 연속 gate 최적화 (p1,p2,w1,w2 직접 최적화 — grid보다 정밀)
DD2: 매우 넓은 gate 탐색 (p1 0.01~0.40, p2 0.05~0.90)
DD3: ref3 blend + aggressive clip (상위 5% 예측값 cap)
DD4: 4-comp mega33+rank_adj+xgb+rem (lv2 제거 — CC3과 다른 subset)
DD5: 4-comp mega33+rank_adj+lv2+rem (xgb 제거)
DD6: ref3 scipy 최적화 (mega33 낮은 bound: 0.30~0.60) — 덜 mega33 의존적
DD7: ref3 예측값 smooth blend (현재 ref3 + no-gate ref3의 평균)
DD8: ref3 w/ triple gate (3단계: huber, mae, huber again at 0.70+)
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

KEYS = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
ct = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

oofs_all = np.stack([co[k] for k in KEYS], axis=1)
tes_all  = np.stack([ct[k] for k in KEYS], axis=1)

saved = []

def mae(pred): return np.mean(np.abs(pred - y_true))

w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3 = np.clip(oofs_all@w_ref3, 0, None)
bt_ref3 = np.clip(tes_all@w_ref3, 0, None)
print(f"ref3 base OOF: {mae(bo_ref3):.5f}\n")

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
        r=minimize(obj,w0,method='L-BFGS-B',bounds=bounds,options={'maxiter':5000,'ftol':1e-12})
        if r.fun<best: best=r.fun; best_w=r.x
    best_w=np.clip(best_w,0,1); best_w/=best_w.sum()
    for k,w in zip(keys,best_w): print(f"    {k}: {w:.4f}")
    return np.clip(oofs_@best_w,0,None), np.clip(tes_@best_w,0,None), best

def std_gate(base_oof, base_test, label):
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
    for p1 in np.arange(0.06,0.18,0.01):
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.005,0.080,0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in np.arange(0.15,0.45,0.02):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.005,0.090,0.005):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        bo=b2; bt=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return bo, bt, best

# ── DD1: Scipy 연속 gate 최적화 ──────────────────────────────────
print("=== DD1: Scipy continuous gate optimization ===")
# gate params: [p1, w1, p2, w2] — scipy finds optimal values
def gate_apply(params, base_oof, tier1_oof, tier2_oof):
    p1, w1, p2, w2 = params
    p1=np.clip(p1,0.01,0.49); w1=np.clip(w1,0,0.15)
    p2=np.clip(p2,p1+0.01,0.99); w2=np.clip(w2,0,0.15)
    m1=(clf_oof>p1).astype(float)
    b1=(1-m1*w1)*base_oof+m1*w1*tier1_oof
    m2=(clf_oof>p2).astype(float)
    b2=(1-m2*w2)*b1+m2*w2*tier2_oof
    return b2
def obj_gate(params):
    b2=gate_apply(params, bo_ref3, rh_oof, rm_oof)
    return np.mean(np.abs(b2-y_true))
best_DD1=mae(bo_ref3); best_params_DD1=None
for _ in range(30):
    np.random.seed(_*13+7)
    p0=np.array([np.random.uniform(0.05,0.20),
                 np.random.uniform(0.01,0.06),
                 np.random.uniform(0.20,0.50),
                 np.random.uniform(0.01,0.08)])
    try:
        r=minimize(obj_gate,p0,method='L-BFGS-B',
                   bounds=[(0.01,0.40),(0.001,0.12),(0.05,0.90),(0.001,0.12)],
                   options={'maxiter':2000,'ftol':1e-11})
        if r.fun<best_DD1: best_DD1=r.fun; best_params_DD1=r.x
    except: pass
if best_params_DD1 is not None:
    p1,w1,p2,w2=best_params_DD1
    print(f"  scipy gate: p1={p1:.3f} w1={w1:.3f} p2={p2:.3f} w2={w2:.3f}")
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
    m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
    bo_DD1=(1-m2*w2)*b1+m2*w2*rm_oof; bt_DD1=(1-m2t*w2)*b1t+m2t*w2*rm_te
else:
    bo_DD1=bo_ref3; bt_DD1=bt_ref3
print(f"  [DD1] scipy_gate OOF={best_DD1:.5f}")
save_sub(bt_DD1, best_DD1, 'DD1_scipy_gate')

# ── DD2: 극광폭 gate 탐색 (p1 0.01~0.40) ──────────────────────────
print("\n=== DD2: Extreme wide gate search ===")
best_DD2=mae(bo_ref3); bo_DD2=bo_ref3; bt_DD2=bt_ref3; cfg_DD2='no_gate'
for p1 in np.arange(0.01,0.40,0.02):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005,0.100,0.010):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        for p2 in np.arange(0.05,0.90,0.05):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005,0.100,0.010):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_DD2:
                    best_DD2=mm; cfg_DD2=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    bo_DD2=b2; bt_DD2=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [DD2] wide_gate={cfg_DD2}  OOF={best_DD2:.5f}")
save_sub(bt_DD2, best_DD2, 'DD2_extreme_wide_gate')

# ── DD3: ref3 + aggressive clip (top 5% 예측 cap) ────────────────
print("\n=== DD3: ref3 + prediction clip (95th percentile) ===")
# 훈련셋 y_true의 95th percentile로 cap
clip_val = np.percentile(y_true, 95)
print(f"  clip threshold (95th): {clip_val:.2f}")
bo_DD3_base = np.clip(bo_ref3, 0, clip_val)
bt_DD3_base = np.clip(bt_ref3, 0, clip_val)
print(f"  clipped OOF: {mae(bo_DD3_base):.5f}")
bo_DD3g, bt_DD3g, bm_DD3 = std_gate(bo_DD3_base, bt_DD3_base, 'DD3')
save_sub(bt_DD3g, bm_DD3, 'DD3_clip95')

# ── DD4: 4-comp mega33+rank+xgb+rem (lv2 제거) ──────────────────
print("\n=== DD4: 4-comp no-lv2 (mega33+rank+xgb+rem) ===")
keys_DD4 = ['mega33','rank_adj','oracle_xgb','oracle_rem']
bo_DD4, bt_DD4, bm_DD4_base = scipy_blend(keys_DD4, n_trials=25)
print(f"  no-lv2 OOF: {bm_DD4_base:.5f}")
bo_DD4g, bt_DD4g, bm_DD4 = std_gate(bo_DD4, bt_DD4, 'DD4')
save_sub(bt_DD4g, bm_DD4, 'DD4_nolv2')

# ── DD5: 4-comp mega33+rank+lv2+rem (xgb 제거) ──────────────────
print("\n=== DD5: 4-comp no-xgb (mega33+rank+lv2+rem) ===")
keys_DD5 = ['mega33','rank_adj','oracle_lv2','oracle_rem']
bo_DD5, bt_DD5, bm_DD5_base = scipy_blend(keys_DD5, n_trials=25)
print(f"  no-xgb OOF: {bm_DD5_base:.5f}")
bo_DD5g, bt_DD5g, bm_DD5 = std_gate(bo_DD5, bt_DD5, 'DD5')
save_sub(bt_DD5g, bm_DD5, 'DD5_noxgb')

# ── DD6: mega33 낮은 bound scipy (0.30~0.60) ─────────────────────
print("\n=== DD6: mega33 lower bound (0.30~0.60), more oracle weight ===")
def obj_dd6(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
best_DD6=999; best_wDD6=None
for _ in range(30):
    np.random.seed(_+900)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_dd6,w0,method='L-BFGS-B',
        bounds=[(0.30,0.60),(0,0.20),(0,0.25),(0,0.30),(0,0.20)],
        options={'maxiter':5000,'ftol':1e-12})
    if r.fun<best_DD6: best_DD6=r.fun; best_wDD6=r.x
best_wDD6=np.clip(best_wDD6,0,1); best_wDD6/=best_wDD6.sum()
for k,w in zip(KEYS,best_wDD6): print(f"    {k}: {w:.4f}")
bo_DD6=np.clip(oofs_all@best_wDD6,0,None); bt_DD6=np.clip(tes_all@best_wDD6,0,None)
print(f"  low-mega33 OOF: {best_DD6:.5f}")
bo_DD6g, bt_DD6g, bm_DD6 = std_gate(bo_DD6, bt_DD6, 'DD6')
save_sub(bt_DD6g, bm_DD6, 'DD6_lowmega33')

# ── DD7: no-gate + gated smooth blend ────────────────────────────
print("\n=== DD7: smooth blend of ref3 (no-gate) + ref3 (standard gate) ===")
# First get gated ref3
bo_ref3_gated, bt_ref3_gated, bm_ref3_gated = std_gate(bo_ref3, bt_ref3, 'ref3_std')
print(f"  gated ref3 OOF: {bm_ref3_gated:.5f}")
# Blend no-gate + gated
best_DD7=999; best_alpha_DD7=0.5
for alpha in np.arange(0.1, 0.9, 0.05):
    blend_oof = alpha*bo_ref3 + (1-alpha)*bo_ref3_gated
    mm=mae(blend_oof)
    if mm<best_DD7: best_DD7=mm; best_alpha_DD7=alpha
print(f"  best alpha={best_alpha_DD7:.2f}  smooth OOF={best_DD7:.5f}")
bt_DD7 = best_alpha_DD7*bt_ref3 + (1-best_alpha_DD7)*bt_ref3_gated
save_sub(bt_DD7, best_DD7, 'DD7_smooth_blend')

# ── DD8: triple gate (3단계) ─────────────────────────────────────
print("\n=== DD8: triple gate (3 tiers: huber, mae, huber@high) ===")
best_DD8=mae(bo_ref3); bo_DD8=bo_ref3; bt_DD8=bt_ref3; cfg_DD8='no_gate'
for p1 in np.arange(0.07,0.15,0.02):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.015,0.050,0.010):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        for p2 in np.arange(0.20,0.40,0.04):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.015,0.060,0.010):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof; b2t=(1-m2t*w2)*b1t+m2t*w2*rm_te
                for p3 in np.arange(0.50,0.80,0.10):
                    if p3<=p2: continue
                    m3=(clf_oof>p3).astype(float); m3t=(clf_test>p3).astype(float)
                    for w3 in np.arange(0.010,0.040,0.010):
                        b3=(1-m3*w3)*b2+m3*w3*rh_oof  # huber again at high conf
                        mm=mae(b3)
                        if mm<best_DD8:
                            best_DD8=mm
                            cfg_DD8=f'p{p1:.2f}+p{p2:.2f}+p{p3:.2f}'
                            bo_DD8=b3; bt_DD8=(1-m3t*w3)*b2t+m3t*w3*rh_te
print(f"  [DD8] triple_gate={cfg_DD8}  OOF={best_DD8:.5f}")
save_sub(bt_DD8, best_DD8, 'DD8_triple_gate')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("DD ROUND SUMMARY (ref3-only)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
