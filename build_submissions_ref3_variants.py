"""
REF3 VARIANTS ONLY: cascade_refined3 성분만 사용.
추가 컴포넌트 전부 LB 악화 확인:
  mega34 → 9.7690, oracle_cb → 9.7655, oracle_combined → 9.7739, neural → 9.7776

오직 5개만: mega33, rank_adj, oracle_xgb, oracle_lv2, oracle_rem + cascade gate

전략:
  R3A: ref3 weights scipy 재최적화 (다른 random seed로 더 많은 시도)
  R3B: oracle_lv2 weight 올린 버전 (lv2가 가장 강한 oracle)
  R3C: oracle_xgb weight 올린 버전
  R3D: rank_adj 제거 버전 (mega33+xgb+lv2+rem only)
  R3E: oracle_rem 제거 버전 (mega33+rank+xgb+lv2 only)
  R3F: gate param 세밀 탐색 (ref3 exact weights, 더 넓은 gate space)
  R3G: mega33 weight 올린 버전 (0.60+) — 더 conservative
  R3H: ref3 weights, single-tier gate only (huber만)
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

def blend(w_arr):
    w = np.clip(w_arr,0,1); w /= w.sum()
    return np.clip(oofs_all@w,0,None), np.clip(tes_all@w,0,None)

def blend_keys(keys, w_arr=None):
    oofs_ = np.stack([co[k] for k in keys], axis=1)
    tes_  = np.stack([ct[k] for k in keys], axis=1)
    if w_arr is None: w_arr = np.ones(len(keys))
    w = np.clip(w_arr,0,1); w /= w.sum()
    return np.clip(oofs_@w,0,None), np.clip(tes_@w,0,None)

def gate_search(base_oof, base_test, label, wide=False):
    p1_rng = np.arange(0.04,0.22,0.01) if wide else np.arange(0.06,0.18,0.01)
    p2_rng = np.arange(0.10,0.55,0.02) if wide else np.arange(0.15,0.45,0.02)
    best=mae(base_oof); best_oof=base_oof; best_te=base_test; cfg='no_gate'
    for p1 in p1_rng:
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.005,0.080,0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in p2_rng:
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.005,0.090,0.005):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        best_oof=b2; best_te=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return best_oof, best_te, best

def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

# Individual MAEs for reference
print("Individual component MAEs:")
for k in KEYS:
    print(f"  {k}: {mae(co[k]):.5f}")
print()

# ── R3A: Scipy 재최적화 (50 trials, seed sweep) ─────────────
print("=== R3A: scipy re-optimize ref3 weights (50 trials) ===")
def obj(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
best_A=999; best_wA=None
for seed in range(5):
    np.random.seed(seed*100)
    for _ in range(10):
        w0=np.random.dirichlet(np.ones(5))
        r=minimize(obj,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':5000,'ftol':1e-12})
        if r.fun<best_A: best_A=r.fun; best_wA=r.x
best_wA=np.clip(best_wA,0,1); best_wA/=best_wA.sum()
print(f"  best scipy OOF: {best_A:.5f}")
for k,w in zip(KEYS,best_wA): print(f"    {k}: {w:.4f}")
bo_A=np.clip(oofs_all@best_wA,0,None); bt_A=np.clip(tes_all@best_wA,0,None)
bo_Ag, bt_Ag, bm_A = gate_search(bo_A, bt_A, 'R3A', wide=True)
save_sub(bt_Ag, bm_A, 'R3A_scipy_reopt')

# ── R3B: oracle_lv2 weight boost ──────────────────────────────
print("\n=== R3B: oracle_lv2 boosted (lv2>=0.25) ===")
best_B=999; best_wB=None
for _ in range(20):
    np.random.seed(_ + 200)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj,w0,method='L-BFGS-B',
        bounds=[(0.35,0.70),(0,0.15),(0,0.20),(0.25,0.45),(0,0.15)],
        options={'maxiter':3000,'ftol':1e-11})
    if r.fun<best_B: best_B=r.fun; best_wB=r.x
best_wB=np.clip(best_wB,0,1); best_wB/=best_wB.sum()
print(f"  lv2-boost OOF: {best_B:.5f}");
for k,w in zip(KEYS,best_wB): print(f"    {k}: {w:.4f}")
bo_B=np.clip(oofs_all@best_wB,0,None); bt_B=np.clip(tes_all@best_wB,0,None)
bo_Bg, bt_Bg, bm_B = gate_search(bo_B, bt_B, 'R3B')
save_sub(bt_Bg, bm_B, 'R3B_lv2boost')

# ── R3C: oracle_xgb weight boost ────────────────────────────
print("\n=== R3C: oracle_xgb boosted (xgb>=0.20) ===")
best_C=999; best_wC=None
for _ in range(20):
    np.random.seed(_ + 300)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj,w0,method='L-BFGS-B',
        bounds=[(0.35,0.70),(0,0.15),(0.20,0.40),(0,0.30),(0,0.15)],
        options={'maxiter':3000,'ftol':1e-11})
    if r.fun<best_C: best_C=r.fun; best_wC=r.x
best_wC=np.clip(best_wC,0,1); best_wC/=best_wC.sum()
print(f"  xgb-boost OOF: {best_C:.5f}")
for k,w in zip(KEYS,best_wC): print(f"    {k}: {w:.4f}")
bo_C=np.clip(oofs_all@best_wC,0,None); bt_C=np.clip(tes_all@best_wC,0,None)
bo_Cg, bt_Cg, bm_C = gate_search(bo_C, bt_C, 'R3C')
save_sub(bt_Cg, bm_C, 'R3C_xgbboost')

# ── R3D: rank_adj 제거 ────────────────────────────────────────
print("\n=== R3D: rank_adj removed (mega33+xgb+lv2+rem) ===")
keys_D=['mega33','oracle_xgb','oracle_lv2','oracle_rem']
oD=np.stack([co[k] for k in keys_D],axis=1); tD=np.stack([ct[k] for k in keys_D],axis=1)
def obj_D(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oD@w,0,None)-y_true))
best_D=999; best_wD=None
for _ in range(20):
    np.random.seed(_ + 400)
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_D,w0,method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':3000})
    if r.fun<best_D: best_D=r.fun; best_wD=r.x
best_wD=np.clip(best_wD,0,1); best_wD/=best_wD.sum()
print(f"  no-rank OOF: {best_D:.5f}");
for k,w in zip(keys_D,best_wD): print(f"    {k}: {w:.4f}")
bo_D=np.clip(oD@best_wD,0,None); bt_D=np.clip(tD@best_wD,0,None)
bo_Dg, bt_Dg, bm_D = gate_search(bo_D, bt_D, 'R3D')
save_sub(bt_Dg, bm_D, 'R3D_norank')

# ── R3E: oracle_rem 제거 ─────────────────────────────────────
print("\n=== R3E: oracle_rem removed (mega33+rank+xgb+lv2) ===")
keys_E=['mega33','rank_adj','oracle_xgb','oracle_lv2']
oE=np.stack([co[k] for k in keys_E],axis=1); tE=np.stack([ct[k] for k in keys_E],axis=1)
def obj_E(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oE@w,0,None)-y_true))
best_E=999; best_wE=None
for _ in range(20):
    np.random.seed(_ + 500)
    w0=np.random.dirichlet(np.ones(4))
    r=minimize(obj_E,w0,method='L-BFGS-B',bounds=[(0,1)]*4,options={'maxiter':3000})
    if r.fun<best_E: best_E=r.fun; best_wE=r.x
best_wE=np.clip(best_wE,0,1); best_wE/=best_wE.sum()
print(f"  no-rem OOF: {best_E:.5f}");
for k,w in zip(keys_E,best_wE): print(f"    {k}: {w:.4f}")
bo_E=np.clip(oE@best_wE,0,None); bt_E=np.clip(tE@best_wE,0,None)
bo_Eg, bt_Eg, bm_E = gate_search(bo_E, bt_E, 'R3E')
save_sub(bt_Eg, bm_E, 'R3E_norem')

# ── R3F: ref3 exact weights + 광폭 gate 탐색 ─────────────────
print("\n=== R3F: ref3 exact weights + wide gate search ===")
w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3=np.clip(oofs_all@w_ref3,0,None); bt_ref3=np.clip(tes_all@w_ref3,0,None)
print(f"  ref3 base OOF: {mae(bo_ref3):.5f}")
bo_Fg, bt_Fg, bm_F = gate_search(bo_ref3, bt_ref3, 'R3F', wide=True)
save_sub(bt_Fg, bm_F, 'R3F_widegate')

# ── R3G: mega33 heavy (>=0.60) ───────────────────────────────
print("\n=== R3G: mega33 heavy (>=0.60) ===")
best_G=999; best_wG=None
for _ in range(20):
    np.random.seed(_ + 600)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj,w0,method='L-BFGS-B',
        bounds=[(0.60,0.80),(0,0.12),(0,0.18),(0,0.20),(0,0.12)],
        options={'maxiter':3000,'ftol':1e-11})
    if r.fun<best_G: best_G=r.fun; best_wG=r.x
best_wG=np.clip(best_wG,0,1); best_wG/=best_wG.sum()
print(f"  mega33-heavy OOF: {best_G:.5f}")
for k,w in zip(KEYS,best_wG): print(f"    {k}: {w:.4f}")
bo_G=np.clip(oofs_all@best_wG,0,None); bt_G=np.clip(tes_all@best_wG,0,None)
bo_Gg, bt_Gg, bm_G = gate_search(bo_G, bt_G, 'R3G')
save_sub(bt_Gg, bm_G, 'R3G_mega33heavy')

# ── R3H: ref3 exact + single-tier gate (huber만) ─────────────
print("\n=== R3H: ref3 exact + single-tier gate (huber only) ===")
best_H = mae(bo_ref3); bo_Hg2 = bo_ref3; bt_Hg2 = bt_ref3; cfg_H='no_gate'
for p1 in np.arange(0.04, 0.22, 0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005, 0.080, 0.005):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        mm=mae(b1)
        if mm<best_H:
            best_H=mm; cfg_H=f'p{p1:.2f}w{w1:.3f}'
            bo_Hg2=b1; bt_Hg2=b1t
print(f"  [R3H] single-gate={cfg_H}  OOF={best_H:.5f}")
save_sub(bt_Hg2, best_H, 'R3H_singletier')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("REF3-VARIANTS SUMMARY (mega33-only, no extra components)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
