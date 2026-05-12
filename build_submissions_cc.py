"""
CC Round: ref3-only 심화 실험 (gate/weight 변형).
All models: ONLY mega33, rank_adj, oracle_xgb, oracle_lv2, oracle_rem.
No mega34, no oracle_cb, no oracle_v31.

CC1: gate tier 순서 역전 (tier1=mae_spec, tier2=huber_spec)
CC2: ref3 no-gate baseline (게이트 기여도 측정)
CC3: 3-comp: mega33+oracle_xgb+oracle_lv2 (rank_adj+rem 제거)
CC4: 2-comp: mega33+oracle_lv2 (가장 강한 오라클만)
CC5: scipy 100 trials (R3A보다 2배 많은 재시작)
CC6: single-tier mae gate (R3H는 huber, 이건 mae)
CC7: equal weights (1/5 each) + fine gate
CC8: top CC+R3 파일 CSV ensemble (ref3-only 파일만)
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

def gate_dual(base_oof, base_test, label, tier1_oof, tier1_te, tier2_oof, tier2_te,
              p1_rng=None, p2_rng=None):
    if p1_rng is None: p1_rng = np.arange(0.06,0.18,0.01)
    if p2_rng is None: p2_rng = np.arange(0.15,0.45,0.02)
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
    for p1 in p1_rng:
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.005,0.080,0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*tier1_oof
            b1t=(1-m1t*w1)*base_test+m1t*w1*tier1_te
            for p2 in p2_rng:
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.005,0.090,0.005):
                    b2=(1-m2*w2)*b1+m2*w2*tier2_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        bo=b2; bt=(1-m2t*w2)*b1t+m2t*w2*tier2_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return bo, bt, best

def gate_fine(base_oof, base_test, label):
    """Fine gate: 0.002 step around typical best region."""
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
    for p1 in np.arange(0.07,0.17,0.005):
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.008,0.060,0.004):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof
            b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in np.arange(0.18,0.42,0.01):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.010,0.070,0.004):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; cfg=f'p{p1:.3f}w{w1:.3f}+p{p2:.3f}w{w2:.3f}'
                        bo=b2; bt=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return bo, bt, best

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
        w=np.clip(w,0,1); w/=w.sum()
        return np.mean(np.abs(np.clip(oofs_@w,0,None)-y_true))
    best=999; best_w=None
    for _ in range(n_trials):
        w0=np.random.dirichlet(np.ones(len(keys)))
        r=minimize(obj,w0,method='L-BFGS-B',bounds=bounds,options={'maxiter':5000,'ftol':1e-12})
        if r.fun<best: best=r.fun; best_w=r.x
    best_w=np.clip(best_w,0,1); best_w/=best_w.sum()
    oof_=np.clip(oofs_@best_w,0,None); te_=np.clip(tes_@best_w,0,None)
    for k,w in zip(keys,best_w): print(f"    {k}: {w:.4f}")
    return oof_, te_, best, best_w

# ref3 exact weights
w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3 = np.clip(oofs_all@w_ref3, 0, None)
bt_ref3 = np.clip(tes_all@w_ref3, 0, None)
print(f"ref3 base OOF: {mae(bo_ref3):.5f}")
print()

# ── CC1: 게이트 순서 역전 (tier1=mae_spec, tier2=huber_spec) ────
print("=== CC1: gate tier 역전 (mae-first, then huber) ===")
bo_CC1, bt_CC1, bm_CC1 = gate_dual(
    bo_ref3, bt_ref3, 'CC1',
    tier1_oof=rm_oof, tier1_te=rm_te,   # tier1: mae specialist
    tier2_oof=rh_oof, tier2_te=rh_te,   # tier2: huber specialist
)
save_sub(bt_CC1, bm_CC1, 'CC1_gate_reversed')

# ── CC2: ref3 no-gate baseline ──────────────────────────────────
print("\n=== CC2: ref3 no-gate (게이트 없는 순수 blend) ===")
bm_CC2 = mae(bo_ref3)
print(f"  [CC2] no_gate  OOF={bm_CC2:.5f}")
save_sub(bt_ref3, bm_CC2, 'CC2_nogate')

# ── CC3: 3-comp mega33+xgb+lv2 ─────────────────────────────────
print("\n=== CC3: 3-comp mega33+oracle_xgb+oracle_lv2 ===")
keys_CC3 = ['mega33','oracle_xgb','oracle_lv2']
bo_CC3, bt_CC3, bm_CC3_base, _ = scipy_blend(keys_CC3, n_trials=30)
print(f"  3-comp OOF: {bm_CC3_base:.5f}")
bo_CC3g, bt_CC3g, bm_CC3 = gate_dual(
    bo_CC3, bt_CC3, 'CC3',
    tier1_oof=rh_oof, tier1_te=rh_te,
    tier2_oof=rm_oof, tier2_te=rm_te,
)
save_sub(bt_CC3g, bm_CC3, 'CC3_3comp_xgb_lv2')

# ── CC4: 2-comp mega33+oracle_lv2 ──────────────────────────────
print("\n=== CC4: 2-comp mega33+oracle_lv2 ===")
keys_CC4 = ['mega33','oracle_lv2']
bo_CC4, bt_CC4, bm_CC4_base, _ = scipy_blend(keys_CC4, n_trials=30)
print(f"  2-comp OOF: {bm_CC4_base:.5f}")
bo_CC4g, bt_CC4g, bm_CC4 = gate_dual(
    bo_CC4, bt_CC4, 'CC4',
    tier1_oof=rh_oof, tier1_te=rh_te,
    tier2_oof=rm_oof, tier2_te=rm_te,
)
save_sub(bt_CC4g, bm_CC4, 'CC4_2comp_lv2only')

# ── CC5: scipy 100 trials (R3A보다 2배) ─────────────────────────
print("\n=== CC5: scipy 100 trials (free bounds) ===")
def obj5(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
best_C5=999; best_w5=None
for seed in range(10):
    np.random.seed(seed*77)
    for _ in range(10):
        w0=np.random.dirichlet(np.ones(5))
        r=minimize(obj5,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':8000,'ftol':1e-13})
        if r.fun<best_C5: best_C5=r.fun; best_w5=r.x
best_w5=np.clip(best_w5,0,1); best_w5/=best_w5.sum()
print(f"  100-trial scipy OOF: {best_C5:.5f}")
for k,w in zip(KEYS,best_w5): print(f"    {k}: {w:.4f}")
bo_C5=np.clip(oofs_all@best_w5,0,None); bt_C5=np.clip(tes_all@best_w5,0,None)
bo_C5g, bt_C5g, bm_C5 = gate_fine(bo_C5, bt_C5, 'CC5')
save_sub(bt_C5g, bm_C5, 'CC5_scipy100')

# ── CC6: single-tier mae gate (R3H는 huber, 이건 mae) ────────────
print("\n=== CC6: single-tier mae gate (ref3 exact weights) ===")
best_C6 = mae(bo_ref3); bo_C6 = bo_ref3; bt_C6 = bt_ref3; cfg_C6='no_gate'
for p1 in np.arange(0.04, 0.22, 0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005, 0.080, 0.005):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rm_oof  # mae specialist (not huber)
        b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rm_te
        mm=mae(b1)
        if mm<best_C6:
            best_C6=mm; cfg_C6=f'p{p1:.2f}w{w1:.3f}'
            bo_C6=b1; bt_C6=b1t
print(f"  [CC6] single-mae-gate={cfg_C6}  OOF={best_C6:.5f}")
save_sub(bt_C6, best_C6, 'CC6_singletier_mae')

# ── CC7: equal weights (1/5) + fine gate ────────────────────────
print("\n=== CC7: equal weights (0.2 each) + fine gate ===")
w_eq = np.ones(5)/5
bo_eq = np.clip(oofs_all@w_eq, 0, None)
bt_eq = np.clip(tes_all@w_eq, 0, None)
print(f"  equal-weight OOF: {mae(bo_eq):.5f}")
bo_C7, bt_C7, bm_C7 = gate_fine(bo_eq, bt_eq, 'CC7')
save_sub(bt_C7, bm_C7, 'CC7_equalweight')

# ── CC8: ref3 weight 섭동 앙상블 (10개 랜덤 weight → 평균) ────────
print("\n=== CC8: ref3 weight perturbation ensemble (10 random) ===")
# ref3 exact weight 근처 ±0.05 랜덤 섭동 10개의 평균
np.random.seed(42)
ens_preds_oof = []; ens_preds_te = []
for i in range(10):
    noise = np.random.uniform(-0.05, 0.05, 5)
    w_perturb = w_ref3 + noise
    w_perturb = np.clip(w_perturb, 0.01, 1)
    w_perturb /= w_perturb.sum()
    ens_preds_oof.append(np.clip(oofs_all@w_perturb, 0, None))
    ens_preds_te.append(np.clip(tes_all@w_perturb, 0, None))
bo_C8 = np.mean(ens_preds_oof, axis=0)
bt_C8 = np.mean(ens_preds_te, axis=0)
print(f"  perturbation ensemble OOF: {mae(bo_C8):.5f}")
bo_C8g, bt_C8g, bm_C8 = gate_dual(
    bo_C8, bt_C8, 'CC8',
    tier1_oof=rh_oof, tier1_te=rh_te,
    tier2_oof=rm_oof, tier2_te=rm_te,
)
save_sub(bt_C8g, bm_C8, 'CC8_perturbation_ens')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("CC ROUND SUMMARY (ref3-only, no extra components)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
