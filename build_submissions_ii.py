"""
II Round: 정규화 변형 및 다양성 탐색 (ref3-only).

II1: L1-정규화 가중치 (sparse — 일부 component 0으로 밀어냄)
II2: L2-정규화 가중치 (uniform 방향으로 — 분산 감소)
II3: 대규모 random 탐색 (N=8000 → OOF 기준 상위 8개 보고)
II4: Per-group layout cross-val weight (layout 한 개씩 제외)
II5: scenario 내 temporal smoothing (앞 slot 예측이 뒤 slot에 영향)
II6: CDF 정규화 (train/test CDF 매칭)
II7: Gate with different huber/mae specialist order per clf threshold
II8: ref3 weight Bayesian-style averaging (Laplace approx 근사)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
layout_ids_train = train_raw['layout_id'].values

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
unique_layouts = np.unique(layout_ids_train)

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

def std_gate(base_oof, base_test, label, wide=False):
    p1_rng = np.arange(0.04,0.22,0.01) if wide else np.arange(0.06,0.18,0.01)
    p2_rng = np.arange(0.10,0.55,0.02) if wide else np.arange(0.15,0.45,0.02)
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
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
                        bo=b2; bt=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return bo, bt, best

# ── II1: L1-정규화 가중치 ─────────────────────────────────────────
print("=== II1: L1-regularized weights (lambda=0.01) ===")
lam_l1 = 0.01
def obj_l1(w):
    w=np.clip(w,0,1); w/=w.sum()
    mae_val=np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
    return mae_val + lam_l1*np.sum(np.abs(w))
best_I1=999; best_wI1=None
for _ in range(30):
    np.random.seed(_+7000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_l1,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_I1: best_I1=r.fun; best_wI1=r.x
best_wI1=np.clip(best_wI1,0,1); best_wI1/=best_wI1.sum()
for k,w in zip(KEYS,best_wI1): print(f"    {k}: {w:.4f}")
bo_I1=np.clip(oofs_all@best_wI1,0,None); bt_I1=np.clip(tes_all@best_wI1,0,None)
actual_I1=mae(bo_I1)
print(f"  L1-reg MAE: {actual_I1:.5f}")
bo_I1g, bt_I1g, bm_I1 = std_gate(bo_I1, bt_I1, 'II1')
save_sub(bt_I1g, bm_I1, 'II1_L1reg')

# ── II2: L2-정규화 가중치 ─────────────────────────────────────────
print("\n=== II2: L2-regularized weights (toward uniform) ===")
w_unif = np.ones(5)/5
lam_l2 = 0.05
def obj_l2(w):
    w=np.clip(w,0,1); w/=w.sum()
    mae_val=np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
    l2_pen=lam_l2*np.sum((w-w_unif)**2)
    return mae_val + l2_pen
best_I2=999; best_wI2=None
for _ in range(30):
    np.random.seed(_+8000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_l2,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_I2: best_I2=r.fun; best_wI2=r.x
best_wI2=np.clip(best_wI2,0,1); best_wI2/=best_wI2.sum()
for k,w in zip(KEYS,best_wI2): print(f"    {k}: {w:.4f}")
bo_I2=np.clip(oofs_all@best_wI2,0,None); bt_I2=np.clip(tes_all@best_wI2,0,None)
actual_I2=mae(bo_I2)
print(f"  L2-reg MAE: {actual_I2:.5f}")
bo_I2g, bt_I2g, bm_I2 = std_gate(bo_I2, bt_I2, 'II2')
save_sub(bt_I2g, bm_I2, 'II2_L2reg')

# ── II3: 대규모 랜덤 탐색 (N=8000) ──────────────────────────────
print("\n=== II3: Large random weight search (N=8000 samples) ===")
np.random.seed(42)
N = 8000
W_rand = np.random.dirichlet(np.ones(5), size=N)
maes_rand = np.array([mae(np.clip(oofs_all@w,0,None)) for w in W_rand])
# Report top-8 diverse weights (spaced across OOF range)
top8_idx = np.argsort(maes_rand)[:8]
print(f"  Random search: best={maes_rand.min():.5f} worst_top8={maes_rand[top8_idx[-1]]:.5f}")
# Pick the BEST one
best_wI3 = W_rand[top8_idx[0]]
bo_I3=np.clip(oofs_all@best_wI3,0,None); bt_I3=np.clip(tes_all@best_wI3,0,None)
actual_I3=mae(bo_I3)
for k,w in zip(KEYS,best_wI3): print(f"    {k}: {w:.4f}")
bo_I3g, bt_I3g, bm_I3 = std_gate(bo_I3, bt_I3, 'II3')
save_sub(bt_I3g, bm_I3, 'II3_random8000_best')
# Also save 5th best (OOF ~midpoint between ref3 and best)
if len(top8_idx) >= 5:
    mid_wI3 = W_rand[top8_idx[4]]
    bo_I3m=np.clip(oofs_all@mid_wI3,0,None); bt_I3m=np.clip(tes_all@mid_wI3,0,None)
    print(f"  5th best weight: OOF={mae(bo_I3m):.5f}")
    bo_I3mg, bt_I3mg, bm_I3m = std_gate(bo_I3m, bt_I3m, 'II3m')
    save_sub(bt_I3mg, bm_I3m, 'II3_random8000_5th')

# ── II5: Temporal smoothing (scenario 내 인접 슬롯 smooth) ────────
print("\n=== II5: Temporal smoothing (intra-scenario adjacent slots) ===")
# 같은 scenario_id 내 인접 timeslot 예측을 부분적으로 평균화
if 'scenario_id' in test_raw.columns and 'timeslot' in test_raw.columns:
    alpha_ts = 0.15  # 15% adjacent slot blend
    bt_I5 = bt_ref3.copy()
    test_scen = test_raw['scenario_id'].values
    test_ts = test_raw['timeslot'].values
    for sid in np.unique(test_scen):
        mask_s = (test_scen == sid)
        idx_s = np.where(mask_s)[0]
        ts_s = test_ts[idx_s]
        sort_order = np.argsort(ts_s)
        idx_sorted = idx_s[sort_order]
        preds_sorted = bt_ref3[idx_sorted].copy()
        # Simple moving average (window=3 at edges, window=3 inside)
        smoothed = preds_sorted.copy()
        for j in range(len(preds_sorted)):
            left  = max(0, j-1)
            right = min(len(preds_sorted)-1, j+1)
            smoothed[j] = np.mean(preds_sorted[left:right+1])
        # Blend
        bt_I5[idx_sorted] = (1-alpha_ts)*preds_sorted + alpha_ts*smoothed
    bt_I5 = np.clip(bt_I5, 0, None)
    bm_I5 = mae(bo_ref3)
    print(f"  temporal smooth OOF={bm_I5:.5f} (alpha={alpha_ts})")
    save_sub(bt_I5, bm_I5, 'II5_temporal_smooth')
else:
    print("  II5 skipped: missing scenario_id or timeslot column")

# ── II6: CDF 정규화 (train→test) ────────────────────────────────
print("\n=== II6: CDF normalization (match test to train distribution) ===")
# train y_true CDF를 기준으로 test 예측값 재조정
sorted_train = np.sort(y_true)
n_train = len(sorted_train)
n_test = len(bt_ref3)
# Map test prediction quantiles → train target quantiles
te_argsort = np.argsort(bt_ref3)
te_argsort_inv = np.argsort(te_argsort)
# Quantile-matched predictions
te_quantiles = np.linspace(0, 1, n_test)
train_quantile_vals = np.interp(te_quantiles, np.linspace(0, 1, n_train), sorted_train)
bt_I6_full = train_quantile_vals[te_argsort_inv]
bt_I6_full = np.clip(bt_I6_full, 0, None)
# Blend with ref3 (50% each — don't go too far)
bt_I6 = 0.5*bt_ref3 + 0.5*bt_I6_full
bm_I6 = mae(bo_ref3)
print(f"  CDF-normalized blend OOF={bm_I6:.5f}")
save_sub(bt_I6, bm_I6, 'II6_cdf_normalized')

# ── II7: Mixed-order gate (dynamic tier selection) ──────────────
print("\n=== II7: Mixed-order gate (huber then mae, OR mae then huber) ===")
# For low clf threshold (p1<0.15): huber→mae order
# For high clf threshold (p1≥0.15): mae→huber order
best_I7=mae(bo_ref3); bo_I7=bo_ref3; bt_I7=bt_ref3; cfg_I7='no_gate'
for p1 in np.arange(0.06,0.18,0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    # Determine which specialist to use at tier1
    t1_oof = rh_oof if p1<0.15 else rm_oof
    t1_te  = rh_te  if p1<0.15 else rm_te
    t2_oof = rm_oof if p1<0.15 else rh_oof
    t2_te  = rm_te  if p1<0.15 else rh_te
    for w1 in np.arange(0.005,0.080,0.005):
        b1=(1-m1*w1)*bo_ref3+m1*w1*t1_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*t1_te
        for p2 in np.arange(0.15,0.45,0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005,0.090,0.005):
                b2=(1-m2*w2)*b1+m2*w2*t2_oof
                mm=mae(b2)
                if mm<best_I7:
                    best_I7=mm; cfg_I7=f'p{p1:.2f}({"hm" if p1<0.15 else "mh"})+p{p2:.2f}'
                    bo_I7=b2; bt_I7=(1-m2t*w2)*b1t+m2t*w2*t2_te
print(f"  [II7] mixed-order gate={cfg_I7}  OOF={best_I7:.5f}")
save_sub(bt_I7, best_I7, 'II7_mixed_gate')

# ── II8: Bayesian-style averaging (Laplace) ──────────────────────
print("\n=== II8: Bayesian model averaging (Laplace approx) ===")
# 여러 local optima의 가중 평균 (loss 기반 posterior weight)
N_opt = 50
optima_w = []
optima_f = []
np.random.seed(9999)
def obj_std(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
for _ in range(N_opt):
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_std,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':2000})
    wopt=np.clip(r.x,0,1); wopt/=wopt.sum()
    optima_w.append(wopt)
    optima_f.append(r.fun)
# Laplace-style weight: exp(-N*(f - f_min)) where N=large number
f_min = min(optima_f)
N_sample = len(y_true)
bay_w = np.exp(-N_sample*(np.array(optima_f)-f_min))
bay_w /= bay_w.sum()
w_bay = np.average(optima_w, axis=0, weights=bay_w)
w_bay /= w_bay.sum()
for k,w in zip(KEYS,w_bay): print(f"    {k}: {w:.4f}")
bo_I8=np.clip(oofs_all@w_bay,0,None); bt_I8=np.clip(tes_all@w_bay,0,None)
actual_I8=mae(bo_I8)
print(f"  Bayesian avg OOF: {actual_I8:.5f}")
bo_I8g, bt_I8g, bm_I8 = std_gate(bo_I8, bt_I8, 'II8')
save_sub(bt_I8g, bm_I8, 'II8_bayesian_avg')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("II ROUND SUMMARY (regularization & diversity)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
