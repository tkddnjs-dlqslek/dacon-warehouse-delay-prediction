"""
JJ Round: Ultra-fine gate search + CC/DD 결과 기반 ensemble.
ref3-only components 전용.

JJ1: Ultra-fine gate (0.002 step, 좁은 범위) — grid 최고점 정밀 탐색
JJ2: ref3 exact weights, 더 넓은 w1/w2 범위 탐색 (w up to 0.15)
JJ3: ref3 + gate: 교차 검증 방식 (각 fold의 best gate → 평균)
JJ4: FF1-type: ref3(seen) + 3가지 unseen 전략 비교
JJ5: test prediction 분포 매칭 (test→train Q-Q)
JJ6: ref3 + sigmoid-weighted gate (연속 threshold)
JJ7: Geometric + arithmetic blend (geo-arith hybrid)
JJ8: ref3 predictions + half-shift to train mean
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
train_layout_ids_set = set(layout_ids_train)
test_layouts = test_raw['layout_id'].values
seen_mask = np.array([lid in train_layout_ids_set for lid in test_layouts])
unseen_mask = ~seen_mask

def mae(pred): return np.mean(np.abs(pred - y_true))

w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3 = np.clip(oofs_all@w_ref3, 0, None)
bt_ref3 = np.clip(tes_all@w_ref3, 0, None)
print(f"ref3 base OOF: {mae(bo_ref3):.5f}")
print(f"seen/unseen test: {seen_mask.sum()}/{unseen_mask.sum()}\n")

def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

# ── JJ1: Ultra-fine gate (0.002 step, narrow range) ─────────────
print("=== JJ1: Ultra-fine gate search (0.002 step, narrow range) ===")
best_J1=mae(bo_ref3); bo_J1=bo_ref3; bt_J1=bt_ref3; cfg_J1='no_gate'
# Narrow range around typical best: p1≈0.08-0.18, p2≈0.20-0.40
for p1 in np.arange(0.06, 0.20, 0.002):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.008, 0.060, 0.003):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        for p2 in np.arange(0.16, 0.44, 0.005):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.008, 0.070, 0.003):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_J1:
                    best_J1=mm; cfg_J1=f'p{p1:.3f}w{w1:.3f}+p{p2:.3f}w{w2:.3f}'
                    bo_J1=b2; bt_J1=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [JJ1] ultra-fine gate={cfg_J1}  OOF={best_J1:.5f}")
save_sub(bt_J1, best_J1, 'JJ1_ultrafine_gate')

# ── JJ2: ref3 + wide w1/w2 (0~0.15) gate ─────────────────────────
print("\n=== JJ2: ref3 + wide weight gate (w up to 0.15) ===")
best_J2=mae(bo_ref3); bo_J2=bo_ref3; bt_J2=bt_ref3; cfg_J2='no_gate'
for p1 in np.arange(0.06,0.18,0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005,0.150,0.005):  # up to 0.15
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        for p2 in np.arange(0.15,0.45,0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005,0.150,0.005):  # up to 0.15
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_J2:
                    best_J2=mm; cfg_J2=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    bo_J2=b2; bt_J2=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [JJ2] wide-weight gate={cfg_J2}  OOF={best_J2:.5f}")
save_sub(bt_J2, best_J2, 'JJ2_wideweight_gate')

# ── JJ3: Cross-fold gate (per GroupKFold fold, avg gate params) ──
print("\n=== JJ3: Cross-fold gate optimization (5-fold) ===")
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
fold_best_params = []
for fold, (tr_idx, va_idx) in enumerate(gkf.split(oofs_all, y_true, groups=layout_ids_train)):
    bo_tr = np.clip(oofs_all[tr_idx]@w_ref3, 0, None)
    y_tr = y_true[tr_idx]
    clf_tr = clf_oof[tr_idx]
    rh_tr = rh_oof[tr_idx]
    rm_tr = rm_oof[tr_idx]
    # Gate search on training fold
    best_f=np.mean(np.abs(bo_tr-y_tr)); best_p=(0,0.02,0.25,0.04)
    for p1 in np.arange(0.06,0.18,0.02):
        m1=(clf_tr>p1).astype(float)
        for w1 in np.arange(0.010,0.060,0.010):
            b1=(1-m1*w1)*bo_tr+m1*w1*rh_tr
            for p2 in np.arange(0.20,0.42,0.04):
                if p2<=p1: continue
                m2=(clf_tr>p2).astype(float)
                for w2 in np.arange(0.010,0.060,0.010):
                    b2=(1-m2*w2)*b1+m2*w2*rm_tr
                    mm=np.mean(np.abs(b2-y_tr))
                    if mm<best_f: best_f=mm; best_p=(p1,w1,p2,w2)
    fold_best_params.append(best_p)
    print(f"  Fold {fold+1}: gate={best_p}  tr_mae={best_f:.5f}")
# Average gate params
avg_p = np.mean(fold_best_params, axis=0)
p1_avg, w1_avg, p2_avg, w2_avg = avg_p
print(f"  Avg gate: p1={p1_avg:.3f} w1={w1_avg:.3f} p2={p2_avg:.3f} w2={w2_avg:.3f}")
m1=(clf_oof>p1_avg).astype(float); m1t=(clf_test>p1_avg).astype(float)
b1=(1-m1*w1_avg)*bo_ref3+m1*w1_avg*rh_oof; b1t=(1-m1t*w1_avg)*bt_ref3+m1t*w1_avg*rh_te
m2=(clf_oof>p2_avg).astype(float); m2t=(clf_test>p2_avg).astype(float)
bo_J3=(1-m2*w2_avg)*b1+m2*w2_avg*rm_oof; bt_J3=(1-m2t*w2_avg)*b1t+m2t*w2_avg*rm_te
bm_J3=mae(bo_J3)
print(f"  [JJ3] cross-fold avg gate OOF={bm_J3:.5f}")
save_sub(bt_J3, bm_J3, 'JJ3_crossfold_gate')

# ── JJ4: FF variants (more unseen layout strategies) ─────────────
print("\n=== JJ4: Extended layout-split strategies ===")
bt_ref3_ngate = np.clip(tes_all@w_ref3, 0, None)
# mega33 with different weights
w_m70 = np.array([0.70,0.10,0.08,0.08,0.04]); w_m70/=w_m70.sum()
w_m60 = np.array([0.60,0.10,0.10,0.13,0.07]); w_m60/=w_m60.sum()
# Apply gate to ref3
best_ref3g=mae(bo_ref3); bo_ref3g=bo_ref3; bt_ref3g=bt_ref3
for p1 in np.arange(0.06,0.18,0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005,0.080,0.005):
        b1=(1-m1*w1)*bo_ref3+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_ref3+m1t*w1*rh_te
        for p2 in np.arange(0.15,0.45,0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005,0.090,0.005):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<best_ref3g:
                    best_ref3g=mm; bo_ref3g=b2; bt_ref3g=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  ref3+gate OOF: {best_ref3g:.5f}")

# Strategy 4a: seen=gated ref3, unseen=mega33(0.70 blend)
bt_unseen_70 = np.clip(tes_all@w_m70, 0, None)
bt_J4a = np.where(seen_mask, bt_ref3g, bt_unseen_70)
save_sub(bt_J4a, best_ref3g, 'JJ4a_gate_vs_m70')

# Strategy 4b: seen=gated ref3, unseen=mega33(0.60 blend)
bt_unseen_60 = np.clip(tes_all@w_m60, 0, None)
bt_J4b = np.where(seen_mask, bt_ref3g, bt_unseen_60)
save_sub(bt_J4b, best_ref3g, 'JJ4b_gate_vs_m60')

# Strategy 4c: seen=gated ref3, unseen=gated ref3 (baseline, gate applied to all)
# This is equivalent to cascade_refined3 with re-found gate
save_sub(bt_ref3g, best_ref3g, 'JJ4c_gated_all_uniform')

# ── JJ5: Test prediction quantile adjustment ─────────────────────
print("\n=== JJ5: Train-test quantile distribution matching ===")
# 테스트 예측값이 훈련 타겟과 같은 quantile 분포를 가지도록 조정
sorted_y = np.sort(y_true)
n_train = len(sorted_y)
n_test = len(bt_ref3)
# Interpolate train quantiles for test quantile positions
te_ranks = np.argsort(np.argsort(bt_ref3)).astype(float)
te_quantiles = te_ranks / (n_test - 1)
matched_preds = np.interp(te_quantiles, np.linspace(0, 1, n_train), sorted_y)
# Blend 30% matched + 70% original
bt_J5 = 0.70 * bt_ref3 + 0.30 * matched_preds
bt_J5 = np.clip(bt_J5, 0, None)
bm_J5 = mae(bo_ref3)
print(f"  Quantile-matched blend OOF={bm_J5:.5f}")
save_sub(bt_J5, bm_J5, 'JJ5_quantile_match')

# ── JJ6: Sigmoid-weighted gate (soft threshold) ─────────────────
print("\n=== JJ6: Sigmoid-weighted gate (soft threshold instead of hard) ===")
# Instead of hard threshold, use sigmoid function for smooth gate
def sigmoid_gate(base_oof, base_test, label):
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
    for p1 in np.arange(0.06,0.18,0.02):
        for steepness in [5.0, 10.0, 20.0]:
            # Soft mask: sigmoid((clf_oof - p1) * steepness)
            m1_soft = 1/(1+np.exp(-steepness*(clf_oof-p1)))
            m1t_soft = 1/(1+np.exp(-steepness*(clf_test-p1)))
            for w1 in np.arange(0.010,0.060,0.010):
                b1=(1-m1_soft*w1)*base_oof+m1_soft*w1*rh_oof
                b1t=(1-m1t_soft*w1)*base_test+m1t_soft*w1*rh_te
                for p2 in np.arange(0.20,0.40,0.04):
                    if p2<=p1: continue
                    m2_soft = 1/(1+np.exp(-steepness*(clf_oof-p2)))
                    m2t_soft = 1/(1+np.exp(-steepness*(clf_test-p2)))
                    for w2 in np.arange(0.010,0.060,0.010):
                        b2=(1-m2_soft*w2)*b1+m2_soft*w2*rm_oof
                        mm=mae(b2)
                        if mm<best:
                            best=mm; cfg=f'p{p1:.2f}s{steepness:.0f}+p{p2:.2f}'
                            bo=b2; bt=(1-m2t_soft*w2)*b1t+m2t_soft*w2*rm_te
    print(f"  [{label}] sigmoid gate={cfg}  OOF={best:.5f}")
    return bo, bt, best
bo_J6, bt_J6, bm_J6 = sigmoid_gate(bo_ref3, bt_ref3, 'JJ6')
save_sub(bt_J6, bm_J6, 'JJ6_sigmoid_gate')

# ── JJ7: Geometric-arithmetic hybrid ────────────────────────────
print("\n=== JJ7: Geometric-arithmetic hybrid blend ===")
eps = 0.01
# Geometric mean of ref3 components
log_stack_oof = np.log(np.clip(np.stack([co[k] for k in KEYS],axis=1), eps, None))
log_stack_te  = np.log(np.clip(np.stack([ct[k] for k in KEYS],axis=1), eps, None))
geo_oof = np.exp(log_stack_oof@w_ref3) - eps
geo_te  = np.exp(log_stack_te@w_ref3) - eps
geo_oof = np.clip(geo_oof, 0, None)
geo_te  = np.clip(geo_te, 0, None)
# Blend geo and arith
for alpha in [0.3, 0.5, 0.7]:
    hybrid_oof = alpha*bo_ref3 + (1-alpha)*geo_oof
    print(f"  alpha={alpha}: hybrid OOF={mae(hybrid_oof):.5f}")
# Use scipy to find best alpha
def obj_hybrid(w):
    a=np.clip(w[0],0,1)
    return np.mean(np.abs(np.clip(a*bo_ref3+(1-a)*geo_oof,0,None)-y_true))
res_J7=minimize(obj_hybrid,[0.5],method='L-BFGS-B',bounds=[(0,1)])
alpha_best=np.clip(res_J7.x[0],0,1)
bo_J7=np.clip(alpha_best*bo_ref3+(1-alpha_best)*geo_oof,0,None)
bt_J7=np.clip(alpha_best*bt_ref3+(1-alpha_best)*geo_te,0,None)
print(f"  Best alpha_arith={alpha_best:.3f}  OOF={mae(bo_J7):.5f}")
std_best=mae(bo_ref3); bo_J7g=bo_J7; bt_J7g=bt_J7
for p1 in np.arange(0.06,0.18,0.01):
    m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1 in np.arange(0.005,0.080,0.005):
        b1=(1-m1*w1)*bo_J7+m1*w1*rh_oof; b1t=(1-m1t*w1)*bt_J7+m1t*w1*rh_te
        for p2 in np.arange(0.15,0.45,0.02):
            if p2<=p1: continue
            m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
            for w2 in np.arange(0.005,0.090,0.005):
                b2=(1-m2*w2)*b1+m2*w2*rm_oof
                mm=mae(b2)
                if mm<std_best:
                    std_best=mm; bo_J7g=b2; bt_J7g=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [JJ7] geo-arith hybrid+gate OOF={std_best:.5f}")
save_sub(bt_J7g, std_best, 'JJ7_geo_arith_hybrid')

# ── JJ8: Half-shift to train mean ────────────────────────────────
print("\n=== JJ8: Prediction shift (50% toward train mean) ===")
train_mean = np.mean(y_true)
# Shift all test predictions 50% toward train mean
bt_J8 = 0.50*bt_ref3 + 0.50*train_mean
bt_J8 = np.clip(bt_J8, 0, None)
bm_J8 = mae(bo_ref3)  # OOF unchanged
print(f"  half-shift to mean ({train_mean:.3f}) OOF={bm_J8:.5f}")
save_sub(bt_J8, bm_J8, 'JJ8_halfshift_mean')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("JJ ROUND SUMMARY (ultra-fine gate & hybrid)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
