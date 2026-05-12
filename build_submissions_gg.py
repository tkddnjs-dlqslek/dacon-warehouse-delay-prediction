"""
GG Round: Layout-generalization 최적화 (핵심 아이디어).

GG1: Bootstrap layout generalization (80% seen → 20% holdout 반복)
     → unseen layout에 잘 맞는 가중치를 학습 데이터 내에서 찾는다
GG2: Layout-stratified OOF weight (holdout layout에서 최소 MAE)
GG3: Median blend (mean 대신 median — 이상치 robust)
GG4: ref3 + huber loss 직접 blend (MAE 대신 Huber loss 최소화)
GG5: Time-slot conditional weighting (timeslot 그룹별 다른 가중치)
GG6: Trimmed mean blend (상하위 20% 제외한 평균)
GG7: Log-space combination (로그 변환 후 blend → 역변환)
GG8: ref3 w/ outlier-robust gate (예측 분산이 낮은 샘플만 gate)
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
def mae_subset(pred, idx): return np.mean(np.abs(pred[idx] - y_true[idx]))

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

# ── GG1: Bootstrap layout generalization ─────────────────────────
print("=== GG1: Bootstrap layout generalization (K=20 trials) ===")
# 아이디어: train의 80% layout으로 weights 학습, 20% layout으로 검증 → 평균화
# unseen layout 일반화에 더 좋은 가중치를 찾는다
K = 20
holdout_frac = 0.20
bootstrap_weights = []
np.random.seed(2024)
for k in range(K):
    # 80/20 layout split
    n_holdout = max(1, int(len(unique_layouts)*holdout_frac))
    holdout_lids = np.random.choice(unique_layouts, n_holdout, replace=False)
    holdout_lids_set = set(holdout_lids)
    holdout_idx = np.array([i for i,lid in enumerate(layout_ids_train) if lid in holdout_lids_set])
    train_idx   = np.array([i for i,lid in enumerate(layout_ids_train) if lid not in holdout_lids_set])
    if len(train_idx)==0 or len(holdout_idx)==0: continue
    # Optimize weights on train_idx
    def obj_gg1(w, idx=train_idx):
        w=np.clip(w,0,1); w/=w.sum()
        pred=np.clip(oofs_all[idx]@w,0,None)
        return np.mean(np.abs(pred-y_true[idx]))
    best_k=999; best_wk=None
    for _ in range(5):
        w0=np.random.dirichlet(np.ones(5))
        r=minimize(obj_gg1,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':2000})
        if r.fun<best_k: best_k=r.fun; best_wk=r.x
    if best_wk is not None:
        best_wk=np.clip(best_wk,0,1); best_wk/=best_wk.sum()
        # Evaluate on holdout
        pred_ho=np.clip(oofs_all[holdout_idx]@best_wk,0,None)
        ho_mae=np.mean(np.abs(pred_ho-y_true[holdout_idx]))
        bootstrap_weights.append(best_wk)
        if k % 5 == 0:
            print(f"  iter {k}: train_mae={best_k:.5f} holdout_mae={ho_mae:.5f}")
if bootstrap_weights:
    w_GG1 = np.mean(bootstrap_weights, axis=0)
    w_GG1 /= w_GG1.sum()
    print(f"  GG1 bootstrap-avg weights:")
    for k,w in zip(KEYS,w_GG1): print(f"    {k}: {w:.4f}")
    bo_GG1=np.clip(oofs_all@w_GG1,0,None); bt_GG1=np.clip(tes_all@w_GG1,0,None)
    bm_GG1=mae(bo_GG1)
    print(f"  GG1 full OOF: {bm_GG1:.5f}")
    bo_GG1g, bt_GG1g, bm_GG1g = std_gate(bo_GG1, bt_GG1, 'GG1')
    save_sub(bt_GG1g, bm_GG1g, 'GG1_bootstrap_layout')

# ── GG2: Holdout layout minimization ─────────────────────────────
print("\n=== GG2: Direct holdout layout MAE minimization ===")
# 각 layout을 한번씩 holdout으로 빼고, 나머지로 학습한 가중치들의 가중 평균
# (leave-one-layout-out과 유사)
n_layouts = len(unique_layouts)
sample_lids = unique_layouts[::max(1,n_layouts//20)]  # 최대 20개 layout만 샘플링
holdout_weights_list = []
holdout_maes = []
for lid in sample_lids:
    holdout_idx = np.where(layout_ids_train==lid)[0]
    train_idx   = np.where(layout_ids_train!=lid)[0]
    if len(holdout_idx)<5: continue
    def obj_gg2(w):
        w=np.clip(w,0,1); w/=w.sum()
        pred=np.clip(oofs_all[train_idx]@w,0,None)
        return np.mean(np.abs(pred-y_true[train_idx]))
    best_k2=999; best_wk2=None
    np.random.seed(int(lid)*7+42)
    for _ in range(5):
        w0=np.random.dirichlet(np.ones(5))
        r=minimize(obj_gg2,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':2000})
        if r.fun<best_k2: best_k2=r.fun; best_wk2=r.x
    if best_wk2 is not None:
        best_wk2=np.clip(best_wk2,0,1); best_wk2/=best_wk2.sum()
        pred_ho=np.clip(oofs_all[holdout_idx]@best_wk2,0,None)
        ho_mae=np.mean(np.abs(pred_ho-y_true[holdout_idx]))
        holdout_weights_list.append(best_wk2)
        holdout_maes.append(ho_mae)
if holdout_weights_list:
    # Inverse-MAE weighted average of holdout weights
    inv_maes = 1.0 / np.array(holdout_maes)
    inv_maes /= inv_maes.sum()
    w_GG2 = np.average(holdout_weights_list, axis=0, weights=inv_maes)
    w_GG2 /= w_GG2.sum()
    print(f"  GG2 holdout-weighted avg weights:")
    for k,w in zip(KEYS,w_GG2): print(f"    {k}: {w:.4f}")
    bo_GG2=np.clip(oofs_all@w_GG2,0,None); bt_GG2=np.clip(tes_all@w_GG2,0,None)
    bm_GG2=mae(bo_GG2)
    print(f"  GG2 full OOF: {bm_GG2:.5f}")
    bo_GG2g, bt_GG2g, bm_GG2g = std_gate(bo_GG2, bt_GG2, 'GG2')
    save_sub(bt_GG2g, bm_GG2g, 'GG2_holdout_layout')

# ── GG3: Median blend ─────────────────────────────────────────────
print("\n=== GG3: Median blend of 5 ref3 components ===")
bo_med = np.median(np.stack([np.clip(co[k],0,None) for k in KEYS], axis=1), axis=1)
bt_med = np.median(np.stack([np.clip(ct[k],0,None) for k in KEYS], axis=1), axis=1)
print(f"  median OOF: {mae(bo_med):.5f}")
bo_GG3, bt_GG3, bm_GG3 = std_gate(bo_med, bt_med, 'GG3', wide=True)
save_sub(bt_GG3, bm_GG3, 'GG3_median_blend')

# ── GG4: Huber loss minimization (delta=5.0) ────────────────────
print("\n=== GG4: Huber-loss weight optimization (delta=5) ===")
delta = 5.0
def huber_loss(pred, delta=delta):
    err = pred - y_true
    abs_err = np.abs(err)
    return np.mean(np.where(abs_err<=delta, 0.5*err**2, delta*(abs_err-0.5*delta)))
def obj_huber(w):
    w=np.clip(w,0,1); w/=w.sum()
    return huber_loss(np.clip(oofs_all@w,0,None))
best_GG4=999; best_wGG4=None
for _ in range(30):
    np.random.seed(_+2000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_huber,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_GG4: best_GG4=r.fun; best_wGG4=r.x
best_wGG4=np.clip(best_wGG4,0,1); best_wGG4/=best_wGG4.sum()
for k,w in zip(KEYS,best_wGG4): print(f"    {k}: {w:.4f}")
bo_GG4=np.clip(oofs_all@best_wGG4,0,None); bt_GG4=np.clip(tes_all@best_wGG4,0,None)
print(f"  Huber-optimized MAE: {mae(bo_GG4):.5f}")
bo_GG4g, bt_GG4g, bm_GG4 = std_gate(bo_GG4, bt_GG4, 'GG4')
save_sub(bt_GG4g, bm_GG4, 'GG4_huber_opt')

# ── GG5: Time-slot conditional blending ─────────────────────────
print("\n=== GG5: Time-slot conditional weighting ===")
if 'timeslot' in train_raw.columns or 'time_slot' in train_raw.columns:
    ts_col = 'timeslot' if 'timeslot' in train_raw.columns else 'time_slot'
    ts_vals = train_raw[ts_col].values
    unique_ts = np.unique(ts_vals)
    # Per-timeslot weight optimization (simple: split early/late)
    early_mask = ts_vals <= np.median(unique_ts)
    late_mask = ~early_mask
    # Optimize weights for early slots
    def obj_early(w):
        w=np.clip(w,0,1); w/=w.sum()
        return np.mean(np.abs(np.clip(oofs_all[early_mask]@w,0,None)-y_true[early_mask]))
    def obj_late(w):
        w=np.clip(w,0,1); w/=w.sum()
        return np.mean(np.abs(np.clip(oofs_all[late_mask]@w,0,None)-y_true[late_mask]))
    best_e=999; best_we=None
    best_l=999; best_wl=None
    for _ in range(15):
        np.random.seed(_+3000)
        w0=np.random.dirichlet(np.ones(5))
        re=minimize(obj_early,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':2000})
        rl=minimize(obj_late,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':2000})
        if re.fun<best_e: best_e=re.fun; best_we=re.x
        if rl.fun<best_l: best_l=rl.fun; best_wl=rl.x
    if best_we is not None and best_wl is not None:
        best_we=np.clip(best_we,0,1); best_we/=best_we.sum()
        best_wl=np.clip(best_wl,0,1); best_wl/=best_wl.sum()
        # Apply to OOF
        bo_GG5=np.zeros(len(y_true))
        bo_GG5[early_mask]=np.clip(oofs_all[early_mask]@best_we,0,None)
        bo_GG5[late_mask]=np.clip(oofs_all[late_mask]@best_wl,0,None)
        print(f"  timeslot-conditional OOF: {mae(bo_GG5):.5f}")
        # Apply to test
        ts_test = test_raw[ts_col].values if ts_col in test_raw.columns else None
        if ts_test is not None:
            ts_median = np.median(unique_ts)
            early_te = ts_test <= ts_median
            bt_GG5=np.zeros(len(test_raw))
            bt_GG5[early_te]=np.clip(tes_all[early_te]@best_we,0,None)
            bt_GG5[~early_te]=np.clip(tes_all[~early_te]@best_wl,0,None)
            bo_GG5g, bt_GG5g, bm_GG5 = std_gate(bo_GG5, bt_GG5, 'GG5')
            save_sub(bt_GG5g, bm_GG5, 'GG5_timeslot_cond')
        else:
            print("  GG5 skipped: no timeslot in test")
    else:
        print("  GG5 skipped: optimization failed")
else:
    print("  GG5 skipped: no timeslot column")
    # Fallback: just use ref3 with gate
    bo_GG5g, bt_GG5g, bm_GG5 = std_gate(bo_ref3, bt_ref3, 'GG5_fallback')
    save_sub(bt_GG5g, bm_GG5, 'GG5_ts_fallback')

# ── GG6: Trimmed mean blend ──────────────────────────────────────
print("\n=== GG6: Trimmed mean blend (drop top/bottom 1 component) ===")
# 5 components → per sample drop highest and lowest → avg of remaining 3
stacked = np.stack([np.clip(co[k],0,None) for k in KEYS], axis=1)
stacked_t = np.stack([np.clip(ct[k],0,None) for k in KEYS], axis=1)
sorted_idx = np.argsort(stacked, axis=1)
sorted_t_idx = np.argsort(stacked_t, axis=1)
# Take middle 3
mid_idx = sorted_idx[:, 1:4]  # drop min and max
mid_t_idx = sorted_t_idx[:, 1:4]
bo_GG6 = np.array([np.mean(stacked[i, mid_idx[i]]) for i in range(len(stacked))])
bt_GG6 = np.array([np.mean(stacked_t[i, mid_t_idx[i]]) for i in range(len(stacked_t))])
print(f"  trimmed mean OOF: {mae(bo_GG6):.5f}")
bo_GG6g, bt_GG6g, bm_GG6 = std_gate(bo_GG6, bt_GG6, 'GG6')
save_sub(bt_GG6g, bm_GG6, 'GG6_trimmed_mean')

# ── GG7: Log-space combination ───────────────────────────────────
print("\n=== GG7: Log-space combination (log→blend→exp) ===")
eps = 0.01  # avoid log(0)
def obj_log(w):
    w=np.clip(w,0,1); w/=w.sum()
    log_preds = np.log(oofs_all + eps)
    log_blend = log_preds @ w
    pred = np.exp(log_blend) - eps
    pred = np.clip(pred, 0, None)
    return np.mean(np.abs(pred - y_true))
best_GG7=999; best_wGG7=None
for _ in range(25):
    np.random.seed(_+4000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_log,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_GG7: best_GG7=r.fun; best_wGG7=r.x
best_wGG7=np.clip(best_wGG7,0,1); best_wGG7/=best_wGG7.sum()
for k,w in zip(KEYS,best_wGG7): print(f"    {k}: {w:.4f}")
log_preds_oof = np.log(oofs_all+eps)
log_preds_te  = np.log(tes_all+eps)
bo_GG7 = np.clip(np.exp(log_preds_oof@best_wGG7)-eps, 0, None)
bt_GG7 = np.clip(np.exp(log_preds_te@best_wGG7)-eps, 0, None)
print(f"  log-blend OOF: {mae(bo_GG7):.5f}")
bo_GG7g, bt_GG7g, bm_GG7 = std_gate(bo_GG7, bt_GG7, 'GG7')
save_sub(bt_GG7g, bm_GG7, 'GG7_logspace_blend')

# ── GG8: ref3 w/ outlier-robust gate (예측 분산 기반) ───────────
print("\n=== GG8: Outlier-robust gate (apply gate only to high-variance samples) ===")
# 5 컴포넌트 간 예측 분산이 높은 샘플 → gate 더 강하게
pred_std_oof = np.std(oofs_all, axis=1)  # per-sample std across components
high_var = pred_std_oof > np.percentile(pred_std_oof, 70)  # 상위 30% high variance
low_var  = ~high_var
print(f"  high_var samples: {high_var.sum()} ({high_var.mean()*100:.1f}%)")

# Gate: high-var 샘플에는 게이트 강하게 (w up), low-var은 약하게
best_GG8=mae(bo_ref3); bo_GG8=bo_ref3; bt_GG8=bt_ref3; cfg_GG8='no_gate'
for p1 in np.arange(0.06,0.18,0.02):
    m1_full=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
    for w1_high in np.arange(0.020,0.080,0.010):
        for w1_low in np.arange(0.005,0.030,0.005):
            w1_arr = np.where(high_var, w1_high, w1_low)
            b1 = (1-m1_full*w1_arr)*bo_ref3 + m1_full*w1_arr*rh_oof
            # Test: use avg of w1_high and w1_low
            w1_avg = 0.6*w1_high + 0.4*w1_low
            b1t = (1-m1t*w1_avg)*bt_ref3 + m1t*w1_avg*rh_te
            for p2 in np.arange(0.18,0.42,0.04):
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.010,0.060,0.010):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best_GG8:
                        best_GG8=mm
                        cfg_GG8=f'p{p1:.2f}+p{p2:.2f}+w1h{w1_high:.2f}l{w1_low:.2f}'
                        bo_GG8=b2; bt_GG8=(1-m2t*w2)*b1t+m2t*w2*rm_te
print(f"  [GG8] var-robust gate={cfg_GG8}  OOF={best_GG8:.5f}")
save_sub(bt_GG8, best_GG8, 'GG8_varrobust_gate')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("GG ROUND SUMMARY (layout generalization & robust blending)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
print("\n⚠️ GG1/GG2: bootstrap/holdout layout 기반 → OOF보다 LB가 더 중요한 지표")
