"""
Physics-based weak learner: M/M/1 queueing theory calibrated model
- delay ∝ pu / (1 - pu)^β  (M/M/1 generalization)
- 파라미터 C, β, eps를 training data에서 fitting
- corr이 낮고 unseen layout 외삽이 올바른 방향이면 blend 효과

검증 전략:
  1. 단일 파라미터 fitting (least-squares)
  2. OOF MAE + corr vs oracle_NEW 계산
  3. blend 개선 여부 확인
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize_scalar, minimize
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir("C:/Users/user/Desktop/데이콘 4월")

# ── oracle_NEW 재구성 ──────────────────────────────────────────────
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = np.array([ls_pos[i] for i in train_raw['ID'].values])

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
id_order = test_raw['ID'].values
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

with open('results/mega33_final.pkl','rb') as f: d33=pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34=pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
oracle_mae=np.mean(np.abs(y_true-fw4_oo))
oracle_new_df=pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df=oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t=oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen=oracle_new_t[unseen_mask].mean()
print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

# ── 원시 데이터에서 physics 변수 가져오기 (v30 cache에는 NaN — raw CSV 사용) ─
fold_ids=np.load('results/eda_v30/fold_idx.npy')

PHYS_COLS = ['pack_utilization', 'order_inflow_15m', 'conveyor_speed_mps']
# ls order: train_ls는 이미 layout_id, scenario_id 기준 정렬됨
y_ls = train_ls['avg_delay_minutes_next_30m'].values

_pu_raw = train_ls['pack_utilization'].values.astype(float)
_pu_med = float(np.nanmedian(_pu_raw))
pu_ls   = np.where(np.isnan(_pu_raw), _pu_med, _pu_raw)
_inf_raw = train_ls['order_inflow_15m'].values.astype(float)
inf_ls  = np.where(np.isnan(_inf_raw), np.nanmedian(_inf_raw), _inf_raw)
_conv_raw = train_ls['conveyor_speed_mps'].values.astype(float)
conv_ls = np.where(np.isnan(_conv_raw), np.nanmedian(_conv_raw), _conv_raw)

pu_rid   = pu_ls[id2]
inf_rid  = inf_ls[id2]
conv_rid = conv_ls[id2]

# test physics features (ls order)
test_ls_phys = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
_pu_te = test_ls_phys['pack_utilization'].values.astype(float)
pu_te_ls   = np.where(np.isnan(_pu_te), _pu_med, _pu_te)
_inf_te = test_ls_phys['order_inflow_15m'].values.astype(float)
inf_te_ls  = np.where(np.isnan(_inf_te), np.nanmedian(_inf_raw), _inf_te)
_conv_te = test_ls_phys['conveyor_speed_mps'].values.astype(float)
conv_te_ls = np.where(np.isnan(_conv_te), np.nanmedian(_conv_raw), _conv_te)

pu_te_rid   = pu_te_ls[te_rid_to_ls]
inf_te_rid  = inf_te_ls[te_rid_to_ls]
conv_te_rid = conv_te_ls[te_rid_to_ls]

print(f"train pu: mean={pu_rid.mean():.3f}  max={pu_rid.max():.3f}")
print(f"test  pu: mean={pu_te_rid.mean():.3f}  max={pu_te_rid.max():.3f}  (unseen max={pu_te_rid[unseen_mask].max():.3f})")

# ── M/M/1 generalized 모델 ────────────────────────────────────────
# f(pu; C, beta, eps) = C * pu^beta / (1 - pu + eps)^beta
# 또는 더 단순하게: C * pu / (1 - pu + eps) + intercept

def mm1_pred(pu, C, beta, eps, intercept):
    """Generalized M/M/1: delay = C * pu^beta / (1-pu+eps)^beta + intercept"""
    ratio = pu / (1.0 - pu + eps)
    return np.clip(C * ratio**beta + intercept, 0, None)

def mm1_multifeat_pred(pu, inf, conv, params):
    """
    더 풍부한 모델:
    delay = C1 * pu/(1-pu+eps) + C2 * inf/conv + intercept
    """
    C1, C2, beta, eps, intercept = params
    mm1   = pu / (1.0 - pu + eps)
    sojourn = inf / (conv + 0.01)
    return np.clip(C1 * mm1**beta + C2 * sojourn + intercept, 0, None)

# ── GroupKFold CV로 파라미터 피팅 ──────────────────────────────────
print("\n=== M/M/1 Physics Model: CV Fitting ===")

# 1. 단순 버전: delay = C * pu/(1-pu+eps)
def fit_simple(pu_tr, y_tr):
    """L1 loss로 파라미터 피팅"""
    from scipy.optimize import minimize
    def loss(params):
        C, beta, eps, intercept = params
        pred = mm1_pred(pu_tr, C, beta, np.clip(eps, 1e-6, 0.5), intercept)
        return np.mean(np.abs(y_tr - pred))
    x0 = [5.0, 1.0, 0.01, 0.0]
    bounds = [(0.1, 200), (0.1, 5.0), (1e-6, 0.5), (-5, 50)]
    res = minimize(loss, x0, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500})
    return res.x

# ls order로 작업
pu_ls_arr  = pu_ls
inf_ls_arr = inf_ls if inf_ls is not None else np.ones(len(pu_ls))
conv_ls_arr= conv_ls if conv_ls is not None else np.ones(len(pu_ls))
y_ls_arr   = y_ls

oof_mm1 = np.zeros(len(y_ls_arr))

for f in range(5):
    vm = fold_ids == f; tm = ~vm
    # training data에서 파라미터 fitting
    params_f = fit_simple(pu_ls_arr[tm], y_ls_arr[tm])
    C, beta, eps, intercept = params_f
    # validation 예측
    oof_mm1[vm] = mm1_pred(pu_ls_arr[vm], C, beta, max(eps, 1e-6), intercept)
    val_mae = np.mean(np.abs(y_ls_arr[vm] - oof_mm1[vm]))
    print(f"  fold {f}: C={C:.3f}  beta={beta:.3f}  eps={eps:.4f}  intercept={intercept:.3f}  MAE_vl={val_mae:.3f}")

# 전체 데이터로 파라미터 재피팅 (test 예측용)
params_full = fit_simple(pu_ls_arr, y_ls_arr)
C_f, beta_f, eps_f, int_f = params_full

# test 예측
test_mm1 = mm1_pred(pu_te_rid, C_f, beta_f, max(eps_f, 1e-6), int_f)

# oof를 rid order로
oof_mm1_rid = oof_mm1[id2]

solo = np.mean(np.abs(y_true - oof_mm1_rid))
corr = np.corrcoef(fw4_oo, oof_mm1_rid)[0,1]
print(f"\nMM1 Physics: OOF={solo:.4f}  corr={corr:.4f}  (oracle={oracle_mae:.4f})")
print(f"test_unseen: {test_mm1[unseen_mask].mean():.3f}  test_seen: {test_mm1[~unseen_mask].mean():.3f}")

# ── 더 풍부한 모델: pu + inflow/conv 결합 ──────────────────────────
print("\n=== Multi-feature Physics Model ===")
def fit_multi(pu_tr, inf_tr, conv_tr, y_tr):
    from scipy.optimize import minimize
    def loss(params):
        C1, C2, beta, eps, intercept = params
        eps_ = np.clip(eps, 1e-6, 0.5)
        pred = mm1_multifeat_pred(pu_tr, inf_tr, conv_tr, [C1, C2, beta, eps_, intercept])
        return np.mean(np.abs(y_tr - pred))
    x0 = [5.0, 1.0, 1.0, 0.01, 0.0]
    bounds = [(0.01, 200), (-50, 200), (0.1, 5.0), (1e-6, 0.5), (-5, 50)]
    res = minimize(loss, x0, bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500})
    return res.x

oof_mm1m = np.zeros(len(y_ls_arr))
for f in range(5):
    vm = fold_ids == f; tm = ~vm
    params_m = fit_multi(pu_ls_arr[tm], inf_ls_arr[tm], conv_ls_arr[tm], y_ls_arr[tm])
    C1, C2, beta, eps, intercept = params_m
    oof_mm1m[vm] = mm1_multifeat_pred(pu_ls_arr[vm], inf_ls_arr[vm], conv_ls_arr[vm],
                                       [C1, C2, beta, max(eps, 1e-6), intercept])
    fold_mae = np.mean(np.abs(y_ls_arr[vm] - oof_mm1m[vm]))
    print(f"  fold {f}: C1={C1:.3f}  C2={C2:.3f}  beta={beta:.3f}  MAE={fold_mae:.4f}")

# full fit for test
params_mf = fit_multi(pu_ls_arr, inf_ls_arr, conv_ls_arr, y_ls_arr)
test_mm1m = mm1_multifeat_pred(pu_te_rid, inf_te_rid, conv_te_rid, params_mf)

oof_mm1m_rid = oof_mm1m[id2]
solo_m = np.mean(np.abs(y_true - oof_mm1m_rid))
corr_m = np.corrcoef(fw4_oo, oof_mm1m_rid)[0,1]
print(f"\nMulti-feat Physics: OOF={solo_m:.4f}  corr={corr_m:.4f}")
print(f"test_unseen: {test_mm1m[unseen_mask].mean():.3f}  test_seen: {test_mm1m[~unseen_mask].mean():.3f}")

# ── Blend 분석 ───────────────────────────────────────────────────
print("\n=== Blend Analysis ===")
for label, oof_r, test_r in [("MM1 simple", oof_mm1_rid, test_mm1),
                                ("MM1 multi",  oof_mm1m_rid, test_mm1m)]:
    best_w, best_bl = 0, oracle_mae
    for w in np.arange(0.01, 0.21, 0.01):
        bl=np.clip((1-w)*fw4_oo+w*oof_r, 0, None)
        mv=np.mean(np.abs(y_true-bl))
        if mv < best_bl: best_bl, best_w = mv, w
    gain = oracle_mae - best_bl
    print(f"  {label}: best_w={best_w:.2f}  blend_OOF={best_bl:.4f}  gain={gain:+.4f}")
    if gain > 0.0001:
        bl_t = np.clip((1-best_w)*oracle_new_t + best_w*test_r, 0, None)
        sub = pd.read_csv('sample_submission.csv')
        sub['avg_delay_minutes_next_30m'] = bl_t
        fname = f"FINAL_physics_{label.replace(' ','_')}_OOF{best_bl:.4f}.csv"
        sub.to_csv(fname, index=False)
        print(f"    *** SAVED: {fname}  unseen={bl_t[unseen_mask].mean():.3f} ***")

print("\nDone.")
