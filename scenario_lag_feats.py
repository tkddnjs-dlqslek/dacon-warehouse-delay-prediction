"""
Scenario-lag features within layout.
- 같은 layout_id 내에서 scenario_id 순으로 정렬 후
- 이전 K개 scenario의 avg_delay 통계를 feature로 추가
- 아이디어: 시뮬레이션이 sequential하면 이전 시나리오가 다음 상태 예측에 도움

테스트 시: oracle_NEW test predictions를 pseudo-label로 사용
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
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

# ── v30 피처 로드 (ls order) ──────────────────────────────────────
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f: blob=pickle.load(f)
train_fe=blob['train_fe']; feat_cols=list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f: test_fe=pickle.load(f)
fold_ids=np.load('results/eda_v30/fold_idx.npy')
y=train_fe['avg_delay_minutes_next_30m'].values; ylog=np.log1p(y)

# 필수 컬럼 확인
req_cols = ['layout_id', 'scenario_id']
has_req = all(c in train_fe.columns for c in req_cols)
print(f"Required cols present: {has_req} — {[c for c in req_cols if c in train_fe.columns]}")

if not has_req:
    print("layout_id or scenario_id not in train_fe — aborting scenario_lag approach")
    import sys; sys.exit(0)

# ── Scenario-lag feature 생성 ─────────────────────────────────────
# ls order: 이미 layout_id, scenario_id 기준 정렬됨
# train_fe 기준으로 각 row 위치에서 이전 K scenario의 avg_delay stats
K = 5  # 이전 5개 scenario 참고

def build_scenario_lag(fe, y_vals, te_pseudo=None):
    """
    같은 layout_id 내에서 scenario_id 순서로 정렬 후
    이전 K scenario의 평균 delay를 feature로 추가.
    te_pseudo: test set에 적용할 oracle 예측값 (pseudo label)
    """
    if 'layout_id' not in fe.columns or 'scenario_id' not in fe.columns:
        return pd.DataFrame(index=fe.index)

    n = len(fe)
    lag_mean  = np.full(n, np.nan)
    lag_max   = np.full(n, np.nan)
    lag_std   = np.full(n, np.nan)
    lag_trend = np.full(n, np.nan)  # 선형 트렌드 (마지막 - 첫 번째) / K

    # 이미 ls 순서(layout_id, scenario_id)로 정렬됨
    lid_arr  = fe['layout_id'].values
    sid_arr  = fe['scenario_id'].values
    y_arr    = y_vals if y_vals is not None else np.zeros(n)

    # layout별로 처리
    for lid in np.unique(lid_arr):
        mask = lid_arr == lid
        idxs = np.where(mask)[0]  # ls order 내 indices
        # scenario_id로 정렬 (ls order 이미 sorted이므로 추가 정렬 불필요)
        # idxs는 이미 scenario_id 순으로 정렬됨
        y_lay = y_arr[idxs]
        for i, idx in enumerate(idxs):
            if i < 1:
                continue  # 첫 scenario는 이전 정보 없음
            start = max(0, i - K)
            past_y = y_lay[start:i]
            lag_mean[idx]  = past_y.mean()
            lag_max[idx]   = past_y.max()
            lag_std[idx]   = past_y.std() if len(past_y) > 1 else 0.0
            if len(past_y) >= 2:
                lag_trend[idx] = (past_y[-1] - past_y[0]) / len(past_y)
            else:
                lag_trend[idx] = 0.0

    # NaN → 해당 layout 전체 평균으로 채우기
    for lid in np.unique(lid_arr):
        mask = lid_arr == lid
        idxs = np.where(mask)[0]
        valid = ~np.isnan(lag_mean[idxs])
        if valid.any():
            fill_val = lag_mean[idxs[valid]].mean()
            lag_mean[idxs[~valid]] = fill_val
            lag_max[idxs[~valid]]  = lag_max[idxs[valid]].mean()
            lag_std[idxs[~valid]]  = 0.0
            lag_trend[idxs[~valid]]= 0.0
        else:
            lag_mean[idxs] = np.nanmean(y_arr)
            lag_max[idxs]  = np.nanmean(y_arr)
            lag_std[idxs]  = 0.0
            lag_trend[idxs]= 0.0

    return pd.DataFrame({
        'scen_lag_mean':  lag_mean,
        'scen_lag_max':   lag_max,
        'scen_lag_std':   lag_std,
        'scen_lag_trend': lag_trend,
    }, index=fe.index, dtype=np.float32)

# 훈련용 lag (실제 타겟 사용)
lag_tr = build_scenario_lag(train_fe, y)
print(f"Lag feature stats (train):")
for c in lag_tr.columns:
    print(f"  {c}: mean={lag_tr[c].mean():.3f}  nan_frac={lag_tr[c].isna().mean():.3f}")

# 테스트용 lag (oracle_NEW 예측을 pseudo label로 사용)
# test_fe도 ls order로 정렬됨
te_pseudo_ls = oracle_new_t[te_rid_to_ls]  # te ls order로 재정렬
lag_te = build_scenario_lag(test_fe, te_pseudo_ls)

# feature 행렬 구성
X_base    = train_fe[feat_cols].values.astype(np.float32)
X_te_base = test_fe[feat_cols].values.astype(np.float32)
X_lag     = np.hstack([X_base,    lag_tr.values.astype(np.float32)])
X_te_lag  = np.hstack([X_te_base, lag_te.values.astype(np.float32)])
print(f"\nFeature shape: {X_lag.shape} (+{lag_tr.shape[1]} lag feats)")

# ── 2-fold quick check ────────────────────────────────────────────
PARAMS = dict(objective='huber', alpha=0.9, n_estimators=2000, learning_rate=0.03,
              num_leaves=63, max_depth=8, min_child_samples=50,
              subsample=0.7, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42)

print("\n=== 2-fold Quick Comparison ===")
def run2fold(X, label):
    oof=np.zeros(len(y)); used=np.zeros(len(y),dtype=bool)
    for f in [0,4]:
        vm=fold_ids==f; tm=~vm
        m=lgb.LGBMRegressor(**PARAMS)
        m.fit(X[tm], ylog[tm], eval_set=[(X[vm], ylog[vm])],
              callbacks=[lgb.early_stopping(80,verbose=False), lgb.log_evaluation(0)])
        oof[vm]=np.clip(np.expm1(m.predict(X[vm])),0,None)
        used[vm]=True
    rid=oof[id2]; mask=used[id2]
    solo=np.mean(np.abs(y_true[mask]-rid[mask]))
    corr=np.corrcoef(fw4_oo[mask], rid[mask])[0,1]
    print(f"  {label:<35} OOF(2fold)={solo:.4f}  corr={corr:.4f}")
    return solo, corr

s0,c0 = run2fold(X_base, "v30 baseline")
s1,c1 = run2fold(X_lag,  "v30 + scenario_lag(K=5)")
print(f"  delta OOF: {s1-s0:+.4f}  delta corr: {c1-c0:+.4f}")

# ── Full 5-fold ───────────────────────────────────────────────────
print("\n=== Full 5-fold: v30 + scenario_lag ===")
oof_full = np.zeros(len(y))
test_full = np.zeros(len(X_te_lag))
for f in range(5):
    vm=fold_ids==f; tm=~vm
    m=lgb.LGBMRegressor(**PARAMS)
    m.fit(X_lag[tm], ylog[tm], eval_set=[(X_lag[vm], ylog[vm])],
          callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(0)])
    oof_full[vm]=np.clip(np.expm1(m.predict(X_lag[vm])),0,None)
    test_full+=np.clip(np.expm1(m.predict(X_te_lag)),0,None)/5
    print(f"  fold {f}: MAE={np.mean(np.abs(y[vm]-oof_full[vm])):.4f}  it={m.best_iteration_}")

oof_full=np.clip(oof_full,0,None)
oof_rid=oof_full[id2]; test_rid=test_full[te_rid_to_ls]
solo_f=np.mean(np.abs(y_true-oof_rid)); corr_f=np.corrcoef(fw4_oo,oof_rid)[0,1]
print(f"\nLag-LGB OOF (rid): {solo_f:.4f}  corr={corr_f:.4f}  (oracle={oracle_mae:.4f})")
print(f"test_unseen: {test_rid[unseen_mask].mean():.3f}  (oracle: {oracle_unseen:.3f})")

# ── Blend ─────────────────────────────────────────────────────────
best_w, best_bl = 0, oracle_mae
for w in np.arange(0.01, 0.31, 0.01):
    bl=np.clip((1-w)*fw4_oo+w*oof_rid,0,None)
    mv=np.mean(np.abs(y_true-bl))
    if mv<best_bl: best_bl,best_w=mv,w
gain=oracle_mae-best_bl
print(f"Best blend: w={best_w:.2f}  blend_OOF={best_bl:.4f}  gain={gain:+.4f}")

if gain > 0.0003:
    bl_t=np.clip((1-best_w)*oracle_new_t+best_w*test_rid,0,None)
    sub=pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m']=bl_t
    fname=f"FINAL_scenlag_OOF{best_bl:.4f}.csv"
    sub.to_csv(fname,index=False)
    np.save('results/scenlag_oof.npy', oof_rid.astype(np.float32))
    np.save('results/scenlag_test.npy', test_rid.astype(np.float32))
    print(f"*** SAVED: {fname}  unseen={bl_t[unseen_mask].mean():.3f} ***")
else:
    print("No blend improvement. No submission.")

print("\nDone.")
