"""
v32 quick test: v31 + M/M/1 physics features + test-layout distribution features
Fold 0만 빠르게 검증: corr(oracle_NEW) < 0.95 여부 확인
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir("C:/Users/user/Desktop/데이콘 4월")

# ── 1. oracle_NEW OOF 재구성 ────────────────────────────────
print("[1] oracle_NEW 재구성...")
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
sample    = pd.read_csv('sample_submission.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['sc_num'] = train_raw['scenario_id'].str.replace('SC_','').astype(int)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = np.array([ls_pos[i] for i in train_raw['ID'].values])

with open('results/mega33_final.pkl','rb') as f: d33=pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34=pickle.load(f)
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))
print(f"  oracle_NEW OOF: {oracle_mae:.4f}")

# ── 2. v31 피처 로드 ─────────────────────────────────────────
print("[2] v31 피처 로드...")
with open('results/eda_v31/v31_fe_cache.pkl','rb') as f: cache = pickle.load(f)
feat_cols = cache['feat_cols']
tr_fe = cache['train_fe']  # layout_id, scenario_id sorted
te_fe = cache['test_fe']

te_cols = [c for c in feat_cols if c in te_fe.columns]
X_base_tr_ls = tr_fe[feat_cols].values.astype(np.float32)  # ls order
X_base_te_ls = te_fe[te_cols].values.astype(np.float32)
X_base_tr = X_base_tr_ls[id2]  # rid order
print(f"  v31 피처: {X_base_tr.shape[1]}개")

# test ls order
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in sample['ID'].values])
X_base_te = X_base_te_ls[te_rid_to_ls]  # sample order

# ── 3. M/M/1 nonlinear features ──────────────────────────────
print("[3] M/M/1 physics 피처 계산...")
eps = 0.01

def add_mm1_feats(df):
    pu = df['pack_utilization'].values.astype(float)
    inflow = df['order_inflow_15m'].values.astype(float)
    pu_med = float(np.nanmedian(pu[~np.isnan(pu) & (pu > 0.01)]))
    pu_f = np.where(np.isnan(pu), pu_med, pu)
    mm1 = pu_f / (1 - pu_f + eps)
    sc_ids = df['scenario_id'].values
    sc_mean_mm1 = pd.Series(mm1, index=df.index).groupby(sc_ids).transform('mean').values
    sc_max_mm1  = pd.Series(mm1, index=df.index).groupby(sc_ids).transform('max').values
    inflow_f = np.where(np.isnan(inflow), np.nanmedian(inflow), inflow)
    inflow_x_mm1 = np.log1p(inflow_f) * mm1
    # lag within scenario (sorted order)
    mm1_lag1 = pd.Series(mm1, index=df.index).groupby(sc_ids).shift(1).fillna(mm1.mean()).values
    mm1_diff1 = mm1 - mm1_lag1
    return np.column_stack([mm1, sc_mean_mm1, sc_max_mm1, inflow_x_mm1, mm1_lag1, mm1_diff1])

mm1_tr_ls = add_mm1_feats(train_ls)
mm1_te_ls = add_mm1_feats(test_ls)
mm1_tr = mm1_tr_ls[id2]
mm1_te = mm1_te_ls[te_rid_to_ls]
print(f"  M/M/1 피처: {mm1_tr.shape[1]}개")

# ── 4. test-layout distribution features ─────────────────────
print("[4] test-layout 분포 피처 계산...")
# test 레이아웃별 평균 (순수 input feature 기반, 타겟 없음)
test_ly_stats = test_raw.groupby('layout_id').agg(
    test_ly_inflow_mean=('order_inflow_15m','mean'),
    test_ly_pu_mean=('pack_utilization','mean'),
    test_ly_congestion_mean=('congestion_score','mean'),
).reset_index()
global_te_inflow = test_raw['order_inflow_15m'].mean()
global_te_pu = float(np.nanmean(test_raw['pack_utilization'].values))
global_te_cong = test_raw['congestion_score'].mean()

# train 레이아웃별 inflow 평균 (shift ratio 계산용)
train_ly_inflow_mean = train_raw.groupby('layout_id')['order_inflow_15m'].mean().reset_index()
train_ly_inflow_mean.columns = ['layout_id','train_ly_inflow_mean']

def make_layout_feats(df, is_train=True):
    df2 = df[['ID','layout_id']].copy()
    df2 = df2.merge(test_ly_stats, on='layout_id', how='left')
    df2 = df2.merge(train_ly_inflow_mean, on='layout_id', how='left')
    # unseen layout → global mean
    df2['test_ly_inflow_mean'] = df2['test_ly_inflow_mean'].fillna(global_te_inflow)
    df2['test_ly_pu_mean'] = df2['test_ly_pu_mean'].fillna(global_te_pu)
    df2['test_ly_congestion_mean'] = df2['test_ly_congestion_mean'].fillna(global_te_cong)
    df2['train_ly_inflow_mean'] = df2['train_ly_inflow_mean'].fillna(
        train_ly_inflow_mean['train_ly_inflow_mean'].mean())
    df2['layout_inflow_shift_ratio'] = (df2['test_ly_inflow_mean'] /
                                         (df2['train_ly_inflow_mean'] + 0.1))
    cols = ['test_ly_inflow_mean','test_ly_pu_mean','test_ly_congestion_mean',
            'layout_inflow_shift_ratio']
    return df2[cols].values.astype(np.float32)

# train_raw (rid order) 기준
tl_tr = make_layout_feats(train_raw, is_train=True)
# test sample order
test_sample_df = test_raw.set_index('ID').reindex(sample['ID']).reset_index()
tl_te = make_layout_feats(test_sample_df, is_train=False)
print(f"  test-layout 피처: {tl_tr.shape[1]}개")
print(f"  test_ly_inflow (seen): {test_ly_stats[test_ly_stats['layout_id'].isin(train_raw['layout_id'].unique())]['test_ly_inflow_mean'].mean():.1f}")
print(f"  test_ly_inflow (unseen): {test_ly_stats[~test_ly_stats['layout_id'].isin(train_raw['layout_id'].unique())]['test_ly_inflow_mean'].mean():.1f}")

# ── 5. 조립 ───────────────────────────────────────────────────
X_tr = np.hstack([X_base_tr, mm1_tr, tl_tr])
X_te = np.hstack([X_base_te, mm1_te, tl_te])
print(f"\n[5] 최종 피처: {X_tr.shape[1]}개 (v31={X_base_tr.shape[1]} + mm1={mm1_tr.shape[1]} + test_layout={tl_tr.shape[1]})")

# ── 6. Fold 0 빠른 검증 (temporal CV) ──────────────────────────
print("\n[6] Fold 0 빠른 검증 (sc 8001-10000)...")
sc_num_arr = train_raw['sc_num'].values
val_mask = (sc_num_arr >= 8001) & (sc_num_arr <= 10000)
tr_idx, val_idx = np.where(~val_mask)[0], np.where(val_mask)[0]
print(f"  train={len(tr_idx):,}  val={len(val_idx):,}")

model = lgb.LGBMRegressor(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=256, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    n_jobs=-1, verbose=-1, random_state=42
)
model.fit(
    X_tr[tr_idx], y_true[tr_idx],
    eval_set=[(X_tr[val_idx], y_true[val_idx])],
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
)
val_pred = model.predict(X_tr[val_idx])
fold0_mae = np.mean(np.abs(y_true[val_idx] - val_pred))
fold0_corr = np.corrcoef(fw4_oo[val_idx], val_pred)[0,1]
print(f"  Fold0 MAE={fold0_mae:.4f}  corr(oracle_NEW)={fold0_corr:.4f}")
print(f"  best_iteration={model.best_iteration_}")
print(f"  (temporal_oracle fold0 LGB was: 9.4295, corr unknown)")

# 상위 피처 중요도 (새 피처)
new_feat_names = (
    [f'v31_{i}' for i in range(X_base_tr.shape[1])] +
    ['mm1','mm1_sc_mean','mm1_sc_max','inflow_x_mm1','mm1_lag1','mm1_diff1'] +
    ['test_ly_inflow','test_ly_pu','test_ly_congestion','layout_inflow_shift']
)
imp = model.feature_importances_
new_feat_start = X_base_tr.shape[1]
print(f"\n  M/M/1 피처 중요도:")
for i, name in enumerate(['mm1','mm1_sc_mean','mm1_sc_max','inflow_x_mm1','mm1_lag1','mm1_diff1']):
    print(f"    {name}: {imp[new_feat_start+i]:,}")
print(f"  test-layout 피처 중요도:")
for i, name in enumerate(['test_ly_inflow','test_ly_pu','test_ly_congestion','layout_inflow_shift']):
    print(f"    {name}: {imp[new_feat_start+6+i]:,}")

# ── 7. 판단 ───────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"판단 기준: corr(oracle_NEW) < 0.95 AND MAE 적절")
print(f"Fold0: MAE={fold0_mae:.4f}  corr={fold0_corr:.4f}")
if fold0_corr < 0.95 and fold0_mae < 9.5:
    print("==> GO: 전체 5-fold 훈련 진행")
elif fold0_corr < 0.95:
    print("==> MARGINAL: corr은 낮지만 MAE가 너무 높음")
else:
    print("==> STOP: corr >= 0.95 — 블렌드 불가능")
print(f"{'='*60}")
print("Done.")
