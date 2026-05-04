"""
Temporal Blocked CV + Between-Scenario True Lag Oracle
------------------------------------------------------
근본 문제: GroupKFold(layout_id) → OOF는 seen layouts만 검증, LB 갭=1.38
해결책:
  1. sc_num 기반 5블록 temporal CV (fold0 val: SC_8001~10000 ≈ test 분포)
  2. between-scenario true lag y (proxy 대신 실제 y, 이전 시나리오의 y_mean)
  3. test 예측: seen layouts → 마지막 training scenario y_mean / unseen → global_mean

피처: v31(335) + sc_num_norm(1) + lag1/2/3_sc_y(3) = 339개
모델: LGB(mae, log1p target) + XGB(reg:absoluteerror, raw target)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, gc, time, pickle
os.chdir("C:/Users/user/Desktop/데이콘 4월")
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import warnings; warnings.filterwarnings('ignore')

os.makedirs('results/temporal_oracle', exist_ok=True)

print("=" * 65)
print("Temporal Oracle: Temporal CV + True Between-Scenario Lag")
print("=" * 65)

# ── 1. 데이터 로드 ───────────────────────────────────────────────────
t0 = time.time()
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_', '').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['sc_num'] = train_raw['scenario_id'].str.replace('SC_', '').astype(int)
train_raw['sc_num_norm'] = (train_raw['sc_num'] - 1) / 9999.0

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_', '').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
test_raw['sc_num'] = test_raw['scenario_id'].str.replace('SC_', '').astype(int)
test_raw['sc_num_norm'] = (test_raw['sc_num'] - 1) / 9999.0  # 1.0001 ~ 1.2001

y_true = train_raw['avg_delay_minutes_next_30m'].values.astype(np.float32)
global_mean = float(y_true.mean())
print(f"  데이터 로드: {time.time()-t0:.1f}s | train={len(train_raw)}, test={len(test_raw)}")
print(f"  sc_num train: {train_raw['sc_num'].min()}~{train_raw['sc_num'].max()}")
print(f"  sc_num test:  {test_raw['sc_num'].min()}~{test_raw['sc_num'].max()}")
print(f"  global_mean y: {global_mean:.4f}")

# ── 2. v31 피처 캐시 로드 ────────────────────────────────────────────
print("\n[2] v31 피처 캐시 로드...")
t0 = time.time()
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
feat_cols = fe_v31['feat_cols']  # 335개
print(f"  feat_cols: {len(feat_cols)}개")

# train: ID 기준 raw order 재정렬
tr_fe = fe_v31['train_fe']
tr_id2idx = {v: i for i, v in enumerate(tr_fe['ID'].values)}
tr_ord = np.array([tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr = tr_fe[feat_cols].values[tr_ord].astype(np.float32)
del tr_fe, tr_id2idx, tr_ord; gc.collect()

# test: ID 기준 raw order 재정렬
te_fe = fe_v31['test_fe']
te_id2idx = {v: i for i, v in enumerate(te_fe['ID'].values)}
te_ord = np.array([te_id2idx[i] for i in test_raw['ID'].values])
# test_fe에 없는 컬럼은 0 fill
te_fe_aligned = te_fe.reindex(columns=feat_cols, fill_value=0.0)
X_base_te = te_fe_aligned.values[te_ord].astype(np.float32)
del te_fe, te_id2idx, te_ord, te_fe_aligned, fe_v31; gc.collect()
print(f"  피처 행렬: train={X_base_tr.shape}, test={X_base_te.shape}  ({time.time()-t0:.1f}s)")

# ── 3. Between-Scenario Lag y 계산 ─────────────────────────────────
print("\n[3] Between-Scenario Lag y 계산...")
t0 = time.time()

# 시나리오 수준 y_mean 및 sc_num
sc_level = (train_raw
    .groupby(['layout_id', 'scenario_id'], sort=False)
    .agg(sc_num=('sc_num', 'first'),
         y_mean=('avg_delay_minutes_next_30m', 'mean'))
    .reset_index()
    .sort_values(['layout_id', 'sc_num'])
    .reset_index(drop=True))

# layout 내 이전 시나리오 y_mean (lag)
for k in [1, 2, 3]:
    sc_level[f'lag{k}_sc_y'] = (sc_level
        .groupby('layout_id')['y_mean']
        .shift(k)
        .fillna(global_mean))

print(f"  sc_level 크기: {len(sc_level)} scenarios")
corr1 = sc_level[['y_mean','lag1_sc_y']].corr().iloc[0,1]
print(f"  corr(y_mean, lag1_sc_y): {corr1:.4f}")

# train_raw에 lag 피처 병합
lag_cols = ['lag1_sc_y', 'lag2_sc_y', 'lag3_sc_y']
train_raw = train_raw.merge(
    sc_level[['layout_id', 'scenario_id'] + lag_cols],
    on=['layout_id', 'scenario_id'], how='left'
)
assert train_raw[lag_cols].isna().sum().sum() == 0, "lag NaN 존재!"
print(f"  lag 계산 완료  ({time.time()-t0:.1f}s)")

# ── 4. Test Lag 계산 ─────────────────────────────────────────────────
print("\n[4] Test Lag 계산...")
train_layout_set = set(train_raw['layout_id'].unique())
test_raw['layout_seen'] = test_raw['layout_id'].isin(train_layout_set)
print(f"  seen: {test_raw['layout_seen'].sum()} rows | unseen: {(~test_raw['layout_seen']).sum()} rows")

# 각 layout의 마지막 3개 시나리오 y_mean
last_sc = (sc_level.sort_values('sc_num')
           .groupby('layout_id')['y_mean']
           .apply(list)
           .apply(lambda lst: lst[-3:] if len(lst) >= 3 else
                              [lst[-1]] * (3 - len(lst)) + lst))
# lag1 = 가장 마지막, lag2 = 두번째 마지막, lag3 = 세번째 마지막
last_sc_lag1 = last_sc.apply(lambda x: x[-1])
last_sc_lag2 = last_sc.apply(lambda x: x[-2])
last_sc_lag3 = last_sc.apply(lambda x: x[-3])

test_raw['lag1_sc_y'] = test_raw['layout_id'].map(last_sc_lag1).fillna(global_mean).astype(np.float32)
test_raw['lag2_sc_y'] = test_raw['layout_id'].map(last_sc_lag2).fillna(global_mean).astype(np.float32)
test_raw['lag3_sc_y'] = test_raw['layout_id'].map(last_sc_lag3).fillna(global_mean).astype(np.float32)

print(f"  test lag1 mean (seen):   {test_raw[test_raw['layout_seen']]['lag1_sc_y'].mean():.4f}")
print(f"  test lag1 mean (unseen): {test_raw[~test_raw['layout_seen']]['lag1_sc_y'].mean():.4f}")

# ── 5. 최종 피처 행렬 조립 (339개) ──────────────────────────────────
print("\n[5] 피처 조립...")
sc_norm_tr = train_raw['sc_num_norm'].values.reshape(-1, 1).astype(np.float32)
lag_tr     = train_raw[lag_cols].values.astype(np.float32)
X_tr_full  = np.hstack([X_base_tr, sc_norm_tr, lag_tr])  # (250000, 339)
del X_base_tr, sc_norm_tr, lag_tr; gc.collect()

sc_norm_te = test_raw['sc_num_norm'].values.reshape(-1, 1).astype(np.float32)
lag_te     = test_raw[lag_cols].values.astype(np.float32)
X_te_full  = np.hstack([X_base_te, sc_norm_te, lag_te])  # (50000, 339)
del X_base_te, sc_norm_te, lag_te; gc.collect()

print(f"  X_tr_full: {X_tr_full.shape}  X_te_full: {X_te_full.shape}")
feat_names = feat_cols + ['sc_num_norm'] + lag_cols

# ── 6. oracle_NEW OOF 재구성 (비교용) ───────────────────────────────
print("\n[6] oracle_NEW OOF 재구성...")
train_ls = pd.read_csv('train.csv').sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']: i for i, row in train_ls.iterrows()}
id2 = np.array([ls_pos[i] for i in train_raw['ID'].values])

with open('results/mega33_final.pkl', 'rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl', 'rb') as f: d34 = pickle.load(f)
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o  = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o  = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1 = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2 = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3 = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34 = 0.25; dr2 = -0.04; dr3 = -0.02; wf = 0.72; w_cb = 0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1 + w2_*r2 + w3_*r3
w2 = 1-wf; wxgb=0.12*w2/0.36; wlv2=0.16*w2/0.36; wrem=0.08*w2/0.36
bb = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
bb = np.clip((1-w_cb)*bb + w_cb*cb_oof, 0, None)
fw4_oo = np.clip(0.74*bb + 0.08*slh + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))
print(f"  oracle_NEW OOF MAE: {oracle_mae:.4f}")
del d33, d34, mega33_oof, mega34_oof, cb_oof, rank_oof, r1, r2, r3, slh
del xgb_o, lv2_o, rem_o, xgbc_o, mono_o, bb, fx_o; gc.collect()

# ── 7. Temporal Blocked CV ───────────────────────────────────────────
print("\n[7] Temporal Blocked CV 학습 (5 folds × 2 models)...")
BLOCKS = [(8001, 10000), (6001, 8000), (4001, 6000), (2001, 4000), (1, 2000)]
sc_num_arr = train_raw['sc_num'].values

LGB_PARAMS = dict(
    objective='mae', n_estimators=5000, learning_rate=0.03,
    num_leaves=256, max_depth=-1, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=6,
    random_state=42, verbose=-1,
)
XGB_PARAMS = dict(
    objective='reg:absoluteerror', n_estimators=3000,
    learning_rate=0.05, max_depth=7, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=4,
    random_state=42, verbosity=0, early_stopping_rounds=100,
    eval_metric='mae', tree_method='hist',
)

oof_lgb   = np.full(len(train_raw), np.nan, dtype=np.float32)
oof_xgb   = np.full(len(train_raw), np.nan, dtype=np.float32)
test_lgb_list = []
test_xgb_list = []

for fold_i, (lo, hi) in enumerate(BLOCKS):
    t_fold = time.time()
    val_mask = (sc_num_arr >= lo) & (sc_num_arr <= hi)
    tr_idx   = np.where(~val_mask)[0]
    val_idx  = np.where(val_mask)[0]

    X_fold_tr  = X_tr_full[tr_idx]
    X_fold_val = X_tr_full[val_idx]
    y_fold_tr  = y_true[tr_idx]
    y_fold_val = y_true[val_idx]
    n_tr, n_vl = len(tr_idx), len(val_idx)

    # ── LGB: log1p(y) ────
    lgb_model = lgb.LGBMRegressor(**LGB_PARAMS)
    lgb_model.fit(
        X_fold_tr, np.log1p(y_fold_tr),
        eval_set=[(X_fold_val, np.log1p(y_fold_val))],
        callbacks=[lgb.early_stopping(100, verbose=False),
                   lgb.log_evaluation(-1)]
    )
    lgb_val_pred = np.maximum(0.0, np.expm1(lgb_model.predict(X_fold_val))).astype(np.float32)
    lgb_te_pred  = np.maximum(0.0, np.expm1(lgb_model.predict(X_te_full))).astype(np.float32)
    oof_lgb[val_idx] = lgb_val_pred
    test_lgb_list.append(lgb_te_pred)
    lgb_mae = np.mean(np.abs(lgb_val_pred - y_fold_val))

    # ── XGB: raw y ───────
    xgb_model = xgb.XGBRegressor(**XGB_PARAMS)
    xgb_model.fit(
        X_fold_tr, y_fold_tr,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )
    xgb_val_pred = np.maximum(0.0, xgb_model.predict(X_fold_val)).astype(np.float32)
    xgb_te_pred  = np.maximum(0.0, xgb_model.predict(X_te_full)).astype(np.float32)
    oof_xgb[val_idx] = xgb_val_pred
    test_xgb_list.append(xgb_te_pred)
    xgb_mae = np.mean(np.abs(xgb_val_pred - y_fold_val))

    print(f"  Fold {fold_i} (sc {lo}-{hi}): n_tr={n_tr:6d} n_val={n_vl:6d} "
          f"LGB={lgb_mae:.4f}  XGB={xgb_mae:.4f}  ({time.time()-t_fold:.0f}s)")
    del lgb_model, xgb_model, X_fold_tr, X_fold_val, lgb_val_pred, xgb_val_pred; gc.collect()

# ── 8. OOF 평가 ─────────────────────────────────────────────────────
print("\n[8] OOF 평가...")
oof_lgb_mae = float(np.mean(np.abs(oof_lgb - y_true)))
oof_xgb_mae = float(np.mean(np.abs(oof_xgb - y_true)))

# LGB + XGB 최적 blend
best_bl_w, best_bl_mae = 0.5, float('inf')
for w in np.arange(0.1, 0.91, 0.1):
    bl = w * oof_lgb + (1 - w) * oof_xgb
    m  = float(np.mean(np.abs(bl - y_true)))
    if m < best_bl_mae:
        best_bl_mae, best_bl_w = m, w
oof_blend = np.clip(best_bl_w * oof_lgb + (1 - best_bl_w) * oof_xgb, 0, None)

corr_lgb   = float(np.corrcoef(fw4_oo, oof_lgb)[0, 1])
corr_xgb   = float(np.corrcoef(fw4_oo, oof_xgb)[0, 1])
corr_blend = float(np.corrcoef(fw4_oo, oof_blend)[0, 1])

print(f"\n  LGB  OOF: {oof_lgb_mae:.4f}  corr(oracle): {corr_lgb:.4f}")
print(f"  XGB  OOF: {oof_xgb_mae:.4f}  corr(oracle): {corr_xgb:.4f}")
print(f"  Blend OOF: {best_bl_mae:.4f}  w_lgb={best_bl_w:.2f}  corr(oracle): {corr_blend:.4f}")
print(f"  oracle_NEW OOF: {oracle_mae:.4f}")

# fold별 MAE 재출력
print("\n  fold별 MAE (val 기준):")
for fold_i, (lo, hi) in enumerate(BLOCKS):
    mask = (sc_num_arr >= lo) & (sc_num_arr <= hi)
    fl = float(np.mean(np.abs(oof_lgb[mask]  - y_true[mask])))
    fx = float(np.mean(np.abs(oof_xgb[mask]  - y_true[mask])))
    fb = float(np.mean(np.abs(oof_blend[mask] - y_true[mask])))
    print(f"    Fold {fold_i} sc{lo}-{hi}: LGB={fl:.4f}  XGB={fx:.4f}  blend={fb:.4f}")

# ── 9. Test 예측 ─────────────────────────────────────────────────────
print("\n[9] Test 예측...")
test_lgb_avg = np.mean(test_lgb_list, axis=0).astype(np.float32)
test_xgb_avg = np.mean(test_xgb_list, axis=0).astype(np.float32)
test_blend   = np.clip(best_bl_w * test_lgb_avg + (1 - best_bl_w) * test_xgb_avg, 0, None)

unseen_mask = ~test_raw['layout_seen'].values
print(f"  test seen   mean: {test_blend[~unseen_mask].mean():.4f}")
print(f"  test unseen mean: {test_blend[unseen_mask].mean():.4f}  (oracle: 22.716)")
print(f"  test overall mean: {test_blend.mean():.4f}")

# ── 10. oracle_NEW + temporal 블렌드 분석 ────────────────────────────
print("\n[10] oracle_NEW + temporal 블렌드 분석...")
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(test_raw['ID'].values).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values.astype(np.float32)

best_w2, best_m2 = 0.0, oracle_mae
print(f"  oracle_NEW 단독 OOF: {oracle_mae:.4f}")
for w in np.arange(0.05, 0.61, 0.05):
    bl2 = (1 - w) * fw4_oo + w * oof_blend
    m2  = float(np.mean(np.abs(bl2 - y_true)))
    gain = oracle_mae - m2
    marker = " ★" if m2 < best_m2 else ""
    print(f"    w_temporal={w:.2f}: OOF={m2:.4f}  gain={gain:+.4f}{marker}")
    if m2 < best_m2:
        best_m2, best_w2 = m2, w

print(f"\n  최적: w_temporal={best_w2:.2f}  OOF={best_m2:.4f}  gain={oracle_mae-best_m2:+.4f}")

# 최적 blend test 예측
final_test = np.clip((1 - best_w2) * oracle_new_t + best_w2 * test_blend, 0, None)
print(f"  final_test unseen mean: {final_test[unseen_mask].mean():.4f}")

# ── 11. 저장 ─────────────────────────────────────────────────────────
print("\n[11] 결과 저장...")
np.save('results/temporal_oracle/oof_temporal_lgb.npy',   oof_lgb)
np.save('results/temporal_oracle/oof_temporal_xgb.npy',   oof_xgb)
np.save('results/temporal_oracle/oof_temporal_blend.npy',  oof_blend)
np.save('results/temporal_oracle/test_temporal_lgb.npy',  test_lgb_avg)
np.save('results/temporal_oracle/test_temporal_xgb.npy',  test_xgb_avg)
np.save('results/temporal_oracle/test_temporal_blend.npy', test_blend)

# sample_submission ID 순서로 재정렬
sub_base = pd.read_csv('sample_submission.csv')
id_to_pred_blend = dict(zip(test_raw['ID'].values, test_blend))
id_to_pred_final = dict(zip(test_raw['ID'].values, final_test))

# temporal blend 단독 (OOF < 9.0 조건)
if best_bl_mae < 9.0:
    sub1 = sub_base.copy()
    sub1['avg_delay_minutes_next_30m'] = sub1['ID'].map(id_to_pred_blend)
    fname1 = f'FINAL_temporal_blend_OOF{best_bl_mae:.4f}.csv'
    sub1.to_csv(fname1, index=False)
    print(f"  *** SAVED: {fname1} ***")

# oracle_NEW + temporal blend (OOF 개선 시)
if best_w2 > 0 and best_m2 < oracle_mae:
    sub2 = sub_base.copy()
    sub2['avg_delay_minutes_next_30m'] = sub2['ID'].map(id_to_pred_final)
    fname2 = f'FINAL_temporal_oracle_blend_w{best_w2:.2f}_OOF{best_m2:.4f}.csv'
    sub2.to_csv(fname2, index=False)
    print(f"  *** SAVED: {fname2} ***")
else:
    print("  oracle_NEW + temporal blend: OOF 개선 없음 — submission 생성 안 함")

print("\nDone.")
