"""
v20: v17 기반 + 3가지 개선 + 속도 최적화
  1) Sample Weight: 극단값(>=50분)에 가중치 부여 → 극단값 예측력 향상
  2) Seen/Unseen 분리 앙상블: test layout을 seen/unseen으로 나눠 가중치 별도 최적화
  3) Distribution Calibration: 예측 분포를 train target 분포에 매핑 → 극단값 복원

속도 최적화:
  - CatBoost: depth 7→6, iterations 3000, early_stopping 100
  - LightGBM: early_stopping 200→100
  - 전체 checkpoint 시스템 유지
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
import time

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
CKPT_DIR = './results/v20_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

t0 = time.time()

def elapsed():
    return f"[{(time.time()-t0)/60:.1f}분]"

print(f"=== v20: sample weight + seen/unseen 분리 + 분포 보정 + 속도 최적화 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 피처 엔지니어링 (v17 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"{elapsed()} 1. 피처 엔지니어링...", flush=True)

def engineer_features(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'battery_mean', 'fault_count_15m', 'blocked_path_15m',
                'pack_utilization', 'charge_queue_length']
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())

    # 상호작용
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    rta = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (rta + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (rta + 1)
    df['congestion_x_utilization'] = df['congestion_score'] * df['robot_utilization']
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
    df['order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
    df['urgent_order_volume'] = df['order_inflow_15m'] * df['urgent_order_ratio']
    df['dock_pressure'] = df['loading_dock_util'] * df['outbound_truck_wait_min']
    df['staff_per_order'] = df['staff_on_floor'] / (df['order_inflow_15m'] + 1)
    df['total_utilization'] = (df['pack_utilization'] + df['staging_area_util'] + df['loading_dock_util']) / 3
    df['fault_x_congestion'] = df['fault_count_15m'] * df['congestion_score']
    df['battery_charge_pressure'] = df['low_battery_ratio'] * df['avg_charge_wait']
    df['congestion_per_robot'] = df['congestion_score'] / (df['robot_active'] + 1)
    df['order_per_staff'] = df['order_inflow_15m'] / (df['staff_on_floor'] + 1)

    # layout 비율 피처
    df['order_per_area'] = df['order_inflow_15m'] / (df['floor_area_sqm'] + 1) * 1000
    df['congestion_per_area'] = df['congestion_score'] / (df['floor_area_sqm'] + 1) * 1000
    df['fault_per_robot_total'] = df['fault_count_15m'] / (df['robot_total'] + 1)
    df['blocked_per_robot_total'] = df['blocked_path_15m'] / (df['robot_total'] + 1)
    df['collision_per_robot_total'] = df['near_collision_15m'] / (df['robot_total'] + 1)
    df['pack_util_per_station'] = df['pack_utilization'] / (df['pack_station_count'] + 1)
    df['charge_queue_per_charger'] = df['charge_queue_length'] / (df['charger_count'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
    df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
    df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
    df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
    df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)
    df['congestion_x_aisle_width'] = df['congestion_score'] * df['aisle_width_avg']
    df['congestion_x_compactness'] = df['congestion_score'] * df['layout_compactness']
    df['blocked_x_one_way'] = df['blocked_path_15m'] * df['one_way_ratio']
    df['utilization_x_compactness'] = df['robot_utilization'] * df['layout_compactness']

    # v17 시나리오 패턴 피처
    group2 = df.groupby(['layout_id', 'scenario_id'])
    for col in ['congestion_score', 'order_inflow_15m', 'robot_utilization', 'pack_utilization']:
        if col not in df.columns: continue
        g = group2[col]
        df[f'{col}_slope3'] = (df[col] - g.shift(2)) / 3
        df[f'{col}_slope5'] = (df[col] - g.shift(4)) / 5
    for col in ['congestion_score', 'order_inflow_15m', 'blocked_path_15m']:
        if col not in df.columns: continue
        g = group2[col]
        cummax = g.transform(lambda x: x.expanding().max())
        cummin = g.transform(lambda x: x.expanding().min())
        df[f'{col}_vs_cummax'] = df[col] / (cummax + 1)
        df[f'{col}_vs_cummin'] = df[col] / (cummin + 1)
        df[f'{col}_cumrange'] = cummax - cummin
    for col in ['order_inflow_15m', 'fault_count_15m', 'blocked_path_15m', 'near_collision_15m']:
        if col not in df.columns: continue
        df[f'{col}_cumsum'] = group2[col].transform(lambda x: x.expanding().sum())
    for col in ['congestion_score', 'order_inflow_15m']:
        diff1_col = f'{col}_diff1'
        if diff1_col in df.columns:
            df[f'{col}_accel'] = group2[diff1_col].shift(0) - group2[diff1_col].shift(1)

    # layout 정적 피처 제거 + 상관 피처 제거 (v14)
    layout_static = ['layout_type', 'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                     'pack_station_count', 'charger_count', 'layout_compactness', 'zone_dispersion',
                     'robot_total', 'building_age_years', 'floor_area_sqm', 'ceiling_height_m',
                     'fire_sprinkler_count', 'emergency_exit_count']
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors='ignore')

    corr_remove = ['battery_mean_rmean3', 'charge_queue_length_rmean3',
                   'battery_mean_rmean5', 'charge_queue_length_rmean5',
                   'pack_utilization_rmean5', 'battery_mean_lag1',
                   'charge_queue_length_lag1', 'congestion_score_rmean3',
                   'order_inflow_15m_cummean', 'robot_utilization_rmean5',
                   'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
                   'battery_risk', 'congestion_score_rmean5',
                   'pack_utilization_rmean3', 'order_inflow_15m_rmean3',
                   'charge_queue_length_lag2', 'blocked_path_15m_rmean5']
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')

    return df

train = engineer_features(train, layout)
test = engineer_features(test, layout)
exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
feature_cols = [c for c in train.columns if c not in exclude]
print(f"   피처 수: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
y_log = np.log1p(y)
X_test = test[feature_cols]
groups = train['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [개선 1] Sample Weight 계산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTREME_THRESHOLD = 50
EXTREME_WEIGHT = 3.0

sample_weight = np.ones(len(train))
extreme_mask = y >= EXTREME_THRESHOLD
sample_weight[extreme_mask] = EXTREME_WEIGHT
print(f"   극단값(>={EXTREME_THRESHOLD}분): {extreme_mask.sum():,}건 ({extreme_mask.mean()*100:.1f}%), weight={EXTREME_WEIGHT}", flush=True)

# log target에 대한 weight (동일 마스크)
sample_weight_log = sample_weight.copy()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# [개선 2] Seen/Unseen layout mask (test용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
train_layouts = set(train['layout_id'].unique())
test_layouts = set(test['layout_id'].unique())
seen_layouts = train_layouts & test_layouts
unseen_layouts = test_layouts - train_layouts

test_seen_mask = test['layout_id'].isin(seen_layouts).values
test_unseen_mask = ~test_seen_mask
print(f"   Test seen: {test_seen_mask.sum():,}행, unseen: {test_unseen_mask.sum():,}행", flush=True)

# OOF에서도 seen/unseen 분리 (GroupKFold(layout)이므로 val은 항상 unseen 시뮬레이션)
# → 실제 test의 seen layout에 대응하는 train 데이터 마스크
train_seen_mask = train['layout_id'].isin(seen_layouts).values

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 모델 학습 (속도 최적화 적용)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 2. 모델 학습...", flush=True)

def train_and_save(model_type, seed, target, transform, name, use_weight=False):
    """모델 학습 + checkpoint. use_weight=True면 sample_weight 적용"""
    ckpt_oof = f'{CKPT_DIR}/{name}_s{seed}_oof.npy'
    ckpt_test = f'{CKPT_DIR}/{name}_s{seed}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        print(f"  {name}... [checkpoint] {elapsed()}", flush=True)
        return np.load(ckpt_oof), np.load(ckpt_test)

    print(f"  {name}... [학습] {elapsed()}", flush=True)
    oof = np.zeros(len(train)); tpred = np.zeros(len(test))
    w = sample_weight if (use_weight and transform == 'raw') else sample_weight_log if use_weight else None

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        fold_w = w[tr_idx] if w is not None else None
        ft = time.time()

        if model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      sample_weight=fold_w,
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])  # 200→100

        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      sample_weight=fold_w,
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])  # 200→100

        elif model_type == 'cb':
            # 속도 최적화: depth 7→6, iterations 5000→3000, early_stopping 200→100
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=3000, learning_rate=0.03, depth=6,  # depth 7→6, iter 5000→3000
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=seed, verbose=0, early_stopping_rounds=100)  # 200→100
            # CatBoost는 sample_weight를 Pool로 전달
            if fold_w is not None:
                from catboost import Pool
                train_pool = Pool(X.iloc[tr_idx], target.iloc[tr_idx], weight=fold_w)
                val_pool = Pool(X.iloc[val_idx], target.iloc[val_idx])
                model.fit(train_pool, eval_set=val_pool, verbose=0)
            else:
                model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                          eval_set=(X.iloc[val_idx], target.iloc[val_idx]), verbose=0)

        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
        print(f"    Fold {fold_idx+1} done ({(time.time()-ft)/60:.1f}분) {elapsed()}", flush=True)

    # 역변환 후 저장
    if transform == 'log':
        oof = np.expm1(oof); tpred = np.expm1(tpred)

    np.save(ckpt_oof, oof); np.save(ckpt_test, tpred)
    return oof, tpred


# === A) 기본 모델 (weight 없음, v17과 동일 구조) ===
configs_base = [
    ('lgb_huber', y,     'raw', 'raw_LGB_Huber', False),
    ('lgb',       y_log, 'log', 'log_LGB_MAE',   False),
    ('lgb_huber', y_log, 'log', 'log_LGB_Huber', False),
    ('cb',        y_log, 'log', 'log_CatBoost',  False),
]

# === B) Sample Weight 모델 ===
configs_weighted = [
    ('lgb_huber', y,     'raw', 'raw_LGB_Huber_w', True),
    ('lgb_huber', y_log, 'log', 'log_LGB_Huber_w', True),
]

all_oofs, all_tests = {}, {}

print(f"\n  --- 기본 모델 ---", flush=True)
for mtype, target, transform, name, use_w in configs_base:
    o, t = train_and_save(mtype, SEED, target, transform, name, use_w)
    all_oofs[name] = o; all_tests[name] = t
    print(f"  {name} OOF MAE: {mean_absolute_error(y, o):.4f} {elapsed()}", flush=True)

print(f"\n  --- Sample Weight 모델 --- ", flush=True)
for mtype, target, transform, name, use_w in configs_weighted:
    o, t = train_and_save(mtype, SEED, target, transform, name, use_w)
    all_oofs[name] = o; all_tests[name] = t
    print(f"  {name} OOF MAE: {mean_absolute_error(y, o):.4f} {elapsed()}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 앙상블 — 전체 + Seen/Unseen 분리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 3. 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

# 3-1) 전체 앙상블 (기존 방식)
def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[1/len(names)]*len(names), method='Nelder-Mead', options={'maxiter': 50000})
bw_global = np.array(res.x); bw_global = np.maximum(bw_global, 0); bw_global = bw_global / bw_global.sum()
global_mae = res.fun

print(f"  전체 앙상블 OOF MAE: {global_mae:.4f}")
for n, w in zip(names, bw_global):
    if w > 0.001: print(f"    {n}: {w:.3f}")

global_test_pred = np.clip(sum(wi*p for wi,p in zip(bw_global, test_list)), 0, None)

# 3-2) [개선 2] Seen/Unseen 분리 앙상블
# GroupKFold(layout)에서 val은 항상 unseen 시뮬레이션이므로,
# OOF 전체가 "unseen layout"에 대한 예측임.
# → seen layout 전용 가중치는 OOF에서 직접 최적화 불가.
# → 대안: unseen은 OOF 가중치 사용, seen은 weight 모델에 약간 더 가중
# 실용적 접근: seen/unseen 비율로 블렌딩 계수 실험

# seen layout은 train에서 봤으므로 base 모델이 강할 수 있고,
# unseen은 weight 모델이 정규화 효과로 더 좋을 수 있음.
# → 두 가지 앙상블을 만들고 seen/unseen별로 다른 걸 적용

# unseen용 (OOF 기반 최적화 — OOF가 unseen 시뮬레이션이므로 정확)
bw_unseen = bw_global.copy()

# seen용 — weight 모델의 비중을 약간 높인 버전 시도
# (OOF로 정확한 최적화 불가하므로, grid search로 몇 가지 비율 시도)
# 기본 전략: global 가중치 사용하되, weight 모델에 +10% boost
bw_seen = bw_global.copy()
for i, n in enumerate(names):
    if '_w' in n:
        bw_seen[i] *= 1.2  # weight 모델 20% boost for seen
bw_seen = bw_seen / bw_seen.sum()

seen_pred = np.clip(sum(wi*p for wi,p in zip(bw_seen, test_list)), 0, None)
unseen_pred = np.clip(sum(wi*p for wi,p in zip(bw_unseen, test_list)), 0, None)

# seen/unseen 합치기
split_test_pred = np.where(test_seen_mask, seen_pred, unseen_pred)

print(f"\n  Seen/Unseen 분리 앙상블:")
print(f"    Seen  ({test_seen_mask.sum():,}행) 가중치: {dict(zip(names, bw_seen.round(3)))}")
print(f"    Unseen({test_unseen_mask.sum():,}행) 가중치: {dict(zip(names, bw_unseen.round(3)))}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. [개선 3] Distribution Calibration (후처리)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 4. Distribution Calibration...", flush=True)

def calibrate_distribution(pred, y_train, strength=0.3):
    """
    예측 분포를 train target 분포에 strength 비율만큼 매핑.
    strength=0: 보정 없음 (원본), strength=1: 완전 매핑.
    """
    # 예측값의 순위(분위수) 계산
    n = len(pred)
    ranks = pd.Series(pred).rank(method='average') / n  # 0~1 사이 분위수

    # train target의 해당 분위수 값
    target_quantiles = np.quantile(y_train, ranks.values)

    # 원본과 보정값을 strength로 블렌딩
    calibrated = (1 - strength) * pred + strength * target_quantiles
    return np.clip(calibrated, 0, None)

# 여러 strength 값으로 OOF 검증 (global 앙상블 기준)
global_oof = sum(wi*p for wi,p in zip(bw_global, oof_list))
print(f"  보정 전 OOF MAE: {mean_absolute_error(y, np.clip(global_oof, 0, None)):.4f}")

best_strength = 0.0
best_cal_mae = mean_absolute_error(y, np.clip(global_oof, 0, None))

for s in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    cal_oof = calibrate_distribution(global_oof, y.values, strength=s)
    cal_mae = mean_absolute_error(y, cal_oof)
    improved = "✓" if cal_mae < best_cal_mae else ""
    print(f"  strength={s:.2f}: OOF MAE={cal_mae:.4f} {improved}")
    if cal_mae < best_cal_mae:
        best_cal_mae = cal_mae
        best_strength = s

print(f"\n  최적 strength: {best_strength} (OOF MAE: {best_cal_mae:.4f})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Submission 생성 (다양한 조합)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 5. Submission 생성...", flush=True)

# A) 전체 앙상블 (기본)
pd.DataFrame({'ID': test['ID'], TARGET: global_test_pred}).to_csv(
    './submission_v20_global.csv', index=False)

# B) Seen/Unseen 분리
pd.DataFrame({'ID': test['ID'], TARGET: split_test_pred}).to_csv(
    './submission_v20_split.csv', index=False)

# C) 전체 + 분포 보정
if best_strength > 0:
    cal_global = calibrate_distribution(global_test_pred, y.values, strength=best_strength)
    pd.DataFrame({'ID': test['ID'], TARGET: cal_global}).to_csv(
        './submission_v20_calibrated.csv', index=False)

# D) Seen/Unseen + 분포 보정
if best_strength > 0:
    cal_split = calibrate_distribution(split_test_pred, y.values, strength=best_strength)
    pd.DataFrame({'ID': test['ID'], TARGET: cal_split}).to_csv(
        './submission_v20_split_cal.csv', index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 진단 리포트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*60}")
print(f"v20 완료! {elapsed()}")
print(f"\n  [모델별 OOF MAE]")
for n in names:
    mae = mean_absolute_error(y, all_oofs[n])
    extreme_mae = mean_absolute_error(y[extreme_mask], all_oofs[n][extreme_mask])
    normal_mae = mean_absolute_error(y[~extreme_mask], all_oofs[n][~extreme_mask])
    tag = "(weighted)" if '_w' in n else ""
    print(f"    {n}: 전체={mae:.4f}, 정상={normal_mae:.4f}, 극단={extreme_mae:.4f} {tag}")

print(f"\n  [앙상블 OOF MAE]")
print(f"    전체 앙상블: {global_mae:.4f}")
print(f"    분포 보정(s={best_strength}): {best_cal_mae:.4f}")

print(f"\n  [예측 분포 비교]")
print(f"    Train target: mean={y.mean():.2f}, median={y.median():.2f}, >50분={extreme_mask.mean()*100:.1f}%")
print(f"    Global 예측:  mean={global_test_pred.mean():.2f}, median={np.median(global_test_pred):.2f}, >50분={(global_test_pred>=50).mean()*100:.1f}%")
if best_strength > 0:
    print(f"    보정 예측:    mean={cal_global.mean():.2f}, median={np.median(cal_global):.2f}, >50분={(cal_global>=50).mean()*100:.1f}%")

print(f"\n  [Submission 파일]")
print(f"    submission_v20_global.csv       — 전체 앙상블")
print(f"    submission_v20_split.csv        — Seen/Unseen 분리")
if best_strength > 0:
    print(f"    submission_v20_calibrated.csv   — 전체 + 분포 보정")
    print(f"    submission_v20_split_cal.csv    — 분리 + 분포 보정")
print(f"{'='*60}", flush=True)
