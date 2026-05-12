"""
v5: 강화된 피처 엔지니어링 + GroupKFold + 3모델 앙상블 (누수 없음)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 데이터 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("1. 데이터 로드...", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 피처 엔지니어링
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("2. 피처 엔지니어링...", flush=True)


def engineer_features(df, layout_df):
    # 2.1 Layout 병합
    df = df.merge(layout_df, on='layout_id', how='left')

    # 2.2 layout_type encoding
    layout_map = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}
    df['layout_type_enc'] = df['layout_type'].map(layout_map)

    # 2.3 타임슬롯 파생
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df['is_early'] = (df['timeslot'] < 5).astype(np.int8)
    df['is_mid'] = ((df['timeslot'] >= 8) & (df['timeslot'] <= 16)).astype(np.int8)
    df['is_late'] = (df['timeslot'] > 19).astype(np.int8)

    # 2.4 Lag/Diff/Rolling (시나리오 내 — 시계열 피처)
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    key_cols = [
        'order_inflow_15m', 'congestion_score', 'robot_utilization',
        'battery_mean', 'fault_count_15m', 'blocked_path_15m',
        'pack_utilization', 'charge_queue_length', 'near_collision_15m',
        'avg_trip_distance', 'max_zone_density', 'loading_dock_util',
    ]

    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        # Lag
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        df[f'{col}_lag3'] = g.shift(3)
        # Diff
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_diff2'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
        # Rolling
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f'{col}_rmax3'] = g.transform(lambda x: x.rolling(3, min_periods=1).max())
        df[f'{col}_rmin3'] = g.transform(lambda x: x.rolling(3, min_periods=1).min())
        df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        # Cumulative
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())
        df[f'{col}_cummax'] = g.transform(lambda x: x.expanding().max())

    # 2.5 상호작용 피처
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    robot_total_active = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (robot_total_active + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (robot_total_active + 1)
    df['battery_risk'] = df['low_battery_ratio'] * df['charge_queue_length']
    df['congestion_x_utilization'] = df['congestion_score'] * df['robot_utilization']
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
    df['order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
    df['urgent_order_volume'] = df['order_inflow_15m'] * df['urgent_order_ratio']
    df['dock_pressure'] = df['loading_dock_util'] * df['outbound_truck_wait_min']
    df['network_issues'] = df['wms_response_time_ms'] * df['network_latency_ms']
    df['quality_composite'] = (
        df['barcode_read_success_rate'] + df['sort_accuracy_pct'] + df['agv_task_success_rate']
    ) / 3
    df['staff_per_order'] = df['staff_on_floor'] / (df['order_inflow_15m'] + 1)
    df['pack_x_staging'] = df['pack_utilization'] * df['staging_area_util']
    df['total_utilization'] = (
        df['pack_utilization'] + df['staging_area_util'] + df['loading_dock_util']
    ) / 3
    df['fault_x_congestion'] = df['fault_count_15m'] * df['congestion_score']
    df['order_density'] = df['order_inflow_15m'] * df['unique_sku_15m']
    df['battery_charge_pressure'] = df['low_battery_ratio'] * df['avg_charge_wait']

    # 2.6 Layout 기반 비율
    if 'floor_area_sqm' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['pack_station_density'] = df['pack_station_count'] / (df['floor_area_sqm'] + 1) * 1000
        df['robot_per_intersection'] = df['robot_total'] / (df['intersection_count'] + 1)
        df['area_compactness'] = df['floor_area_sqm'] * df['layout_compactness']

    # 2.7 결측 플래그 (상위 중요 컬럼)
    missing_cols = ['congestion_score', 'battery_mean', 'avg_recovery_time',
                    'avg_charge_wait', 'charge_efficiency_pct']
    for col in missing_cols:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)

    return df


train = engineer_features(train, layout)
test = engineer_features(test, layout)

# 피처 컬럼 정의
exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
feature_cols = [c for c in train.columns if c not in exclude]
print(f"   피처 수: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]
groups = train['scenario_id']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. GroupKFold CV + 3모델 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

# ── LightGBM ──
print("\n3-A. LightGBM 학습...", flush=True)
lgb_oof = np.zeros(len(train))
lgb_test = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(folds):
    print(f"  LGB Fold {fold+1}...", flush=True)
    model = lgb.LGBMRegressor(
        objective='mae', n_estimators=5000, learning_rate=0.03,
        num_leaves=127, max_depth=-1, min_child_samples=20,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1,
    )
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
    )
    lgb_oof[val_idx] = model.predict(X.iloc[val_idx])
    lgb_test += model.predict(X_test) / N_SPLITS

lgb_mae = mean_absolute_error(y, lgb_oof)
print(f"  LightGBM OOF MAE: {lgb_mae:.4f}", flush=True)

# ── XGBoost ──
print("\n3-B. XGBoost 학습...", flush=True)
xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(folds):
    print(f"  XGB Fold {fold+1}...", flush=True)
    model = xgb.XGBRegressor(
        objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
        max_depth=8, min_child_weight=5,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        tree_method='hist', random_state=SEED, verbosity=0, n_jobs=-1,
        early_stopping_rounds=200,
    )
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
        verbose=0,
    )
    xgb_oof[val_idx] = model.predict(X.iloc[val_idx])
    xgb_test += model.predict(X_test) / N_SPLITS

xgb_mae = mean_absolute_error(y, xgb_oof)
print(f"  XGBoost OOF MAE: {xgb_mae:.4f}", flush=True)

# ── CatBoost ──
print("\n3-C. CatBoost 학습...", flush=True)
cb_oof = np.zeros(len(train))
cb_test = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(folds):
    print(f"  CB Fold {fold+1}...", flush=True)
    model = CatBoostRegressor(
        loss_function='MAE', eval_metric='MAE',
        iterations=5000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
        random_seed=SEED, verbose=0, early_stopping_rounds=200,
    )
    model.fit(
        X.iloc[tr_idx], y.iloc[tr_idx],
        eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
        verbose=0,
    )
    cb_oof[val_idx] = model.predict(X.iloc[val_idx])
    cb_test += model.predict(X_test) / N_SPLITS

cb_mae = mean_absolute_error(y, cb_oof)
print(f"  CatBoost OOF MAE: {cb_mae:.4f}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 가중 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. 가중 앙상블 최적화...", flush=True)

oof_list = [lgb_oof, xgb_oof, cb_oof]
test_list = [lgb_test, xgb_test, cb_test]


def ens_mae(w):
    w = np.array(w)
    w = w / w.sum()
    return mean_absolute_error(y, sum(wi * p for wi, p in zip(w, oof_list)))


res = minimize(ens_mae, x0=[1/3, 1/3, 1/3], method='Nelder-Mead')
bw = np.array(res.x)
bw = bw / bw.sum()

# 단순 평균도 비교
simple_mae = mean_absolute_error(y, (lgb_oof + xgb_oof + cb_oof) / 3)
weighted_mae = res.fun

print(f"  단순 평균 OOF MAE: {simple_mae:.4f}")
print(f"  가중 앙상블 OOF MAE: {weighted_mae:.4f}")
print(f"  가중치: LGB={bw[0]:.3f}, XGB={bw[1]:.3f}, CB={bw[2]:.3f}")

# 더 좋은 것 선택
if weighted_mae < simple_mae:
    final_pred = sum(wi * p for wi, p in zip(bw, test_list))
    final_mae = weighted_mae
    method = 'weighted'
else:
    final_pred = (lgb_test + xgb_test + cb_test) / 3
    final_mae = simple_mae
    method = 'simple_avg'

final_pred = np.clip(final_pred, 0, None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 제출 파일 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n5. 제출 파일 생성...", flush=True)

sub = pd.DataFrame({'ID': test['ID'], TARGET: final_pred})
sub.to_csv('./submission_v5.csv', index=False)

print(f"\n{'='*60}")
print(f"결과 요약")
print(f"{'='*60}")
print(f"  LightGBM OOF MAE: {lgb_mae:.4f}")
print(f"  XGBoost  OOF MAE: {xgb_mae:.4f}")
print(f"  CatBoost OOF MAE: {cb_mae:.4f}")
print(f"  앙상블 ({method}) OOF MAE: {final_mae:.4f}")
print(f"  예측: mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
print(f"  submission_v5.csv 저장 완료")
print(f"{'='*60}", flush=True)
