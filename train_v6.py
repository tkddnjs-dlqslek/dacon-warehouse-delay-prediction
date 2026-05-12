"""
v6: target encoding + seed averaging + quantile + 추가 FE
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
SEEDS = [42, 123, 2024]

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
    df = df.merge(layout_df, on='layout_id', how='left')

    layout_map = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}
    df['layout_type_enc'] = df['layout_type'].map(layout_map)

    # 타임슬롯
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df['is_early'] = (df['timeslot'] < 5).astype(np.int8)
    df['is_mid'] = ((df['timeslot'] >= 8) & (df['timeslot'] <= 16)).astype(np.int8)
    df['is_late'] = (df['timeslot'] > 19).astype(np.int8)
    df['timeslot_sin'] = np.sin(2 * np.pi * df['timeslot'] / 25)
    df['timeslot_cos'] = np.cos(2 * np.pi * df['timeslot'] / 25)

    # Lag/Diff/Rolling
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
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        df[f'{col}_lag3'] = g.shift(3)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_diff2'] = df[f'{col}_lag1'] - df[f'{col}_lag2']
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f'{col}_rmax3'] = g.transform(lambda x: x.rolling(3, min_periods=1).max())
        df[f'{col}_rmin3'] = g.transform(lambda x: x.rolling(3, min_periods=1).min())
        df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())
        df[f'{col}_cummax'] = g.transform(lambda x: x.expanding().max())
        df[f'{col}_cumstd'] = g.transform(lambda x: x.expanding().std())
        # range (max - min) in window
        df[f'{col}_rrange3'] = df[f'{col}_rmax3'] - df[f'{col}_rmin3']

    # 상호작용 피처
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

    # 추가 v6 피처
    df['congestion_per_robot'] = df['congestion_score'] / (df['robot_active'] + 1)
    df['blocked_per_intersection'] = df['blocked_path_15m'] / (df['intersection_count'] + 1)
    df['collision_per_robot'] = df['near_collision_15m'] / (df['robot_active'] + 1)
    df['order_x_pack'] = df['order_inflow_15m'] * df['pack_utilization']
    df['idle_x_congestion'] = df['avg_idle_duration_min'] * df['congestion_score']
    df['trip_x_congestion'] = df['avg_trip_distance'] * df['congestion_score']
    df['wms_x_order'] = df['wms_response_time_ms'] * df['order_inflow_15m']
    df['scanner_x_order'] = df['scanner_error_rate'] * df['order_inflow_15m']
    df['fault_rate_per_robot'] = df['fault_count_15m'] / (df['robot_active'] + 1)
    df['charge_wait_x_queue'] = df['avg_charge_wait'] * df['charge_queue_length']
    df['heavy_x_conveyor'] = df['heavy_item_ratio'] * df['conveyor_speed_mps']
    df['bulk_x_pack'] = df['bulk_order_ratio'] * df['pack_utilization']
    df['sku_concentration_x_order'] = df['sku_concentration'] * df['order_inflow_15m']

    # Layout 기반
    if 'floor_area_sqm' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['pack_station_density'] = df['pack_station_count'] / (df['floor_area_sqm'] + 1) * 1000
        df['robot_per_intersection'] = df['robot_total'] / (df['intersection_count'] + 1)
        df['area_compactness'] = df['floor_area_sqm'] * df['layout_compactness']
        df['exit_per_area'] = df['emergency_exit_count'] / (df['floor_area_sqm'] + 1) * 10000
        df['charger_per_area'] = df['charger_count'] / (df['floor_area_sqm'] + 1) * 10000
        df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
        df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)

    # 결측 플래그
    missing_cols = ['congestion_score', 'battery_mean', 'avg_recovery_time',
                    'avg_charge_wait', 'charge_efficiency_pct', 'low_battery_ratio',
                    'aisle_traffic_score']
    for col in missing_cols:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)

    return df


train = engineer_features(train, layout)
test = engineer_features(test, layout)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.5 Target Encoding (OOF 방식)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("2.5. Target Encoding (layout_id, layout_type)...", flush=True)

y_full = train[TARGET]
gkf_te = GroupKFold(n_splits=N_SPLITS)
groups_te = train['scenario_id']

# layout_id target encoding
train['layout_id_te'] = 0.0
for tr_idx, val_idx in gkf_te.split(train, y_full, groups_te):
    means = train.loc[tr_idx].groupby('layout_id')[TARGET].mean()
    train.loc[val_idx, 'layout_id_te'] = train.loc[val_idx, 'layout_id'].map(means)

global_layout_means = train.groupby('layout_id')[TARGET].mean()
test['layout_id_te'] = test['layout_id'].map(global_layout_means)
test['layout_id_te'] = test['layout_id_te'].fillna(y_full.mean())

# layout_type target encoding
train['layout_type_te'] = 0.0
for tr_idx, val_idx in gkf_te.split(train, y_full, groups_te):
    means = train.loc[tr_idx].groupby('layout_type')[TARGET].mean()
    train.loc[val_idx, 'layout_type_te'] = train.loc[val_idx, 'layout_type'].map(means)

global_type_means = train.groupby('layout_type')[TARGET].mean()
test['layout_type_te'] = test['layout_type'].map(global_type_means)

# 피처 컬럼
exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
feature_cols = [c for c in train.columns if c not in exclude]
print(f"   피처 수: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]
groups = train['scenario_id']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. Seed Averaging + 3모델 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

all_oof = {}  # model_name -> oof predictions
all_test = {}  # model_name -> test predictions


def train_lgb_seed(seed, objective='mae'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(
            objective=objective, n_estimators=5000, learning_rate=0.03,
            num_leaves=127, max_depth=-1, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


def train_xgb_seed(seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
            max_depth=8, min_child_weight=5,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            tree_method='hist', random_state=seed, verbosity=0, n_jobs=-1,
            early_stopping_rounds=200,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


def train_cb_seed(seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = CatBoostRegressor(
            loss_function='MAE', eval_metric='MAE',
            iterations=5000, learning_rate=0.03, depth=8,
            l2_leaf_reg=3.0, random_strength=1.0, bagging_temperature=1.0,
            random_seed=seed, verbose=0, early_stopping_rounds=200,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


# LightGBM MAE (3 seeds)
print("\n3-A. LightGBM MAE (3 seeds)...", flush=True)
lgb_oofs, lgb_tests = [], []
for seed in SEEDS:
    print(f"  Seed {seed}...", flush=True)
    o, t = train_lgb_seed(seed, 'mae')
    lgb_oofs.append(o)
    lgb_tests.append(t)
lgb_oof = np.mean(lgb_oofs, axis=0)
lgb_test_pred = np.mean(lgb_tests, axis=0)
print(f"  LGB MAE OOF: {mean_absolute_error(y, lgb_oof):.4f}", flush=True)

# LightGBM Quantile 0.5 (3 seeds)
print("\n3-B. LightGBM Quantile (3 seeds)...", flush=True)
lgq_oofs, lgq_tests = [], []
for seed in SEEDS:
    print(f"  Seed {seed}...", flush=True)
    o, t = train_lgb_seed(seed, 'quantile')
    lgq_oofs.append(o)
    lgq_tests.append(t)
lgq_oof = np.mean(lgq_oofs, axis=0)
lgq_test_pred = np.mean(lgq_tests, axis=0)
print(f"  LGB Quantile OOF: {mean_absolute_error(y, lgq_oof):.4f}", flush=True)

# XGBoost (3 seeds)
print("\n3-C. XGBoost (3 seeds)...", flush=True)
xgb_oofs, xgb_tests = [], []
for seed in SEEDS:
    print(f"  Seed {seed}...", flush=True)
    o, t = train_xgb_seed(seed)
    xgb_oofs.append(o)
    xgb_tests.append(t)
xgb_oof = np.mean(xgb_oofs, axis=0)
xgb_test_pred = np.mean(xgb_tests, axis=0)
print(f"  XGB OOF: {mean_absolute_error(y, xgb_oof):.4f}", flush=True)

# CatBoost (3 seeds)
print("\n3-D. CatBoost (3 seeds)...", flush=True)
cb_oofs, cb_tests = [], []
for seed in SEEDS:
    print(f"  Seed {seed}...", flush=True)
    o, t = train_cb_seed(seed)
    cb_oofs.append(o)
    cb_tests.append(t)
cb_oof = np.mean(cb_oofs, axis=0)
cb_test_pred = np.mean(cb_tests, axis=0)
print(f"  CB OOF: {mean_absolute_error(y, cb_oof):.4f}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 가중 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. 가중 앙상블 최적화...", flush=True)

oof_list = [lgb_oof, lgq_oof, xgb_oof, cb_oof]
test_list = [lgb_test_pred, lgq_test_pred, xgb_test_pred, cb_test_pred]
model_names = ['LGB_MAE', 'LGB_Quantile', 'XGB', 'CatBoost']

for name, oof in zip(model_names, oof_list):
    print(f"  {name}: {mean_absolute_error(y, oof):.4f}")


def ens_mae(w):
    w = np.array(w)
    w = w / w.sum()
    return mean_absolute_error(y, sum(wi * p for wi, p in zip(w, oof_list)))


res = minimize(ens_mae, x0=[0.25]*4, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x)
bw = bw / bw.sum()
weighted_mae = res.fun

simple_pred = sum(o for o in oof_list) / len(oof_list)
simple_mae = mean_absolute_error(y, simple_pred)

print(f"\n  단순 평균 OOF MAE: {simple_mae:.4f}")
print(f"  가중 앙상블 OOF MAE: {weighted_mae:.4f}")
print(f"  가중치: {dict(zip(model_names, bw.round(3)))}")

final_pred = sum(wi * p for wi, p in zip(bw, test_list))
final_pred = np.clip(final_pred, 0, None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 제출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sub = pd.DataFrame({'ID': test['ID'], TARGET: final_pred})
sub.to_csv('./submission_v6.csv', index=False)

print(f"\n{'='*60}")
print(f"v6 완료!")
print(f"  가중 앙상블 OOF MAE: {weighted_mae:.4f}")
print(f"  예측: mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
print(f"  submission_v6.csv 저장 완료")
print(f"{'='*60}", flush=True)
