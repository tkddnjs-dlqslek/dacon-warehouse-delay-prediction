"""
v22: v14 기반 + 3가지 새로운 접근법
  1. Forward-Looking Features (미래 timestep 피처 활용)
  2. Pseudo-Labeling (unseen layout 분포 shift 해결)
  3. Quantile Regression Blend (예측 다양성)
  + 추가 lag 컬럼 (max_zone_density, robot_charging 등)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
import time

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

t0 = time.time()
print("=== v22: Forward-Looking + Pseudo-Label + Quantile Blend ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')
print(f"  데이터 로드 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n1. 피처 엔지니어링...", flush=True)

def engineer_features_v22(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    # === v14 기존 lag/rolling (8 key cols) ===
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

    # === NEW: Forward-Looking Features (미래 피처) ===
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lead1'] = g.shift(-1)     # 다음 timestep
        df[f'{col}_lead2'] = g.shift(-2)     # 2 timestep 후
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]  # 미래 변화량

    # === NEW: 추가 lag/lead 컬럼 (EDA Section 6 Top 5) ===
    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === v14 상호작용 피처 ===
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    rta = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (rta + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (rta + 1)
    df['battery_risk'] = df['low_battery_ratio'] * df['charge_queue_length']
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

    # === v14 layout 비율 피처 ===
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

    # === 제거 ===
    layout_static = ['layout_type', 'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                     'pack_station_count', 'charger_count', 'layout_compactness', 'zone_dispersion',
                     'robot_total', 'building_age_years', 'floor_area_sqm', 'ceiling_height_m',
                     'fire_sprinkler_count', 'emergency_exit_count']
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors='ignore')

    corr_remove = [
        'battery_mean_rmean3', 'charge_queue_length_rmean3',
        'battery_mean_rmean5', 'charge_queue_length_rmean5',
        'pack_utilization_rmean5', 'battery_mean_lag1',
        'charge_queue_length_lag1', 'congestion_score_rmean3',
        'order_inflow_15m_cummean', 'robot_utilization_rmean5',
        'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
        'battery_risk', 'congestion_score_rmean5',
        'pack_utilization_rmean3', 'order_inflow_15m_rmean3',
        'charge_queue_length_lag2', 'blocked_path_15m_rmean5',
    ]
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')

    return df


train_fe = engineer_features_v22(train, layout)
test_fe = engineer_features_v22(test, layout)

y = train_fe[TARGET].values
y_log = np.log1p(y)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
feature_cols = [c for c in train_fe.columns if c not in exclude]
print(f"  피처 수: {len(feature_cols)} (v14: 139 → v22: {len(feature_cols)})", flush=True)

X = train_fe[feature_cols]
X_test = test_fe[feature_cols]
groups = train_fe['layout_id']

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups=groups))

# Sample weights (극단값 약간 가중)
sample_weight = np.ones(len(y))
sample_weight[y > 100] = 1.5
print(f"  극단값 가중치 적용: {(y > 100).sum()}개 rows, weight=1.5", flush=True)
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Base Model Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n2. Base Model Training (5-fold)...", flush=True)

def train_fold(model_type, seed, target, X_train, X_test_df, folds_list, sw=None):
    """단일 모델 5-fold 학습 → OOF + test predictions"""
    oof = np.zeros(len(X_train))
    tpred = np.zeros(len(X_test_df))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds_list):
        tr_w = sw[tr_idx] if sw is not None else None

        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X_train.iloc[tr_idx], target[tr_idx], sample_weight=tr_w,
                      eval_set=[(X_train.iloc[val_idx], target[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        elif model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X_train.iloc[tr_idx], target[tr_idx], sample_weight=tr_w,
                      eval_set=[(X_train.iloc[val_idx], target[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        elif model_type == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=seed, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X_train.iloc[tr_idx], target[tr_idx], sample_weight=tr_w,
                      eval_set=[(X_train.iloc[val_idx], target[val_idx])], verbose=0)

        elif model_type == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=seed, verbose=0, early_stopping_rounds=200)
            model.fit(X_train.iloc[tr_idx], target[tr_idx], sample_weight=tr_w,
                      eval_set=(X_train.iloc[val_idx], target[val_idx]), verbose=0)

        elif model_type.startswith('lgb_q'):
            alpha = float(model_type.split('_q')[1])
            model = lgb.LGBMRegressor(
                objective='quantile', alpha=alpha,
                n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X_train.iloc[tr_idx], target[tr_idx], sample_weight=tr_w,
                      eval_set=[(X_train.iloc[val_idx], target[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        oof[val_idx] = model.predict(X_train.iloc[val_idx])
        tpred += model.predict(X_test_df) / N_SPLITS

    return oof, tpred


# --- log target 모델들 ---
models_config = [
    ('lgb', 'LGB_MAE'),
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
    ('lgb_q0.3', 'LGB_Q30'),
    ('lgb_q0.5', 'LGB_Q50'),
    ('lgb_q0.7', 'LGB_Q70'),
]

log_oofs = {}
log_tests = {}

for mtype, mname in models_config:
    t1 = time.time()
    print(f"  {mname}...", end='', flush=True)
    oof_log, tpred_log = train_fold(mtype, SEED, y_log, X, X_test, folds, sw=sample_weight)
    # expm1 역변환
    oof_real = np.expm1(oof_log)
    tpred_real = np.expm1(tpred_log)
    log_oofs[mname] = oof_real
    log_tests[mname] = tpred_real
    mae = mean_absolute_error(y, oof_real)
    print(f" OOF MAE: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

# --- 원본 target 모델들 (주요 4개만) ---
raw_oofs = {}
raw_tests = {}
for mtype, mname in [('lgb', 'LGB_MAE'), ('lgb_huber', 'LGB_Huber'), ('xgb', 'XGB'), ('cb', 'CatBoost')]:
    t1 = time.time()
    print(f"  raw_{mname}...", end='', flush=True)
    oof_raw, tpred_raw = train_fold(mtype, SEED, y, X, X_test, folds, sw=sample_weight)
    raw_oofs[mname] = oof_raw
    raw_tests[mname] = tpred_raw
    mae = mean_absolute_error(y, oof_raw)
    print(f" OOF MAE: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

print(f"  Phase 2 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Ensemble (Pre-Pseudo)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n3. Pre-Pseudo 앙상블...", flush=True)

# log 모델 앙상블
log_names = list(log_oofs.keys())
log_oof_list = [log_oofs[n] for n in log_names]
log_test_list = [log_tests[n] for n in log_names]

def optimize_weights(oof_list, y_true):
    n = len(oof_list)
    def obj(w):
        w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
        return mean_absolute_error(y_true, sum(wi*p for wi, p in zip(w, oof_list)))
    # maxiter 조정: 모델 수에 비례
    maxiter = min(5000 * n, 30000)
    res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead', options={'maxiter': maxiter})
    w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
    return w, res.fun

# log ensemble
log_w, log_mae = optimize_weights(log_oof_list, y)
print(f"  Log 앙상블 MAE: {log_mae:.4f}")
for n, w in zip(log_names, log_w):
    if w > 0.01: print(f"    {n}: {w:.3f}")

# raw ensemble
raw_names = list(raw_oofs.keys())
raw_oof_list = [raw_oofs[n] for n in raw_names]
raw_test_list = [raw_tests[n] for n in raw_names]
raw_w, raw_mae = optimize_weights(raw_oof_list, y)
print(f"  Raw 앙상블 MAE: {raw_mae:.4f}")

# mixed ensemble (log + raw = 11 models)
all_names = [f'log_{n}' for n in log_names] + [f'raw_{n}' for n in raw_names]
all_oof_list = log_oof_list + raw_oof_list
all_test_list = log_test_list + raw_test_list
mix_w, mix_mae = optimize_weights(all_oof_list, y)
print(f"  Mixed 앙상블 MAE: {mix_mae:.4f}")
for n, w in zip(all_names, mix_w):
    if w > 0.01: print(f"    {n}: {w:.3f}")

# pre-pseudo best
pre_best_mae = min(log_mae, raw_mae, mix_mae)
if pre_best_mae == mix_mae:
    pre_pred = np.clip(sum(wi*p for wi, p in zip(mix_w, all_test_list)), 0, None)
    pre_name = 'mixed'
elif pre_best_mae == log_mae:
    pre_pred = np.clip(sum(wi*p for wi, p in zip(log_w, log_test_list)), 0, None)
    pre_name = 'log'
else:
    pre_pred = np.clip(sum(wi*p for wi, p in zip(raw_w, raw_test_list)), 0, None)
    pre_name = 'raw'

print(f"  → Pre-Pseudo Best: {pre_name} (MAE: {pre_best_mae:.4f})")

# Pre-pseudo submission 저장
pd.DataFrame({'ID': test_fe['ID'], TARGET: pre_pred}).to_csv('./submission_v22_pre.csv', index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Pseudo-Labeling
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. Pseudo-Labeling...", flush=True)

# pseudo labels = pre-pseudo ensemble 예측
pseudo_target = pre_pred.copy()
pseudo_target_log = np.log1p(pseudo_target)

# test 데이터에 pseudo target 추가
test_pseudo = test_fe.copy()
test_pseudo[TARGET] = pseudo_target

# train + pseudo test 합치기
combined = pd.concat([train_fe, test_pseudo], ignore_index=True)
combined_y = combined[TARGET].values
combined_y_log = np.log1p(combined_y)

X_combined = combined[feature_cols]
n_train = len(train_fe)
n_test = len(test_fe)

# sample weights: train=1.0 (극단값 1.5), pseudo=0.3
combined_sw = np.ones(len(combined))
combined_sw[:n_train] = sample_weight  # 기존 train weights
combined_sw[n_train:] = 0.3  # pseudo labels

# 재학습: fold에서 validation은 기존 train 데이터만
# training에는 train_fold + pseudo 전체 포함
print("  Pseudo-label 모델 학습...", flush=True)

pl_oofs = {}
pl_tests = {}

for mtype, mname in [('lgb', 'LGB_MAE'), ('lgb_huber', 'LGB_Huber'), ('xgb', 'XGB'), ('cb', 'CatBoost')]:
    t1 = time.time()
    print(f"  PL_{mname}...", end='', flush=True)

    oof = np.zeros(n_train)
    tpred = np.zeros(n_test)

    for fold_idx, (orig_tr_idx, orig_val_idx) in enumerate(folds):
        # training: original train fold + ALL pseudo test
        pseudo_idx = np.arange(n_train, n_train + n_test)
        tr_idx = np.concatenate([orig_tr_idx, pseudo_idx])
        val_idx = orig_val_idx  # validation은 original train만

        tr_target = combined_y_log[tr_idx]
        val_target = combined_y_log[val_idx]
        tr_w = combined_sw[tr_idx]

        if mtype == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X_combined.iloc[tr_idx], tr_target, sample_weight=tr_w,
                      eval_set=[(X_combined.iloc[val_idx], val_target)],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X_combined.iloc[tr_idx], tr_target, sample_weight=tr_w,
                      eval_set=[(X_combined.iloc[val_idx], val_target)],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X_combined.iloc[tr_idx], tr_target, sample_weight=tr_w,
                      eval_set=[(X_combined.iloc[val_idx], val_target)], verbose=0)
        elif mtype == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=200)
            model.fit(X_combined.iloc[tr_idx], tr_target, sample_weight=tr_w,
                      eval_set=(X_combined.iloc[val_idx], val_target), verbose=0)

        oof[val_idx] = np.expm1(model.predict(X_combined.iloc[val_idx]))
        # test prediction: X_test 사용 (combined에서 n_train: 위치)
        tpred += np.expm1(model.predict(X_combined.iloc[n_train:])) / N_SPLITS

    pl_oofs[mname] = oof
    pl_tests[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF MAE: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

print(f"  Phase 4 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 5: Final Ensemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n5. 최종 앙상블...", flush=True)

# PL ensemble
pl_names = list(pl_oofs.keys())
pl_oof_list = [pl_oofs[n] for n in pl_names]
pl_test_list = [pl_tests[n] for n in pl_names]
pl_w, pl_mae = optimize_weights(pl_oof_list, y)
print(f"  PL 앙상블 MAE: {pl_mae:.4f}")
for n, w in zip(pl_names, pl_w):
    if w > 0.01: print(f"    PL_{n}: {w:.3f}")

# Grand ensemble: all models (log + raw + quantile + PL = 15 models)
grand_names = ([f'log_{n}' for n in log_names] +
               [f'raw_{n}' for n in raw_names] +
               [f'pl_{n}' for n in pl_names])
grand_oof_list = log_oof_list + raw_oof_list + pl_oof_list
grand_test_list = log_test_list + raw_test_list + pl_test_list
grand_w, grand_mae = optimize_weights(grand_oof_list, y)
print(f"  Grand 앙상블 MAE: {grand_mae:.4f}")
for n, w in zip(grand_names, grand_w):
    if w > 0.01: print(f"    {n}: {w:.3f}")

# 최적 선택
results = {
    'pre_log': (log_mae, log_w, log_test_list),
    'pre_raw': (raw_mae, raw_w, raw_test_list),
    'pre_mix': (mix_mae, mix_w, all_test_list),
    'pl': (pl_mae, pl_w, pl_test_list),
    'grand': (grand_mae, grand_w, grand_test_list),
}

best_key = min(results, key=lambda k: results[k][0])
best_mae_val, best_w, best_tests = results[best_key]
final_pred = np.clip(sum(wi*p for wi, p in zip(best_w, best_tests)), 0, None)

# 저장
pd.DataFrame({'ID': test_fe['ID'], TARGET: final_pred}).to_csv('./submission_v22.csv', index=False)

# PL만 따로도 저장
pl_pred = np.clip(sum(wi*p for wi, p in zip(pl_w, pl_test_list)), 0, None)
pd.DataFrame({'ID': test_fe['ID'], TARGET: pl_pred}).to_csv('./submission_v22_pl.csv', index=False)

print(f"\n{'='*60}")
print(f"v22 완료!")
print(f"  피처 수: {len(feature_cols)}")
print(f"  Pre-Pseudo Best ({pre_name}): OOF MAE {pre_best_mae:.4f}")
print(f"  PL Ensemble: OOF MAE {pl_mae:.4f}")
print(f"  Grand Ensemble: OOF MAE {grand_mae:.4f}")
print(f"  → 최종 선택: {best_key} (OOF MAE: {best_mae_val:.4f})")
print(f"  v14 대비: {8.736 - best_mae_val:+.4f}")
print(f"  submission_v22.csv ({best_key}) 저장")
print(f"  submission_v22_pl.csv 저장")
print(f"  submission_v22_pre.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
