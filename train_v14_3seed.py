"""
v14 3seed: v14 기반 + 나머지 2 seeds 추가
- checkpoint: seed 42는 이미 완료 → 123, 2024만 학습
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEEDS = [42, 123, 2024]
CKPT_DIR = './results/v14_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

print("=== v14 3-seed ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

print("1. 피처 엔지니어링...", flush=True)

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

print("\n2. 학습 (3 seeds, checkpoint)...", flush=True)

def train_model(model_type, seed, target, is_log=False):
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in folds:
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=seed, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])], verbose=0)
        elif model_type == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=seed, verbose=0, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      eval_set=(X.iloc[val_idx], target.iloc[val_idx]), verbose=0)
        elif model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    if is_log:
        oof = np.expm1(oof)
        tpred = np.expm1(tpred)
    return oof, tpred

configs = [
    ('lgb', 'raw_LGB_MAE', y, False),
    ('lgb_huber', 'raw_LGB_Huber', y, False),
    ('xgb', 'raw_XGB', y, False),
    ('cb', 'raw_CatBoost', y, False),
    ('lgb', 'log_LGB_MAE', y_log, True),
    ('lgb_huber', 'log_LGB_Huber', y_log, True),
    ('xgb', 'log_XGB', y_log, True),
    ('cb', 'log_CatBoost', y_log, True),
]

all_oofs = {}
all_tests = {}

for mtype, mname, target, is_log in configs:
    print(f"\n  {mname} (3 seeds)...", flush=True)
    oofs, tests = [], []
    for seed in SEEDS:
        ckpt_oof = f'{CKPT_DIR}/{mname}_s{seed}_oof.npy'
        ckpt_test = f'{CKPT_DIR}/{mname}_s{seed}_test.npy'
        if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
            print(f"    Seed {seed}... [checkpoint]", flush=True)
            o, t = np.load(ckpt_oof), np.load(ckpt_test)
        else:
            print(f"    Seed {seed}... [학습]", flush=True)
            o, t = train_model(mtype, seed, target, is_log)
            np.save(ckpt_oof, o); np.save(ckpt_test, t)
            print(f"    → 저장", flush=True)
        oofs.append(o); tests.append(t)
    all_oofs[mname] = np.mean(oofs, axis=0)
    all_tests[mname] = np.mean(tests, axis=0)
    print(f"  {mname} OOF MAE: {mean_absolute_error(y, all_oofs[mname]):.4f}", flush=True)

# 앙상블
print("\n3. 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[0.125]*8, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
best_mae = mean_absolute_error(y, sum(wi*p for wi,p in zip(bw, oof_list)))

print(f"  혼합 앙상블 OOF MAE: {best_mae:.4f}")
for n, w in zip(names, bw):
    if w > 0.01: print(f"    {n}: {w:.3f}")

final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: final_pred}).to_csv('./submission_v14_3seed.csv', index=False)

print(f"\n{'='*60}")
print(f"v14 3-seed 완료!")
print(f"  앙상블 OOF MAE: {best_mae:.4f}")
print(f"  submission_v14_3seed.csv 저장")
print(f"{'='*60}", flush=True)
