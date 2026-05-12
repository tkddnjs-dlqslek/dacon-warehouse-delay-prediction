"""
v22 PL Only: v22_pre.csv의 pseudo label로 4모델 재학습
- v22_pre.csv 이미 존재 (LB 10.08)
- Phase 1: 피처 생성 (v22 동일)
- Phase 2: PL 4모델 학습 (LGB_MAE, LGB_Huber, XGB, CatBoost)
- Phase 3: 앙상블 → submission_v22_pl.csv
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
print("=== v22 PL Only: Pseudo-Label 4모델 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')
pseudo_labels = pd.read_csv('./submission_v22_pre.csv')
print(f"  데이터 로드 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Feature Engineering (v22 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n1. 피처 엔지니어링...", flush=True)

def engineer_features_v22(df, layout_df):
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
        # forward-looking
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_lead2'] = g.shift(-2)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

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
print(f"  피처 수: {len(feature_cols)}", flush=True)

X = train_fe[feature_cols]
groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups=groups))

# sample weights
sample_weight = np.ones(len(y))
sample_weight[y > 100] = 1.5
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Pseudo-Label 합치기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n2. Pseudo-Label 데이터 준비...", flush=True)

# test_fe에 pseudo label 붙이기
# pseudo_labels는 원래 test 순서 (ID 기준), test_fe는 sort된 순서
# ID로 매칭
test_fe_with_pl = test_fe.merge(pseudo_labels[['ID', TARGET]], on='ID', how='left')
test_fe_with_pl = test_fe_with_pl.sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)

# combined
combined = pd.concat([train_fe, test_fe_with_pl], ignore_index=True)
combined_y_log = np.log1p(combined[TARGET].values)
X_combined = combined[feature_cols]

n_train = len(train_fe)
n_test = len(test_fe)

# sample weights
combined_sw = np.ones(len(combined))
combined_sw[:n_train] = sample_weight
combined_sw[n_train:] = 0.3
print(f"  Combined: {len(combined)} rows (train {n_train} + pseudo {n_test})", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: PL 모델 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n3. PL 모델 학습...", flush=True)

pl_oofs = {}
pl_tests = {}

models = [
    ('lgb', 'LGB_MAE'),
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
]

for mtype, mname in models:
    t1 = time.time()
    print(f"  PL_{mname}...", end='', flush=True)

    oof = np.zeros(n_train)
    tpred = np.zeros(n_test)

    for fold_idx, (orig_tr_idx, orig_val_idx) in enumerate(folds):
        pseudo_idx = np.arange(n_train, n_train + n_test)
        tr_idx = np.concatenate([orig_tr_idx, pseudo_idx])
        val_idx = orig_val_idx

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
        tpred += np.expm1(model.predict(X_combined.iloc[n_train:])) / N_SPLITS

    pl_oofs[mname] = oof
    pl_tests[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF MAE: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. PL 앙상블...", flush=True)

pl_names = list(pl_oofs.keys())
pl_oof_list = [pl_oofs[n] for n in pl_names]
pl_test_list = [pl_tests[n] for n in pl_names]

def optimize_weights(oof_list, y_true):
    n = len(oof_list)
    def obj(w):
        w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
        return mean_absolute_error(y_true, sum(wi*p for wi, p in zip(w, oof_list)))
    res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead', options={'maxiter': 20000})
    w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
    return w, res.fun

pl_w, pl_mae = optimize_weights(pl_oof_list, y)
print(f"  PL 앙상블 OOF MAE: {pl_mae:.4f}")
for n, w in zip(pl_names, pl_w):
    print(f"    PL_{n}: {w:.3f}")

# 저장
pl_pred = np.clip(sum(wi*p for wi, p in zip(pl_w, pl_test_list)), 0, None)
pd.DataFrame({'ID': test_fe['ID'], TARGET: pl_pred}).to_csv('./submission_v22_pl.csv', index=False)

# 개별 모델도 저장 (비교용)
for mname in pl_names:
    pred = np.clip(pl_tests[mname], 0, None)
    pd.DataFrame({'ID': test_fe['ID'], TARGET: pred}).to_csv(f'./submission_v22_pl_{mname}.csv', index=False)

print(f"\n{'='*60}")
print(f"v22 PL Only 완료!")
print(f"  PL 앙상블 OOF MAE: {pl_mae:.4f}")
print(f"  v14 대비: {8.736 - pl_mae:+.4f}")
print(f"  v22_pre 대비: {8.602 - pl_mae:+.4f}")
print(f"  submission_v22_pl.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
