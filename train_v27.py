"""
v27: 진짜 2-Stage 모델
  Stage 1: 고값(>=60) 분류기
  Stage 2a: 저값 전용 모델 (log1p, target<60만 학습)
  Stage 2b: 고값 전용 모델 (sqrt, target>=60만 학습)
  최종: soft blending = (1-p)*pred_2a + p*pred_2b
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)
THRESHOLD = 60  # 고값 기준

t0 = time.time()
print("=== v27: 2-Stage Model (Low/High Split) ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
print(f"  v23 피처 로드: {len(v23_selected)}개", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering (v23 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링...", flush=True)

def engineer_features_v23(df, layout_df):
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

    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_sc_mean'] = g.transform('mean')
        df[f'{col}_sc_std'] = g.transform('std')
        df[f'{col}_sc_max'] = g.transform('max')
        df[f'{col}_sc_min'] = g.transform('min')
        df[f'{col}_sc_range'] = df[f'{col}_sc_max'] - df[f'{col}_sc_min']
        df[f'{col}_sc_rank'] = g.rank(pct=True)
        df[f'{col}_sc_dev'] = df[col] - df[f'{col}_sc_mean']

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

train_fe = engineer_features_v23(train, layout)
test_fe = engineer_features_v23(test, layout)

y = train_fe[TARGET].values
feature_cols = [f for f in v23_selected if f in train_fe.columns]

# extreme_prob 추가
exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_cols = [c for c in train_fe.columns if c not in exclude]

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

print(f"  피처: {len(feature_cols)}개")
print(f"  target > {THRESHOLD}: {(y >= THRESHOLD).sum()}개 ({(y >= THRESHOLD).mean()*100:.1f}%)")
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

pickle.dump({'feature_cols': feature_cols, 'folds': folds}, open(f'{RESULT_DIR}/v27_phase0.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Stage 1 — 고값 분류기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Stage 1: 고값 분류기...", flush=True)

X = train_fe[feature_cols]
X_test = test_fe[feature_cols]
y_cls = (y >= THRESHOLD).astype(int)

high_prob_oof = np.zeros(len(train_fe))
high_prob_test = np.zeros(len(test_fe))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    clf.fit(X.iloc[tr_idx], y_cls[tr_idx],
            eval_set=[(X.iloc[val_idx], y_cls[val_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    high_prob_oof[val_idx] = clf.predict_proba(X.iloc[val_idx])[:, 1]
    high_prob_test += clf.predict_proba(X_test)[:, 1] / N_SPLITS

auc = roc_auc_score(y_cls, high_prob_oof)
print(f"  AUC (>={THRESHOLD}): {auc:.4f}", flush=True)

pickle.dump({'high_prob_oof': high_prob_oof, 'high_prob_test': high_prob_test, 'auc': auc},
            open(f'{RESULT_DIR}/v27_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2a: 저값 모델 (target < THRESHOLD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2a] 저값 모델 (log1p, target < 60)...", flush=True)

y_log = np.log1p(y)
low_mask = y < THRESHOLD

models_2a = [
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
]

oofs_2a = {}
tests_2a = {}

for mtype, mname in models_2a:
    t1 = time.time()
    print(f"  2a_{mname}...", end='', flush=True)

    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        # 저값만 학습
        tr_low = tr_idx[low_mask[tr_idx]]

        if mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_low], y_log[tr_low],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X.iloc[tr_low], y_log[tr_low],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])], verbose=0)
        elif mtype == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=200)
            model.fit(X.iloc[tr_low], y_log[tr_low],
                      eval_set=(X.iloc[val_idx], y_log[val_idx]), verbose=0)

        oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))
        tpred += np.expm1(model.predict(X_test)) / N_SPLITS

    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    oofs_2a[mname] = oof
    tests_2a[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

# 2a 앙상블
names_2a = list(oofs_2a.keys())
oof_list_2a = [oofs_2a[n] for n in names_2a]
test_list_2a = [tests_2a[n] for n in names_2a]

def optimize_weights(oofs_list, y_true):
    n = len(oofs_list)
    def obj(w):
        w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
        return mean_absolute_error(y_true, sum(wi*p for wi, p in zip(w, oofs_list)))
    res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead', options={'maxiter': 20000})
    w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
    return w, res.fun

w_2a, mae_2a = optimize_weights(oof_list_2a, y)
pred_2a_oof = np.clip(sum(wi*p for wi, p in zip(w_2a, oof_list_2a)), 0, None)
pred_2a_test = np.clip(sum(wi*p for wi, p in zip(w_2a, test_list_2a)), 0, None)
print(f"  2a 앙상블 OOF MAE: {mae_2a:.4f}")

pickle.dump({'oofs': oofs_2a, 'tests': tests_2a, 'weights': w_2a, 'pred_oof': pred_2a_oof, 'pred_test': pred_2a_test},
            open(f'{RESULT_DIR}/v27_phase2a.pkl', 'wb'))
print(f"  Phase 2a 완료 ({time.time()-t0:.0f}s)", flush=True)

# v23과 동일한 전체 모델도 같이 돌림 (비교용 baseline)
print("\n[Phase 2-base] v23 동일 모델 (전체 데이터, 비교용)...", flush=True)
oofs_base = {}
tests_base = {}

for mtype, mname in models_2a:
    t1 = time.time()
    print(f"  base_{mname}...", end='', flush=True)

    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        if mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])], verbose=0)
        elif mtype == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=(X.iloc[val_idx], y_log[val_idx]), verbose=0)

        oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))
        tpred += np.expm1(model.predict(X_test)) / N_SPLITS

    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    oofs_base[mname] = oof
    tests_base[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

w_base, mae_base = optimize_weights(list(oofs_base.values()), y)
pred_base_oof = np.clip(sum(wi*p for wi, p in zip(w_base, oofs_base.values())), 0, None)
pred_base_test = np.clip(sum(wi*p for wi, p in zip(w_base, tests_base.values())), 0, None)
print(f"  base 앙상블 OOF MAE: {mae_base:.4f}")

# base submission 저장 (안전장치 — v23과 거의 동일해야 함)
pd.DataFrame({'ID': test_fe['ID'], TARGET: pred_base_test}).to_csv(
    './submission_v27_base.csv', index=False)

pickle.dump({'oofs': oofs_base, 'tests': tests_base, 'weights': w_base},
            open(f'{RESULT_DIR}/v27_base.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2b: 고값 모델 (target >= THRESHOLD)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n[Phase 2b] 고값 모델 (sqrt, target >= {THRESHOLD})...", flush=True)

high_mask = y >= THRESHOLD
y_sqrt = np.sqrt(y)

models_2b = [
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
]

oofs_2b = {}
tests_2b = {}

for mtype, mname in models_2b:
    t1 = time.time()
    print(f"  2b_{mname}...", end='', flush=True)

    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        # 고값만 학습
        tr_high = tr_idx[high_mask[tr_idx]]

        if mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=31, max_depth=6, min_child_samples=30,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=2.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_high], y_sqrt[tr_high],
                      eval_set=[(X.iloc[val_idx], y_sqrt[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=6, min_child_weight=20, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=2.0, reg_lambda=2.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X.iloc[tr_high], y_sqrt[tr_high],
                      eval_set=[(X.iloc[val_idx], y_sqrt[val_idx])], verbose=0)

        pred_sqrt = model.predict(X.iloc[val_idx])
        oof[val_idx] = np.clip(pred_sqrt, 0, None) ** 2
        pred_sqrt_test = model.predict(X_test)
        tpred += np.clip(pred_sqrt_test, 0, None) ** 2 / N_SPLITS

    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    oofs_2b[mname] = oof
    tests_2b[mname] = tpred
    mae = mean_absolute_error(y, oof)
    mae_high = mean_absolute_error(y[high_mask], oof[high_mask])
    print(f" OOF: {mae:.4f}, High MAE: {mae_high:.4f} ({time.time()-t1:.0f}s)", flush=True)

# 2b 앙상블
names_2b = list(oofs_2b.keys())
oof_list_2b = [oofs_2b[n] for n in names_2b]
test_list_2b = [tests_2b[n] for n in names_2b]
w_2b, mae_2b = optimize_weights(oof_list_2b, y)
pred_2b_oof = np.clip(sum(wi*p for wi, p in zip(w_2b, oof_list_2b)), 0, None)
pred_2b_test = np.clip(sum(wi*p for wi, p in zip(w_2b, test_list_2b)), 0, None)
print(f"  2b 앙상블 OOF MAE: {mae_2b:.4f}")
print(f"  2b High-only MAE: {mean_absolute_error(y[high_mask], pred_2b_oof[high_mask]):.4f} (base: {mean_absolute_error(y[high_mask], pred_base_oof[high_mask]):.4f})")

pickle.dump({'oofs': oofs_2b, 'tests': tests_2b, 'weights': w_2b, 'pred_oof': pred_2b_oof, 'pred_test': pred_2b_test},
            open(f'{RESULT_DIR}/v27_phase2b.pkl', 'wb'))
print(f"  Phase 2b 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Soft Blending
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] Soft Blending...", flush=True)

# 방법 1: soft blending with high_prob
final_oof_soft = (1 - high_prob_oof) * pred_2a_oof + high_prob_oof * pred_2b_oof
final_test_soft = (1 - high_prob_test) * pred_2a_test + high_prob_test * pred_2b_test
mae_soft = mean_absolute_error(y, final_oof_soft)
print(f"  Soft blend (2a+2b): OOF MAE={mae_soft:.4f}")

# 방법 2: base + 2b blending (base는 전체 학습, 2b는 고값 전용)
final_oof_base_2b = (1 - high_prob_oof) * pred_base_oof + high_prob_oof * pred_2b_oof
final_test_base_2b = (1 - high_prob_test) * pred_base_test + high_prob_test * pred_2b_test
mae_base_2b = mean_absolute_error(y, final_oof_base_2b)
print(f"  Soft blend (base+2b): OOF MAE={mae_base_2b:.4f}")

# 방법 3: 다양한 blending weight 시도
print("\n  Blending weight 탐색:")
best_blend_mae = 999
best_blend_alpha = 0
for alpha in np.arange(0, 1.05, 0.05):
    blended = (1 - alpha * high_prob_oof) * pred_base_oof + alpha * high_prob_oof * pred_2b_oof
    mae_b = mean_absolute_error(y, blended)
    if mae_b < best_blend_mae:
        best_blend_mae = mae_b
        best_blend_alpha = alpha
    if alpha in [0, 0.25, 0.5, 0.75, 1.0]:
        print(f"    alpha={alpha:.2f}: MAE={mae_b:.4f}")

print(f"  Best alpha={best_blend_alpha:.2f}: MAE={best_blend_mae:.4f}")

# 최적 blending으로 최종 예측
final_oof = (1 - best_blend_alpha * high_prob_oof) * pred_base_oof + best_blend_alpha * high_prob_oof * pred_2b_oof
final_test = (1 - best_blend_alpha * high_prob_test) * pred_base_test + best_blend_alpha * high_prob_test * pred_2b_test
final_test = np.clip(final_test, 0, None)

pd.DataFrame({'ID': test_fe['ID'], TARGET: final_test}).to_csv(
    './submission_v27.csv', index=False)

# 방법 4: v23 pickle과 블렌딩
print("\n  v23과 블렌딩...", flush=True)
try:
    v23_s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
    v23_w = v23_s42['weights']
    v23_names = list(v23_s42['tests'].keys())
    v23_test_pred = np.clip(sum(wi*p for wi, p in zip(v23_w, [v23_s42['tests'][n] for n in v23_names])), 0, None)

    for ratio in [0.3, 0.5, 0.7]:
        blend = ratio * final_test + (1 - ratio) * v23_test_pred
        print(f"    v27:{ratio:.0%} + v23:{1-ratio:.0%} 저장")
        pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
            f'./submission_v27_v23blend_{int(ratio*100)}.csv', index=False)
except Exception as e:
    print(f"  v23 블렌딩 실패: {e}")

pickle.dump({
    'final_oof': final_oof, 'final_test': final_test,
    'best_blend_alpha': best_blend_alpha, 'best_blend_mae': best_blend_mae,
    'mae_soft': mae_soft, 'mae_base_2b': mae_base_2b, 'mae_base': mae_base,
}, open(f'{RESULT_DIR}/v27_final.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*60}")
print(f"v27 완료!")
print(f"  Stage 1 AUC (>={THRESHOLD}): {auc:.4f}")
print(f"  2a (저값) 앙상블 OOF: {mae_2a:.4f}")
print(f"  2b (고값) 앙상블 OOF: {mae_2b:.4f}")
print(f"  Base (전체) 앙상블 OOF: {mae_base:.4f}")
print(f"  Soft blend (2a+2b): {mae_soft:.4f}")
print(f"  Soft blend (base+2b): {mae_base_2b:.4f}")
print(f"  Best blend (alpha={best_blend_alpha:.2f}): {best_blend_mae:.4f}")
print(f"  v23 대비: {8.5787 - best_blend_mae:+.4f}")
print(f"")
print(f"  High-only MAE comparison:")
print(f"    base: {mean_absolute_error(y[high_mask], pred_base_oof[high_mask]):.4f}")
print(f"    2b:   {mean_absolute_error(y[high_mask], pred_2b_oof[high_mask]):.4f}")
print(f"")
print(f"  submission_v27_base.csv (전체 모델)")
print(f"  submission_v27.csv (best blend)")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
