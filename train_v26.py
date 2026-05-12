"""
v26: v23 피처 150개 + HP 튜닝 + 모델 다양성 + Target Transform
  A. Optuna HP 튜닝 (LGB_Huber)
  B. 모델 다양성 (DART, 다른 colsample)
  C. Target transform 다양화 (log1p, sqrt, power0.25)
  - seed=42 단독
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
import optuna
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)

t0 = time.time()
print("=== v26: HP 튜닝 + 모델 다양성 + Transform 다양화 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# v23 피처 리스트 로드
v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
print(f"  v23 피처 로드: {len(v23_selected)}개", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering (v23 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링 (v23 동일)...", flush=True)

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
y_log = np.log1p(y)
y_sqrt = np.sqrt(y)
y_pow = np.power(y + 1, 0.25)

# v23 피처 중 존재하는 것만
feature_cols = [f for f in v23_selected if f in train_fe.columns]

# extreme_prob 추가
exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_cols = [c for c in train_fe.columns if c not in exclude]

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

# extreme_prob
print("  극단값 분류기...", flush=True)
extreme_oof = np.zeros(len(train_fe))
extreme_test = np.zeros(len(test_fe))
y_cls = (y > 50).astype(int)
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        random_state=SEED, verbose=-1, n_jobs=4)
    clf.fit(train_fe[all_cols].iloc[tr_idx], y_cls[tr_idx],
            eval_set=[(train_fe[all_cols].iloc[val_idx], y_cls[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    extreme_oof[val_idx] = clf.predict_proba(train_fe[all_cols].iloc[val_idx])[:, 1]
    extreme_test += clf.predict_proba(test_fe[all_cols])[:, 1] / N_SPLITS

train_fe['extreme_prob'] = extreme_oof
test_fe['extreme_prob'] = extreme_test
if 'extreme_prob' not in feature_cols:
    feature_cols.append('extreme_prob')

print(f"  피처 수: {len(feature_cols)}")
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

X = train_fe[feature_cols]
X_test = test_fe[feature_cols]

pickle.dump({'feature_cols': feature_cols}, open(f'{RESULT_DIR}/v26_phase0.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Optuna HP 튜닝 (LGB_Huber)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Optuna HP 튜닝 (LGB_Huber, 50 trials)...", flush=True)

def objective(trial):
    params = {
        'objective': 'huber',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 127),
        'max_depth': trial.suggest_int('max_depth', 5, 10),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.9),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 5.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 5.0, log=True),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }

    oof = np.zeros(len(X))
    for tr_idx, val_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(X.iloc[tr_idx], y_log[tr_idx],
                  eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))

    oof = np.clip(oof, 0, None)
    return mean_absolute_error(y, oof)

study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=50, show_progress_bar=False,
               callbacks=[lambda study, trial: print(
                   f"    Trial {trial.number}: MAE={trial.value:.4f} "
                   f"(best={study.best_value:.4f})", flush=True)])

best_params = study.best_params
best_optuna_mae = study.best_value
print(f"\n  Optuna best MAE: {best_optuna_mae:.4f}")
print(f"  Best params: {best_params}")

# v23 기본 HP와 비교
print(f"  v23 기본 HP OOF (LGB_Huber): 8.6118")
print(f"  Optuna 개선: {8.6118 - best_optuna_mae:+.4f}")

pickle.dump({
    'best_params': best_params,
    'best_mae': best_optuna_mae,
    'study': study,
}, open(f'{RESULT_DIR}/v26_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Training (다양한 모델)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Training...", flush=True)

oofs = {}
tests_dict = {}

def train_5fold(model_fn, target, inv_fn, name):
    """5-fold 학습 헬퍼"""
    t1 = time.time()
    print(f"  {name}...", end='', flush=True)
    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))
    for tr_idx, val_idx in folds:
        model = model_fn()
        if hasattr(model, 'fit'):
            if isinstance(model, CatBoostRegressor):
                model.fit(X.iloc[tr_idx], target[tr_idx],
                          eval_set=(X.iloc[val_idx], target[val_idx]), verbose=0)
            elif isinstance(model, xgb.XGBRegressor):
                model.fit(X.iloc[tr_idx], target[tr_idx],
                          eval_set=[(X.iloc[val_idx], target[val_idx])], verbose=0)
            else:
                model.fit(X.iloc[tr_idx], target[tr_idx],
                          eval_set=[(X.iloc[val_idx], target[val_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = inv_fn(model.predict(X.iloc[val_idx]))
        tpred += inv_fn(model.predict(X_test)) / N_SPLITS
    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    mae = mean_absolute_error(y, oof)
    print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)
    return oof, tpred

# 1. Tuned LGB_Huber (log target)
oofs['Tuned_Huber'], tests_dict['Tuned_Huber'] = train_5fold(
    lambda: lgb.LGBMRegressor(
        objective='huber', n_estimators=5000,
        **{k: v for k, v in best_params.items()},
        random_state=SEED, verbose=-1, n_jobs=4),
    y_log, np.expm1, 'Tuned_Huber(log)')

# 2. 기존 HP LGB_Huber (baseline 비교)
oofs['Base_Huber'], tests_dict['Base_Huber'] = train_5fold(
    lambda: lgb.LGBMRegressor(
        objective='huber', n_estimators=5000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=4),
    y_log, np.expm1, 'Base_Huber(log)')

# 3. XGB (기존 HP, log)
oofs['XGB'], tests_dict['XGB'] = train_5fold(
    lambda: xgb.XGBRegressor(
        objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
        max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
        random_state=SEED, verbosity=0, n_jobs=4, early_stopping_rounds=200),
    y_log, np.expm1, 'XGB(log)')

# 4. CatBoost (기존 HP, log)
oofs['CatBoost'], tests_dict['CatBoost'] = train_5fold(
    lambda: CatBoostRegressor(
        loss_function='MAE', eval_metric='MAE',
        iterations=5000, learning_rate=0.03, depth=7,
        l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
        random_seed=SEED, verbose=0, early_stopping_rounds=200, thread_count=4, task_type='CPU'),
    y_log, np.expm1, 'CatBoost(log)')

# 5. LGB DART (다양성)
oofs['DART'], tests_dict['DART'] = train_5fold(
    lambda: lgb.LGBMRegressor(
        boosting_type='dart', objective='mae', n_estimators=2000,
        learning_rate=0.03, num_leaves=63, max_depth=8,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, drop_rate=0.1,
        random_state=SEED, verbose=-1, n_jobs=4),
    y_log, np.expm1, 'DART(log)')

# 6. Tuned Huber + sqrt target
oofs['Tuned_sqrt'], tests_dict['Tuned_sqrt'] = train_5fold(
    lambda: lgb.LGBMRegressor(
        objective='huber', n_estimators=5000,
        **{k: v for k, v in best_params.items()},
        random_state=SEED, verbose=-1, n_jobs=4),
    y_sqrt, lambda x: np.clip(x, 0, None)**2, 'Tuned_Huber(sqrt)')

# 7. Tuned Huber + power0.25 target
oofs['Tuned_pow'], tests_dict['Tuned_pow'] = train_5fold(
    lambda: lgb.LGBMRegressor(
        objective='huber', n_estimators=5000,
        **{k: v for k, v in best_params.items()},
        random_state=SEED, verbose=-1, n_jobs=4),
    y_pow, lambda x: np.clip(x, 0, None)**4 - 1, 'Tuned_Huber(pow0.25)')

pickle.dump({'oofs': oofs, 'tests': tests_dict}, open(f'{RESULT_DIR}/v26_phase2.pkl', 'wb'))
print(f"  Phase 2 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Ensemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] 앙상블...", flush=True)

names = list(oofs.keys())
oof_list = [oofs[n] for n in names]
test_list = [tests_dict[n] for n in names]

def optimize_weights(oofs_list, y_true):
    n = len(oofs_list)
    def obj(w):
        w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
        return mean_absolute_error(y_true, sum(wi*p for wi, p in zip(w, oofs_list)))
    res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead',
                   options={'maxiter': min(5000*n, 30000)})
    w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
    return w, res.fun

w, ens_mae = optimize_weights(oof_list, y)
ens_pred = np.clip(sum(wi*p for wi, p in zip(w, test_list)), 0, None)

print(f"  앙상블 OOF MAE: {ens_mae:.4f}")
for n, wi in zip(names, w):
    if wi > 0.01:
        print(f"    {n}: {wi:.3f}")

pd.DataFrame({'ID': test_fe['ID'], TARGET: ens_pred}).to_csv(
    './submission_v26.csv', index=False)

# v23 모델과도 합쳐보기 (v23 pickle에서 test predictions 로드)
print("\n  v23 + v26 혼합 앙상블 시도...", flush=True)
try:
    v23_s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
    v23_names = [f'v23_{n}' for n in v23_s42['oofs'].keys()]
    v23_oof_list = list(v23_s42['oofs'].values())
    v23_test_list = list(v23_s42['tests'].values())

    grand_names = names + v23_names
    grand_oof = oof_list + v23_oof_list
    grand_test = test_list + v23_test_list

    gw, grand_mae = optimize_weights(grand_oof, y)
    grand_pred = np.clip(sum(wi*p for wi, p in zip(gw, grand_test)), 0, None)

    print(f"  v26+v23 Grand 앙상블 OOF MAE: {grand_mae:.4f}")
    for n, wi in zip(grand_names, gw):
        if wi > 0.01:
            print(f"    {n}: {wi:.3f}")

    pd.DataFrame({'ID': test_fe['ID'], TARGET: grand_pred}).to_csv(
        './submission_v26_grand.csv', index=False)
except Exception as e:
    print(f"  v23 혼합 실패: {e}")

pickle.dump({
    'oofs': oofs, 'tests': tests_dict,
    'weights': w, 'ens_mae': ens_mae,
    'best_params': best_params,
}, open(f'{RESULT_DIR}/v26_final.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"v26 완료!")
print(f"  피처: {len(feature_cols)}개")
print(f"  Optuna best HP MAE: {best_optuna_mae:.4f}")
for n in names:
    print(f"  {n} OOF: {mean_absolute_error(y, oofs[n]):.4f}")
print(f"  v26 앙상블 OOF: {ens_mae:.4f}")
print(f"  v23 대비: {8.5787 - ens_mae:+.4f}")
print(f"  submission_v26.csv 저장")
print(f"  submission_v26_grand.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
