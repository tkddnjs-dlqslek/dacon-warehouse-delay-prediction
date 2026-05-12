"""
v28: Forward-Looking 제거 실험 + 피처 축소 비교
  - Phase 1에서 5가지 피처셋 1-fold 비교
  - 최적 피처셋으로 5-fold 학습
  - 모델별 csv 저장 (체크포인트)
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
import pickle
import json
import os

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)

t0 = time.time()
print("=== v28: Forward-Looking 제거 실험 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
v23_imp = v23_phase1['importance']
print(f"  v23 피처: {len(v23_selected)}개", flush=True)

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
    corr_remove = ['battery_mean_rmean3', 'charge_queue_length_rmean3', 'battery_mean_rmean5',
                   'charge_queue_length_rmean5', 'pack_utilization_rmean5', 'battery_mean_lag1',
                   'charge_queue_length_lag1', 'congestion_score_rmean3', 'order_inflow_15m_cummean',
                   'robot_utilization_rmean5', 'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
                   'battery_risk', 'congestion_score_rmean5', 'pack_utilization_rmean3',
                   'order_inflow_15m_rmean3', 'charge_queue_length_lag2', 'blocked_path_15m_rmean5']
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')
    return df

train_fe = engineer_features_v23(train, layout)
test_fe = engineer_features_v23(test, layout)
y = train_fe[TARGET].values
y_log = np.log1p(y)
groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 피처셋 비교 (1-fold)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] 피처셋 비교 (fold 0)...", flush=True)

# 피처 분류
available = [f for f in v23_selected if f in train_fe.columns]
lead_feats = [f for f in available if '_lead' in f or '_diff_lead' in f]
lag_feats = [f for f in available if ('_lag' in f or f.endswith('_diff1') or '_rmean' in f or '_rstd' in f or '_cummean' in f) and '_lead' not in f]

# 5가지 피처셋
feat_sets = {
    'full_150': available,
    'no_lead_139': [f for f in available if f not in lead_feats],
    'no_lead_lag_128': [f for f in available if f not in lead_feats + lag_feats],
    'top_120': v23_imp[v23_imp['feature'].isin(available)].head(120)['feature'].tolist(),
    'top_100': v23_imp[v23_imp['feature'].isin(available)].head(100)['feature'].tolist(),
}

tr_idx, val_idx = folds[0]
results = {}

for name, feats in feat_sets.items():
    m = lgb.LGBMRegressor(
        objective='huber', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(train_fe[feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(train_fe[feats].iloc[val_idx])), 0, None)
    mae = mean_absolute_error(y[val_idx], pred)
    results[name] = mae
    print(f"  {name:20s} ({len(feats):3d} feats): fold0 MAE={mae:.4f}", flush=True)

# 최적 선택
best_name = min(results, key=results.get)
best_feats = feat_sets[best_name]
print(f"\n  -> 최적: {best_name} (MAE={results[best_name]:.4f})", flush=True)

pickle.dump({
    'results': results,
    'best_name': best_name,
    'best_feats': best_feats,
    'feat_sets': {k: len(v) for k, v in feat_sets.items()},
}, open(f'{RESULT_DIR}/v28_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 최적 피처셋으로 5-fold 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n[Phase 2] Training ({best_name}, {len(best_feats)} feats)...", flush=True)

X = train_fe[best_feats]
X_test = test_fe[best_feats]

models_config = [
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
]

oofs = {}
tests_dict = {}

for mtype, mname in models_config:
    t1 = time.time()
    print(f"  {mname}...", end='', flush=True)

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
    oofs[mname] = oof
    tests_dict[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # 모델별 csv 체크포인트
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
        f'./submission_v28_{mname}.csv', index=False)
    print(f"    submission_v28_{mname}.csv 저장", flush=True)

    # 중간 pickle
    pickle.dump({'oofs': oofs, 'tests': tests_dict}, open(f'{RESULT_DIR}/v28_phase2.pkl', 'wb'))

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
    res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead', options={'maxiter': 20000})
    w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
    return w, res.fun

w, ens_mae = optimize_weights(oof_list, y)
ens_pred = np.clip(sum(wi*p for wi, p in zip(w, test_list)), 0, None)

print(f"  앙상블 OOF MAE: {ens_mae:.4f}")
for n, wi in zip(names, w):
    print(f"    {n}: {wi:.3f}")

pd.DataFrame({'ID': test_fe['ID'], TARGET: ens_pred}).to_csv(
    './submission_v28.csv', index=False)

# v23과 블렌딩
print("\n  v23 블렌딩...", flush=True)
try:
    v23_s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
    v23_w = v23_s42['weights']
    v23_names = list(v23_s42['tests'].keys())
    v23_test_pred = np.clip(sum(wi*p for wi, p in zip(v23_w, [v23_s42['tests'][n] for n in v23_names])), 0, None)

    for ratio in [0.3, 0.5, 0.7]:
        blend = ratio * ens_pred + (1 - ratio) * v23_test_pred
        pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
            f'./submission_v28_v23blend_{int(ratio*100)}.csv', index=False)
        print(f"    v28:{ratio:.0%} + v23:{1-ratio:.0%} 저장", flush=True)
except Exception as e:
    print(f"    v23 블렌딩 실패: {e}")

pickle.dump({
    'oofs': oofs, 'tests': tests_dict,
    'weights': w, 'ens_mae': ens_mae,
    'best_name': best_name, 'best_feats': best_feats,
    'phase1_results': results,
}, open(f'{RESULT_DIR}/v28_final.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"v28 완료!")
print(f"  선택 피처셋: {best_name} ({len(best_feats)}개)")
print(f"  Phase 1 비교:")
for name, mae in sorted(results.items(), key=lambda x: x[1]):
    marker = " <-- best" if name == best_name else ""
    print(f"    {name:20s}: fold0 MAE={mae:.4f}{marker}")
print(f"  5-fold 결과:")
for n in names:
    print(f"    {n}: OOF={mean_absolute_error(y, oofs[n]):.4f}")
print(f"  앙상블 OOF: {ens_mae:.4f}")
print(f"  v23 대비: {8.5787 - ens_mae:+.4f}")
print(f"  submission_v28.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
