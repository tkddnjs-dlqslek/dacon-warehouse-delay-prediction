"""
Offset Decomposition
  y = scenario_mean + row_offset
  Scenario-level model + Offset model 학습
  → mega-stacking에 추가
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
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'

t0 = time.time()
print("=== Offset Decomposition ===", flush=True)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: v23 피처 재생성 + scenario aggregation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 생성...", flush=True)

def engineer_v23(df, layout_df):
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

train_fe = engineer_v23(train, layout)
test_fe = engineer_v23(test, layout)
y = train_fe[TARGET].values
groups = train_fe['layout_id']

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
v23_feats = [f for f in v23_selected if f in train_fe.columns]

# Scenario-level aggregated data
print("  Scenario-level aggregation...", flush=True)
key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
            'battery_mean', 'fault_count_15m', 'blocked_path_15m',
            'pack_utilization', 'charge_queue_length', 'max_zone_density',
            'robot_charging', 'low_battery_ratio', 'robot_idle', 'near_collision_15m']
key_cols = [c for c in key_cols if c in train_fe.columns]

def make_scenario_df(df, has_target=True):
    agg_dict = {c: ['mean', 'std', 'max', 'min'] for c in key_cols}
    if has_target:
        agg_dict[TARGET] = 'mean'
    sc = df.groupby(['layout_id', 'scenario_id']).agg(agg_dict).reset_index()
    sc.columns = ['layout_id', 'scenario_id'] + \
                 [f'{c}_{s}' for c in key_cols for s in ['mean','std','max','min']] + \
                 (['target_mean'] if has_target else [])
    sc = sc.merge(layout, on='layout_id', how='left')
    sc = pd.get_dummies(sc, columns=['layout_type'])
    return sc

train_sc = make_scenario_df(train, has_target=True)
test_sc = make_scenario_df(test, has_target=False)

# Align columns
missing_cols = [c for c in train_sc.columns if c not in test_sc.columns]
for c in missing_cols:
    if c != 'target_mean':
        test_sc[c] = 0
test_sc = test_sc[[c for c in train_sc.columns if c != 'target_mean']]

sc_exclude = ['layout_id', 'scenario_id', 'target_mean']
sc_feats = [c for c in train_sc.columns if c not in sc_exclude]
y_sc = train_sc['target_mean'].values
X_sc = train_sc[sc_feats].fillna(0).values
X_sc_test = test_sc[sc_feats].fillna(0).values
groups_sc = train_sc['layout_id'].values

print(f"  Train scenarios: {len(train_sc)}, Test scenarios: {len(test_sc)}")
print(f"  Scenario features: {len(sc_feats)}")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))
folds_sc = list(gkf.split(train_sc, y_sc, groups=groups_sc))

print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Scenario-Level Model (3 models)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Scenario-Level Model...", flush=True)

y_sc_log = np.log1p(y_sc)
sc_oofs = {}
sc_tests = {}

for mtype, mname in [('lgb_huber', 'LGB_Huber'), ('xgb', 'XGB'), ('cb', 'CatBoost')]:
    t1 = time.time()
    print(f"  sc_{mname}...", end='', flush=True)

    oof = np.zeros(len(y_sc))
    tpred = np.zeros(len(test_sc))

    for tr_idx, val_idx in folds_sc:
        if mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=3000, learning_rate=0.03,
                num_leaves=31, max_depth=6, min_child_samples=20,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X_sc[tr_idx], y_sc_log[tr_idx],
                      eval_set=[(X_sc[val_idx], y_sc_log[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=3000, learning_rate=0.03,
                max_depth=5, min_child_weight=5, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=100)
            model.fit(X_sc[tr_idx], y_sc_log[tr_idx],
                      eval_set=[(X_sc[val_idx], y_sc_log[val_idx])], verbose=0)
        elif mtype == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', iterations=3000, learning_rate=0.03,
                depth=5, l2_leaf_reg=5.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=100)
            model.fit(X_sc[tr_idx], y_sc_log[tr_idx],
                      eval_set=(X_sc[val_idx], y_sc_log[val_idx]), verbose=0)
        oof[val_idx] = np.expm1(model.predict(X_sc[val_idx]))
        tpred += np.expm1(model.predict(X_sc_test)) / N_SPLITS
    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    sc_oofs[mname] = oof
    sc_tests[mname] = tpred
    mae = mean_absolute_error(y_sc, oof)
    print(f" MAE: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

# Average of 3 scenario models
sc_oof_avg = np.mean(list(sc_oofs.values()), axis=0)
sc_test_avg = np.mean(list(sc_tests.values()), axis=0)
print(f"  SC avg OOF: {mean_absolute_error(y_sc, sc_oof_avg):.4f}")

# Broadcast to row level
train_sc['sc_pred'] = sc_oof_avg
test_sc['sc_pred'] = sc_test_avg

train_with_sc = train_fe.merge(train_sc[['layout_id','scenario_id','sc_pred']],
                                on=['layout_id','scenario_id'])
test_with_sc = test_fe.merge(test_sc[['layout_id','scenario_id','sc_pred']],
                              on=['layout_id','scenario_id'])

sc_pred_row = train_with_sc['sc_pred'].values
sc_pred_row_test = test_with_sc['sc_pred'].values
print(f"  SC broadcast row MAE: {mean_absolute_error(y, sc_pred_row):.4f}")

pickle.dump({
    'sc_oofs': sc_oofs, 'sc_tests': sc_tests,
    'sc_oof_avg': sc_oof_avg, 'sc_test_avg': sc_test_avg,
    'sc_pred_row': sc_pred_row, 'sc_pred_row_test': sc_pred_row_test,
}, open(f'{RESULT_DIR}/offset_phase1.pkl', 'wb'))

# CSV 저장
pd.DataFrame({'ID': test_fe['ID'], TARGET: sc_pred_row_test}).to_csv(
    './submission_scenario_only.csv', index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Offset Model (y - scenario_mean 예측)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Offset Model...", flush=True)

# y_offset = y - (true scenario mean)
y_scenario_mean_true = train_fe.groupby(['layout_id','scenario_id'])[TARGET].transform('mean').values
y_offset = y - y_scenario_mean_true
print(f"  Offset stats: mean={y_offset.mean():.4f}, std={y_offset.std():.4f}, abs_mean={np.abs(y_offset).mean():.4f}")

# Offset Features: v23 150개 + sc_pred (scenario level prediction)
train_offset = train_with_sc.copy()
test_offset = test_with_sc.copy()
offset_feats = v23_feats + ['sc_pred']
X_off = train_offset[offset_feats].fillna(0).values
X_off_test = test_offset[offset_feats].fillna(0).values

offset_oofs = {}
offset_tests = {}

for mtype, mname in [('lgb_huber', 'LGB_Huber'), ('lgb_mae', 'LGB_MAE'), ('xgb', 'XGB')]:
    t1 = time.time()
    print(f"  offset_{mname}...", end='', flush=True)

    oof = np.zeros(len(y))
    tpred = np.zeros(len(X_off_test))

    for tr_idx, val_idx in folds:
        if mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
        elif mtype == 'lgb_mae':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
            model.fit(X_off[tr_idx], y_offset[tr_idx],
                      eval_set=[(X_off[val_idx], y_offset[val_idx])], verbose=0)
            oof[val_idx] = model.predict(X_off[val_idx])
            tpred += model.predict(X_off_test) / N_SPLITS
            continue

        model.fit(X_off[tr_idx], y_offset[tr_idx],
                  eval_set=[(X_off[val_idx], y_offset[val_idx])],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict(X_off[val_idx])
        tpred += model.predict(X_off_test) / N_SPLITS

    offset_oofs[mname] = oof
    offset_tests[mname] = tpred
    mae = mean_absolute_error(y_offset, oof)
    print(f" offset_MAE: {mae:.4f} (baseline abs: {np.abs(y_offset).mean():.4f}) ({time.time()-t1:.0f}s)", flush=True)

pickle.dump({
    'offset_oofs': offset_oofs, 'offset_tests': offset_tests,
    'y_offset': y_offset,
}, open(f'{RESULT_DIR}/offset_phase2.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: 결합 (scenario + offset = full pred)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] 결합 예측...", flush=True)

# 각 offset model과 결합
combined_results = {}
for mname, offset_oof in offset_oofs.items():
    combined_oof = sc_pred_row + offset_oof
    combined_oof = np.clip(combined_oof, 0, None)
    combined_test = sc_pred_row_test + offset_tests[mname]
    combined_test = np.clip(combined_test, 0, None)
    mae = mean_absolute_error(y, combined_oof)
    print(f"  scenario + offset_{mname}: OOF={mae:.4f}")
    combined_results[mname] = {'oof': combined_oof, 'test': combined_test, 'mae': mae}
    pd.DataFrame({'ID': test_fe['ID'], TARGET: combined_test}).to_csv(
        f'./submission_offset_{mname}.csv', index=False)

# Best 선택
best_name = min(combined_results, key=lambda k: combined_results[k]['mae'])
best = combined_results[best_name]
print(f"\n  Best: {best_name} (OOF={best['mae']:.4f})")

pickle.dump(combined_results, open(f'{RESULT_DIR}/offset_phase3.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Mega-Stacking에 추가 (27 모델)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 4] Mega-Stacking 27-model...", flush=True)

selected_oofs = {}
selected_tests = {}
for seed in [42, 123, 2024]:
    s = pickle.load(open(f'{RESULT_DIR}/v23_seed{seed}.pkl', 'rb'))
    for name in ['LGB_Huber', 'XGB', 'CatBoost']:
        selected_oofs[f'v23s{seed}_{name}'] = s['oofs'][name]
        selected_tests[f'v23s{seed}_{name}'] = s['tests'][name]
v24 = pickle.load(open(f'{RESULT_DIR}/v24_final.pkl', 'rb'))
for name, oof in v24['oofs'].items():
    selected_oofs[f'v24_{name}'] = oof
    selected_tests[f'v24_{name}'] = v24['tests'][name]
v26 = pickle.load(open(f'{RESULT_DIR}/v26_final.pkl', 'rb'))
for name in ['Tuned_Huber', 'Tuned_sqrt', 'Tuned_pow', 'DART']:
    selected_oofs[f'v26_{name}'] = v26['oofs'][name]
    selected_tests[f'v26_{name}'] = v26['tests'][name]
mlp1 = pickle.load(open(f'{RESULT_DIR}/mlp_final.pkl', 'rb'))
mlp2 = pickle.load(open(f'{RESULT_DIR}/mlp2_final.pkl', 'rb'))
cnn = pickle.load(open(f'{RESULT_DIR}/cnn_final.pkl', 'rb'))
domain = pickle.load(open(f'{RESULT_DIR}/domain_phase2.pkl', 'rb'))
mlp_aug = pickle.load(open(f'{RESULT_DIR}/mlp_aug_final.pkl', 'rb'))
selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']
for name in domain['oofs']:
    selected_oofs[f'domain_{name}'] = domain['oofs'][name]
    selected_tests[f'domain_{name}'] = domain['tests'][name]
selected_oofs['mlp_aug'] = mlp_aug['mlp_aug_oof']
selected_tests['mlp_aug'] = mlp_aug['mlp_aug_test']

# 신규 3개 추가
for mname, data in combined_results.items():
    selected_oofs[f'offset_{mname}'] = data['oof']
    selected_tests[f'offset_{mname}'] = data['test']

print(f"  총 {len(selected_oofs)}개 모델")

stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])
y_log = np.log1p(y)

# 3-meta ensemble (LGB/XGB/CatBoost)
print("\n  3-meta ensemble...", flush=True)
meta_oofs = {}
meta_tests = {}

# LGB meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(objective='mae', n_estimators=500, learning_rate=0.05,
                           num_leaves=15, max_depth=4, min_child_samples=100,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs['lgb'] = oof
meta_tests['lgb'] = tpred
print(f"    LGB meta: {mean_absolute_error(y, oof):.4f}")

# XGB meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=500, learning_rate=0.05,
                          max_depth=4, min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=50)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])], verbose=0)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs['xgb'] = oof
meta_tests['xgb'] = tpred
print(f"    XGB meta: {mean_absolute_error(y, oof):.4f}")

# CatBoost meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = CatBoostRegressor(loss_function='MAE', iterations=500, learning_rate=0.05,
                           depth=4, l2_leaf_reg=5, random_seed=SEED, verbose=0, early_stopping_rounds=50)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=(stack_train[val_idx], y_log[val_idx]), verbose=0)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs['cb'] = oof
meta_tests['cb'] = tpred
print(f"    CatBoost meta: {mean_absolute_error(y, oof):.4f}")

# 3-meta average
meta_avg_oof = (meta_oofs['lgb'] + meta_oofs['xgb'] + meta_oofs['cb']) / 3
meta_avg_test = (meta_tests['lgb'] + meta_tests['xgb'] + meta_tests['cb']) / 3
print(f"    3-meta avg: {mean_absolute_error(y, meta_avg_oof):.4f}")

# CSV
for name, tpred in meta_tests.items():
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
        f'./submission_mega27_{name}.csv', index=False)
pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_avg_test}).to_csv(
    './submission_mega27_avg.csv', index=False)

# 블렌딩
prev_best = pd.read_csv('./submission_megastack_domain.csv')[TARGET].values
for ratio in [0.3, 0.5, 0.7]:
    blend = ratio * meta_avg_test + (1 - ratio) * prev_best
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_mega27_avg_prev_{int(ratio*100)}.csv', index=False)

print(f"\n{'='*60}")
print(f"Offset Decomposition 완료!")
print(f"  Scenario-level OOF: {mean_absolute_error(y_sc, sc_oof_avg):.4f}")
print(f"  Scenario broadcast row MAE: {mean_absolute_error(y, sc_pred_row):.4f}")
for name, data in combined_results.items():
    print(f"  scenario + offset_{name}: OOF={data['mae']:.4f}")
print(f"  Best combined: {best_name} (OOF={best['mae']:.4f})")
print(f"  27-model mega LGB: {mean_absolute_error(y, meta_oofs['lgb']):.4f}")
print(f"  27-model mega XGB: {mean_absolute_error(y, meta_oofs['xgb']):.4f}")
print(f"  27-model mega CB: {mean_absolute_error(y, meta_oofs['cb']):.4f}")
print(f"  27-model 3-meta avg: {mean_absolute_error(y, meta_avg_oof):.4f}")
print(f"  (이전 mega 24: 8.4541)")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
