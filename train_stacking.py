"""
Stacking: 6개 모델 OOF + 원본 피처 top30 → Meta-Learner
  - Ridge (안전) + LGB (보수적)
  - fold별 csv 저장
  - 5-way blend와 조합
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
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
print("=== Stacking Meta-Learner ===", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: OOF/Test predictions + 원본 피처 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 데이터 로드...", flush=True)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# v23 피처 생성
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

# 6개 모델 OOF/test 로드
s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
mlp1 = pickle.load(open(f'{RESULT_DIR}/mlp_final.pkl', 'rb'))
mlp2 = pickle.load(open(f'{RESULT_DIR}/mlp2_final.pkl', 'rb'))
cnn = pickle.load(open(f'{RESULT_DIR}/cnn_final.pkl', 'rb'))

model_oofs = {
    'lgb_huber': s42['oofs']['LGB_Huber'],
    'xgb': s42['oofs']['XGB'],
    'catboost': s42['oofs']['CatBoost'],
    'mlp1': mlp1['mlp_oof'],
    'mlp2': mlp2['mlp2_oof'],
    'cnn': cnn['cnn_oof'],
}
model_tests = {
    'lgb_huber': s42['tests']['LGB_Huber'],
    'xgb': s42['tests']['XGB'],
    'catboost': s42['tests']['CatBoost'],
    'mlp1': mlp1['mlp_test'],
    'mlp2': mlp2['mlp2_test'],
    'cnn': cnn['cnn_test'],
}

# 원본 피처 top 30 (v23 importance 기준)
v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
imp = v23_phase1['importance']
top30_feats = imp.head(30)['feature'].tolist()
top30_feats = [f for f in top30_feats if f in train_fe.columns]

print(f"  모델 OOF: {len(model_oofs)}개")
print(f"  원본 피처 top30: {len(top30_feats)}개")

# Stacking 피처 구성
def make_stack_features(oofs_dict, raw_df, top_feats, is_log=True):
    """OOF predictions (log) + 원본 피처 top30"""
    parts = []
    for name, oof in oofs_dict.items():
        if is_log:
            parts.append(np.log1p(np.clip(oof, 0, None)).reshape(-1, 1))
        else:
            parts.append(np.clip(oof, 0, None).reshape(-1, 1))
    # 원본 피처
    raw_vals = raw_df[top_feats].values.astype(np.float32)
    raw_vals = np.nan_to_num(raw_vals, 0)
    parts.append(raw_vals)
    return np.hstack(parts)

stack_train = make_stack_features(model_oofs, train_fe, top30_feats)
stack_test = make_stack_features(model_tests, test_fe, top30_feats)

stack_names = list(model_oofs.keys()) + top30_feats
print(f"  Stacking 피처: {stack_train.shape[1]}개 (6 OOF + {len(top30_feats)} raw)")
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

pickle.dump({'stack_names': stack_names}, open(f'{RESULT_DIR}/stacking_phase0.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Meta-Learner 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Meta-Learner 학습...", flush=True)

results = {}

# --- Ridge ---
print("\n  [Ridge] alpha 탐색...", flush=True)
best_ridge_mae = 999
best_alpha = 1.0
for alpha in [0.1, 1.0, 5.0, 10.0, 50.0, 100.0]:
    oof = np.zeros(len(y))
    for tr_idx, val_idx in folds:
        m = Ridge(alpha=alpha)
        m.fit(stack_train[tr_idx], y_log[tr_idx])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    oof = np.clip(oof, 0, None)
    mae = mean_absolute_error(y, oof)
    if mae < best_ridge_mae:
        best_ridge_mae = mae
        best_alpha = alpha
    print(f"    alpha={alpha:5.1f}: OOF={mae:.4f}", flush=True)

print(f"  Ridge best: alpha={best_alpha}, OOF={best_ridge_mae:.4f}")

# Ridge 최종 OOF + test
ridge_oof = np.zeros(len(y))
ridge_test = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = Ridge(alpha=best_alpha)
    m.fit(stack_train[tr_idx], y_log[tr_idx])
    ridge_oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    ridge_test += np.expm1(m.predict(stack_test)) / N_SPLITS
ridge_oof = np.clip(ridge_oof, 0, None)
ridge_test = np.clip(ridge_test, 0, None)

pd.DataFrame({'ID': test_fe['ID'], TARGET: ridge_test}).to_csv(
    './submission_stack_ridge.csv', index=False)
print(f"  submission_stack_ridge.csv 저장", flush=True)

results['ridge'] = {'oof': ridge_oof, 'test': ridge_test, 'mae': best_ridge_mae}

# --- LGB Meta (보수적) ---
print("\n  [LGB Meta] 보수적 설정...", flush=True)
lgb_configs = [
    ('lgb_7_3', {'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 200}),
    ('lgb_15_4', {'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 100}),
    ('lgb_7_3_500', {'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 500}),
]

for config_name, params in lgb_configs:
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=500, learning_rate=0.05,
            **params, random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    mae = mean_absolute_error(y, oof)
    print(f"    {config_name}: OOF={mae:.4f}", flush=True)

    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
        f'./submission_stack_{config_name}.csv', index=False)
    print(f"    submission_stack_{config_name}.csv 저장", flush=True)

    results[config_name] = {'oof': oof, 'test': tpred, 'mae': mae}

# --- OOF만으로 stacking (원본 피처 없이) ---
print("\n  [LGB Meta OOF-only] 비교...", flush=True)
stack_oof_only = make_stack_features(model_oofs, train_fe, [], is_log=True)
stack_test_only = make_stack_features(model_tests, test_fe, [], is_log=True)

oof_only = np.zeros(len(y))
tpred_only = np.zeros(len(stack_test_only))
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=500, learning_rate=0.05,
        num_leaves=7, max_depth=3, min_child_samples=200,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(stack_oof_only[tr_idx], y_log[tr_idx],
          eval_set=[(stack_oof_only[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_only[val_idx] = np.expm1(m.predict(stack_oof_only[val_idx]))
    tpred_only += np.expm1(m.predict(stack_test_only)) / N_SPLITS
oof_only = np.clip(oof_only, 0, None)
tpred_only = np.clip(tpred_only, 0, None)
mae_only = mean_absolute_error(y, oof_only)
print(f"    OOF-only: OOF={mae_only:.4f}", flush=True)

pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred_only}).to_csv(
    './submission_stack_oofonly.csv', index=False)
results['oofonly'] = {'oof': oof_only, 'test': tpred_only, 'mae': mae_only}

pickle.dump(results, open(f'{RESULT_DIR}/stacking_phase1.pkl', 'wb'))
print(f"\n  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] 블렌딩...", flush=True)

# 현재 best: 5-way blend
v22_test = pd.read_csv('submission_v22_pre.csv')[TARGET].values
v24_test = pd.read_csv('submission_v24.csv')[TARGET].values
v23_test = np.clip(sum(wi*p for wi, p in zip(s42['weights'], [s42['tests'][n] for n in s42['oofs'].keys()])), 0, None)

best5_test = 0.50*v23_test + 0.15*v22_test + 0.10*v24_test + 0.125*mlp1['mlp_test'] + 0.125*mlp2['mlp2_test']

# best stacking 선택
best_stack_name = min(results, key=lambda k: results[k]['mae'])
best_stack = results[best_stack_name]
print(f"  Best stacking: {best_stack_name} (OOF={best_stack['mae']:.4f})")

# stacking + 5-way 블렌딩
from scipy.stats import pearsonr
r, _ = pearsonr(best_stack['test'], best5_test)
print(f"  Stacking vs 5-way 상관: {r:.4f}")

for ratio in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
    blend = ratio * best_stack['test'] + (1 - ratio) * best5_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_stack_5way_{int(ratio*100)}.csv', index=False)

# 모든 stacking 결과 + 5-way 조합
for sname, sdata in results.items():
    for ratio in [0.5, 0.7]:
        blend = ratio * sdata['test'] + (1 - ratio) * best5_test
        pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
            f'./submission_{sname}_5way_{int(ratio*100)}.csv', index=False)

print(f"  블렌딩 csv 생성 완료", flush=True)

pickle.dump({
    'results': {k: {'mae': v['mae']} for k, v in results.items()},
    'best_stack_name': best_stack_name,
}, open(f'{RESULT_DIR}/stacking_final.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*60}")
print(f"Stacking 완료!")
print(f"  Ridge (alpha={best_alpha}): OOF={results['ridge']['mae']:.4f}")
for config_name, _, in lgb_configs:
    print(f"  {config_name}: OOF={results[config_name]['mae']:.4f}")
print(f"  OOF-only: OOF={results['oofonly']['mae']:.4f}")
print(f"  Best: {best_stack_name} (OOF={best_stack['mae']:.4f})")
print(f"  v23 대비: {8.5787 - best_stack['mae']:+.4f}")
print(f"  5-way LB 9.907 대비 Stacking 상관: {r:.4f}")
print(f"  submission_stack_*.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
