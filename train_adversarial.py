"""
Adversarial Sample Weighting + Reweighted 학습
  - Adversarial classifier로 train의 test-likeness 점수 계산
  - sample_weight로 LGB/XGB/CB 재학습
  - mega-stacking 23-model로 확장
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
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
print("=== Adversarial Sample Weighting ===", flush=True)

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

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
feature_cols = [f for f in v23_selected if f in train_fe.columns]
print(f"  v23 피처: {len(feature_cols)}개")

X = train_fe[feature_cols].values
X_test = test_fe[feature_cols].values
X = np.nan_to_num(X, 0)
X_test = np.nan_to_num(X_test, 0)

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups=groups))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Adversarial Classifier
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Adversarial Classifier 학습...", flush=True)

# train + test 합치기
X_combined = np.vstack([X, X_test])
y_adv = np.hstack([np.zeros(len(X)), np.ones(len(X_test))])

# 5-fold OOF로 train sample마다 test-likeness score
adv_oof_train = np.zeros(len(X))
n_train = len(X)

# 합친 데이터를 KFold (그룹 무관, layer leakage 없음)
adv_kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
for fold_idx, (tr_idx, val_idx) in enumerate(adv_kf.split(X_combined)):
    m = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        max_depth=6, min_child_samples=100,
        subsample=0.7, colsample_bytree=0.7,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(X_combined[tr_idx], y_adv[tr_idx])
    val_pred = m.predict_proba(X_combined[val_idx])[:, 1]
    # train sample만 골라서 저장
    train_mask = val_idx < n_train
    train_val_idx = val_idx[train_mask]
    adv_oof_train[train_val_idx] = val_pred[train_mask]

auc = roc_auc_score(y_adv, np.concatenate([adv_oof_train, np.ones(len(X_test))*0.7]))  # rough check
print(f"  Train test-likeness: mean={adv_oof_train.mean():.4f}, P50={np.median(adv_oof_train):.4f}, P90={np.percentile(adv_oof_train, 90):.4f}")

pickle.dump({'adv_oof': adv_oof_train}, open(f'{RESULT_DIR}/adversarial_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Reweighted 학습 (alpha 후보 3개)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Reweighted 학습...", flush=True)

# Sample weights — alpha 3개 후보
def make_weights(alpha):
    w = 1.0 + alpha * adv_oof_train
    return w / w.mean()  # 평균 1로 정규화

alpha_configs = {
    'alpha1': 1.0,
    'alpha3': 3.0,
    'alpha5': 5.0,
}

reweighted_results = {}

for alpha_name, alpha in alpha_configs.items():
    print(f"\n  --- {alpha_name} (weight = 1 + {alpha} * test_likeness) ---", flush=True)
    sw = make_weights(alpha)
    print(f"    Weight stats: min={sw.min():.3f}, max={sw.max():.3f}, P95={np.percentile(sw, 95):.3f}")

    rw_oofs = {}
    rw_tests = {}

    for mtype, mname in [('lgb_huber', 'LGB_Huber'), ('xgb', 'XGB'), ('cb', 'CatBoost')]:
        t1 = time.time()
        print(f"    {mname}...", end='', flush=True)

        oof = np.zeros(len(X))
        tpred = np.zeros(len(X_test))

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            tr_w = sw[tr_idx]

            if mtype == 'lgb_huber':
                model = lgb.LGBMRegressor(
                    objective='huber', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=SEED, verbose=-1, n_jobs=-1)
                model.fit(X[tr_idx], y_log[tr_idx], sample_weight=tr_w,
                          eval_set=[(X[val_idx], y_log[val_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            elif mtype == 'xgb':
                model = xgb.XGBRegressor(
                    objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                    max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                    random_state=SEED, verbosity=0, n_jobs=-1, early_stopping_rounds=200)
                model.fit(X[tr_idx], y_log[tr_idx], sample_weight=tr_w,
                          eval_set=[(X[val_idx], y_log[val_idx])], verbose=0)
            elif mtype == 'cb':
                model = CatBoostRegressor(
                    loss_function='MAE', eval_metric='MAE',
                    iterations=5000, learning_rate=0.03, depth=7,
                    l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                    random_seed=SEED, verbose=0, early_stopping_rounds=200)
                model.fit(X[tr_idx], y_log[tr_idx], sample_weight=tr_w,
                          eval_set=(X[val_idx], y_log[val_idx]), verbose=0)

            oof[val_idx] = np.expm1(model.predict(X[val_idx]))
            tpred += np.expm1(model.predict(X_test)) / N_SPLITS

        oof = np.clip(oof, 0, None)
        tpred = np.clip(tpred, 0, None)
        rw_oofs[mname] = oof
        rw_tests[mname] = tpred
        mae = mean_absolute_error(y, oof)
        print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

        # 모델별 csv 체크포인트
        pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
            f'./submission_adv_{alpha_name}_{mname}.csv', index=False)

    reweighted_results[alpha_name] = {'oofs': rw_oofs, 'tests': rw_tests, 'alpha': alpha}
    pickle.dump(reweighted_results, open(f'{RESULT_DIR}/adversarial_phase2.pkl', 'wb'))

print(f"\n  Phase 2 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Mega-Stacking 23-model
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] Mega-Stacking 23-model 재학습...", flush=True)

# 기존 20개 OOF/test 로드
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
selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']

# 가장 좋은 alpha의 reweighted 모델 추가 (3개)
# 일단 모든 alpha 추가해서 큰 stacking
for alpha_name, data in reweighted_results.items():
    for mname, oof in data['oofs'].items():
        selected_oofs[f'rw{alpha_name}_{mname}'] = oof
        selected_tests[f'rw{alpha_name}_{mname}'] = data['tests'][mname]

print(f"  총 {len(selected_oofs)}개 모델")

stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])

# LGB meta
print("\n  LGB Meta-Learner 학습...", flush=True)
mega_oof = np.zeros(len(y))
mega_test = np.zeros(len(stack_test))
for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=500, learning_rate=0.05,
        num_leaves=15, max_depth=4, min_child_samples=100,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    mega_oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    mega_test += np.expm1(m.predict(stack_test)) / N_SPLITS
mega_oof = np.clip(mega_oof, 0, None)
mega_test = np.clip(mega_test, 0, None)
mega_mae = mean_absolute_error(y, mega_oof)
print(f"  Mega 23-model OOF: {mega_mae:.4f}")

pd.DataFrame({'ID': test_fe['ID'], TARGET: mega_test}).to_csv(
    './submission_megastack_adv.csv', index=False)

# 다양한 LGB config
for params, name in [
    ({'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 200}, 'lgb73'),
    ({'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 50}, 'lgb315'),
]:
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = lgb.LGBMRegressor(objective='mae', n_estimators=500, learning_rate=0.05,
                               **params, random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    print(f"  Mega 23-model ({name}): OOF={mean_absolute_error(y, oof):.4f}")
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
        f'./submission_megastack_adv_{name}.csv', index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 4] 블렌딩...", flush=True)

prev_best = pd.read_csv('./submission_megastack_lgb_15_4.csv')[TARGET].values

for ratio in [0.3, 0.5, 0.7]:
    blend = ratio * mega_test + (1 - ratio) * prev_best
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_adv_prev_{int(ratio*100)}.csv', index=False)

print(f"\n{'='*60}")
print(f"Adversarial Weighting 완료!")
for alpha_name, data in reweighted_results.items():
    for mname, oof in data['oofs'].items():
        print(f"  {alpha_name}_{mname}: OOF={mean_absolute_error(y, oof):.4f}")
print(f"  Mega 23-model: OOF={mega_mae:.4f}")
print(f"  v23 단독 대비: {8.5787 - mega_mae:+.4f}")
print(f"  submission_megastack_adv*.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
