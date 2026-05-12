"""
v23: v22 기반 + 4가지 새 전략
  A. Scenario-Level Global Features (시나리오 전체 통계)
  B. Extreme Probability Feature (극단값 분류기 확률)
  C. Multi-Seed Averaging (3 seeds)
  D. Feature Selection (importance 상위 선택)
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
SEEDS = [42, 123, 2024]
RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)

t0 = time.time()
print("=== v23: Scenario Features + Extreme Prob + Multi-Seed + Feature Selection ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')
print(f"  데이터 로드 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering
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

    # === v22 기존: lag/rolling ===
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

    # === v22: forward-looking ===
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_lead2'] = g.shift(-2)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === v22: extra cols lag/lead ===
    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === NEW: Strategy A — Scenario-Level Global Features ===
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

    # === v22 상호작용 피처 ===
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

    # === v22 layout 비율 피처 ===
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

train_fe = engineer_features_v23(train, layout)
test_fe = engineer_features_v23(test, layout)

y = train_fe[TARGET].values
y_log = np.log1p(y)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_feature_cols = [c for c in train_fe.columns if c not in exclude]
print(f"  전체 피처 수: {len(all_feature_cols)}", flush=True)

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Strategy B: Extreme Probability Feature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0b] 극단값 분류기 OOF...", flush=True)
X_cls = train_fe[all_feature_cols]
y_cls = (y > 50).astype(int)

extreme_oof = np.zeros(len(train_fe))
extreme_test = np.zeros(len(test_fe))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1, n_jobs=4)
    clf.fit(X_cls.iloc[tr_idx], y_cls[tr_idx],
            eval_set=[(X_cls.iloc[val_idx], y_cls[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    extreme_oof[val_idx] = clf.predict_proba(X_cls.iloc[val_idx])[:, 1]
    extreme_test += clf.predict_proba(test_fe[all_feature_cols])[:, 1] / N_SPLITS

auc = roc_auc_score(y_cls, extreme_oof)
print(f"  Extreme AUC (>50min): {auc:.4f}", flush=True)

# 피처로 추가
train_fe['extreme_prob'] = extreme_oof
test_fe['extreme_prob'] = extreme_test
all_feature_cols.append('extreme_prob')
print(f"  피처 수 (+extreme_prob): {len(all_feature_cols)}", flush=True)
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Feature Selection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Feature Selection...", flush=True)

# 1-fold로 importance 계산
tr_idx, val_idx = folds[0]
sel_model = lgb.LGBMRegressor(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=4)
sel_model.fit(train_fe[all_feature_cols].iloc[tr_idx], y_log[tr_idx],
              eval_set=[(train_fe[all_feature_cols].iloc[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

imp = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': sel_model.feature_importances_
}).sort_values('importance', ascending=False)

# 상위 150개 선택
N_SELECT = 150
selected_features = imp.head(N_SELECT)['feature'].tolist()

# scenario features 몇 개 선택됐는지
sc_selected = [f for f in selected_features if '_sc_' in f]
lead_selected = [f for f in selected_features if '_lead' in f]
print(f"  전체 {len(all_feature_cols)} → 선택 {N_SELECT}개", flush=True)
print(f"  scenario features 선택: {len(sc_selected)}개", flush=True)
print(f"  lead features 선택: {len(lead_selected)}개", flush=True)
print(f"  extreme_prob 선택: {'extreme_prob' in selected_features}", flush=True)

# importance top 20 출력
print("  Top 20 features:")
for i, (_, row) in enumerate(imp.head(20).iterrows()):
    tag = ""
    if '_sc_' in row['feature']: tag = " [SC]"
    elif '_lead' in row['feature']: tag = " [LEAD]"
    elif row['feature'] == 'extreme_prob': tag = " [EXT]"
    print(f"    {i+1:2d}. {row['feature']:40s} imp={row['importance']:6.0f}{tag}")

# Phase 1 결과 저장
pickle.dump({
    'all_feature_cols': all_feature_cols,
    'selected_features': selected_features,
    'importance': imp,
}, open(f'{RESULT_DIR}/v23_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Multi-Seed Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Multi-Seed Training...", flush=True)

X = train_fe[selected_features]
X_test = test_fe[selected_features]

models_config = [
    ('lgb', 'LGB_MAE'),
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
]

all_seed_results = {}

for seed_idx, seed in enumerate(SEEDS):
    print(f"\n  --- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---", flush=True)
    seed_oofs = {}
    seed_tests = {}

    for mtype, mname in models_config:
        t1 = time.time()
        print(f"    {mname}...", end='', flush=True)

        oof = np.zeros(len(X))
        tpred = np.zeros(len(X_test))

        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            if mtype == 'lgb':
                model = lgb.LGBMRegressor(
                    objective='mae', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=seed, verbose=-1, n_jobs=4)
                model.fit(X.iloc[tr_idx], y_log[tr_idx],
                          eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            elif mtype == 'lgb_huber':
                model = lgb.LGBMRegressor(
                    objective='huber', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=seed, verbose=-1, n_jobs=4)
                model.fit(X.iloc[tr_idx], y_log[tr_idx],
                          eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            elif mtype == 'xgb':
                model = xgb.XGBRegressor(
                    objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                    max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                    random_state=seed, verbosity=0, n_jobs=4, early_stopping_rounds=200)
                model.fit(X.iloc[tr_idx], y_log[tr_idx],
                          eval_set=[(X.iloc[val_idx], y_log[val_idx])], verbose=0)
            elif mtype == 'cb':
                model = CatBoostRegressor(
                    loss_function='MAE', eval_metric='MAE',
                    iterations=5000, learning_rate=0.03, depth=7,
                    l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                    random_seed=seed, verbose=0, early_stopping_rounds=200, thread_count=4, task_type='CPU')
                model.fit(X.iloc[tr_idx], y_log[tr_idx],
                          eval_set=(X.iloc[val_idx], y_log[val_idx]), verbose=0)

            oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))
            tpred += np.expm1(model.predict(X_test)) / N_SPLITS

        oof = np.clip(oof, 0, None)
        tpred = np.clip(tpred, 0, None)
        seed_oofs[mname] = oof
        seed_tests[mname] = tpred
        mae = mean_absolute_error(y, oof)
        print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # seed별 앙상블
    names = list(seed_oofs.keys())
    oof_list = [seed_oofs[n] for n in names]
    test_list = [seed_tests[n] for n in names]

    def optimize_weights(oofs, y_true):
        n = len(oofs)
        def obj(w):
            w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
            return mean_absolute_error(y_true, sum(wi*p for wi, p in zip(w, oofs)))
        res = minimize(obj, x0=[1/n]*n, method='Nelder-Mead', options={'maxiter': 20000})
        w = np.array(res.x); w = np.maximum(w, 0); w = w / w.sum()
        return w, res.fun

    w, ens_mae = optimize_weights(oof_list, y)
    ens_pred = np.clip(sum(wi*p for wi, p in zip(w, test_list)), 0, None)

    print(f"  Seed {seed} 앙상블 OOF MAE: {ens_mae:.4f}")
    for n, wi in zip(names, w):
        if wi > 0.01: print(f"    {n}: {wi:.3f}")

    all_seed_results[seed] = {
        'oofs': seed_oofs, 'tests': seed_tests,
        'weights': w, 'ens_mae': ens_mae, 'ens_pred': ens_pred,
    }

    # seed별 pickle 저장
    pickle.dump(all_seed_results[seed], open(f'{RESULT_DIR}/v23_seed{seed}.pkl', 'wb'))
    print(f"  Seed {seed} 저장 완료", flush=True)

    # seed=42 결과로 submission 미리 저장 (안전장치)
    if seed == 42:
        pd.DataFrame({'ID': test_fe['ID'], TARGET: ens_pred}).to_csv(
            './submission_v23_seed42.csv', index=False)
        print(f"  submission_v23_seed42.csv 저장 (안전장치)", flush=True)

print(f"\n  Phase 2 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Final Ensemble (3-seed 평균)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] 3-Seed 최종 앙상블...", flush=True)

# 3-seed 단순 평균
seed_preds = [all_seed_results[s]['ens_pred'] for s in SEEDS]
final_pred = np.clip(np.mean(seed_preds, axis=0), 0, None)

# seed별 OOF MAE 요약
for seed in SEEDS:
    print(f"  Seed {seed}: OOF MAE = {all_seed_results[seed]['ens_mae']:.4f}")

# 최종 저장
pd.DataFrame({'ID': test_fe['ID'], TARGET: final_pred}).to_csv(
    './submission_v23.csv', index=False)

# seed=42 단독도 비교용 저장
seed42_pred = all_seed_results[42]['ens_pred']

print(f"\n{'='*60}")
print(f"v23 완료!")
print(f"  전체 피처: {len(all_feature_cols)} → 선택: {N_SELECT}")
print(f"  Scenario features 선택: {len(sc_selected)}/{56}")
print(f"  Extreme AUC: {auc:.4f}")
for seed in SEEDS:
    print(f"  Seed {seed} OOF MAE: {all_seed_results[seed]['ens_mae']:.4f}")
print(f"  v22_pre 대비 (seed42): {8.602 - all_seed_results[42]['ens_mae']:+.4f}")
print(f"  submission_v23.csv (3-seed 평균) 저장")
print(f"  submission_v23_seed42.csv (단독) 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
