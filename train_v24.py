"""
v24: v23 기반 + SC 확장 + Scenario Slope + Quantile SC
  A. SC Features 18개 컬럼으로 확장 (8 → 18)
  B. Scenario Trend (기울기)
  C. Feature Selection 최적화 (N 비교)
  D. Quantile SC (Q25/Q75/IQR)
  - seed=42 단독 (3-seed 효과 없었음)
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
from scipy.stats import linregress
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

t0 = time.time()
print("=== v24: SC 확장 + Slope + Quantile SC ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')
print(f"  데이터 로드 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링...", flush=True)

def engineer_features_v24(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'battery_mean', 'fault_count_15m', 'blocked_path_15m',
                'pack_utilization', 'charge_queue_length']

    # SC 확장 대상 (key_cols + 추가 10개)
    sc_cols = key_cols + [
        'avg_trip_distance', 'max_zone_density', 'robot_idle',
        'robot_charging', 'low_battery_ratio', 'avg_items_per_order',
        'manual_override_ratio', 'unique_sku_15m', 'avg_recovery_time',
        'near_collision_15m',
    ]

    # === Lag/Rolling (key_cols만) ===
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

    # === Forward-looking (key_cols만) ===
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_lead2'] = g.shift(-2)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === Extra cols lag/lead ===
    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === Strategy A: SC Features 확장 (18개 컬럼) ===
    for col in sc_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_sc_mean'] = g.transform('mean')
        df[f'{col}_sc_std'] = g.transform('std')
        df[f'{col}_sc_max'] = g.transform('max')
        df[f'{col}_sc_min'] = g.transform('min')
        df[f'{col}_sc_range'] = df[f'{col}_sc_max'] - df[f'{col}_sc_min']
        df[f'{col}_sc_rank'] = g.rank(pct=True)
        df[f'{col}_sc_dev'] = df[col] - df[f'{col}_sc_mean']

    # === Strategy B: Scenario Slope (18개 컬럼) ===
    def calc_slope(x):
        if len(x) < 3:
            return 0.0
        try:
            return linregress(range(len(x)), x.values).slope
        except:
            return 0.0

    for col in sc_cols:
        if col not in df.columns: continue
        df[f'{col}_sc_slope'] = group[col].transform(calc_slope)

    # === Strategy D: Quantile SC (18개 컬럼) ===
    for col in sc_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_sc_q25'] = g.transform(lambda x: x.quantile(0.25))
        df[f'{col}_sc_q75'] = g.transform(lambda x: x.quantile(0.75))
        df[f'{col}_sc_iqr'] = df[f'{col}_sc_q75'] - df[f'{col}_sc_q25']

    # === 상호작용 피처 (v23 동일) ===
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

    # === Layout 비율 피처 ===
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

train_fe = engineer_features_v24(train, layout)
test_fe = engineer_features_v24(test, layout)

y = train_fe[TARGET].values
y_log = np.log1p(y)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_feature_cols = [c for c in train_fe.columns if c not in exclude]
print(f"  전체 피처 수: {len(all_feature_cols)}", flush=True)

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))
print(f"  Phase 0a 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Extreme Probability Feature
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0b] 극단값 분류기...", flush=True)
X_cls = train_fe[all_feature_cols]
y_cls = (y > 50).astype(int)

extreme_oof = np.zeros(len(train_fe))
extreme_test = np.zeros(len(test_fe))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        max_depth=6, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        random_state=SEED, verbose=-1, n_jobs=4)
    clf.fit(X_cls.iloc[tr_idx], y_cls[tr_idx],
            eval_set=[(X_cls.iloc[val_idx], y_cls[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    extreme_oof[val_idx] = clf.predict_proba(X_cls.iloc[val_idx])[:, 1]
    extreme_test += clf.predict_proba(test_fe[all_feature_cols])[:, 1] / N_SPLITS

auc = roc_auc_score(y_cls, extreme_oof)
print(f"  Extreme AUC (>50min): {auc:.4f}", flush=True)

train_fe['extreme_prob'] = extreme_oof
test_fe['extreme_prob'] = extreme_test
all_feature_cols.append('extreme_prob')
print(f"  피처 수 (+extreme_prob): {len(all_feature_cols)}", flush=True)

pickle.dump({'all_feature_cols': all_feature_cols}, open(f'{RESULT_DIR}/v24_phase0.pkl', 'wb'))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Feature Selection (Strategy C)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Feature Selection...", flush=True)

# 1-fold importance 계산
tr_idx, val_idx = folds[0]
sel_model = lgb.LGBMRegressor(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=4)
sel_model.fit(train_fe[all_feature_cols].iloc[tr_idx], y_log[tr_idx],
              eval_set=[(train_fe[all_feature_cols].iloc[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

imp = pd.DataFrame({
    'feature': all_feature_cols,
    'importance': sel_model.feature_importances_
}).sort_values('importance', ascending=False)

# N 비교 (1-fold에서 빠르게)
print("  Feature count 비교 (fold 0):", flush=True)
best_n = 150
best_fold_mae = 999

for N in [100, 120, 130, 150, 180]:
    sel_feats = imp.head(N)['feature'].tolist()
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(train_fe[sel_feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[sel_feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.expm1(m.predict(train_fe[sel_feats].iloc[val_idx]))
    fold_mae = mean_absolute_error(y[val_idx], np.clip(pred, 0, None))
    print(f"    N={N:3d}: fold0 MAE={fold_mae:.4f}", flush=True)
    if fold_mae < best_fold_mae:
        best_fold_mae = fold_mae
        best_n = N

print(f"  -> 최적 N={best_n} (fold0 MAE={best_fold_mae:.4f})", flush=True)

selected_features = imp.head(best_n)['feature'].tolist()

# SC 통계
sc_selected = [f for f in selected_features if '_sc_' in f]
slope_selected = [f for f in selected_features if '_sc_slope' in f]
q_selected = [f for f in selected_features if '_sc_q' in f or '_sc_iqr' in f]
print(f"  선택: {best_n}개 (SC={len(sc_selected)}, slope={len(slope_selected)}, quantile={len(q_selected)})", flush=True)

# Top 20
print("  Top 20 features:")
for i, (_, row) in enumerate(imp.head(20).iterrows()):
    tag = ""
    if '_sc_slope' in row['feature']: tag = " [SLOPE]"
    elif '_sc_q' in row['feature'] or '_sc_iqr' in row['feature']: tag = " [Q]"
    elif '_sc_' in row['feature']: tag = " [SC]"
    elif '_lead' in row['feature']: tag = " [LEAD]"
    elif row['feature'] == 'extreme_prob': tag = " [EXT]"
    print(f"    {i+1:2d}. {row['feature']:45s} imp={row['importance']:6.0f}{tag}")

pickle.dump({
    'all_feature_cols': all_feature_cols,
    'selected_features': selected_features,
    'importance': imp,
    'best_n': best_n,
}, open(f'{RESULT_DIR}/v24_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Training (seed=42 단독)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Training (seed=42)...", flush=True)

X = train_fe[selected_features]
X_test = test_fe[selected_features]

models_config = [
    ('lgb', 'LGB_MAE'),
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
]

oofs = {}
tests = {}

for mtype, mname in models_config:
    t1 = time.time()
    print(f"  {mname}...", end='', flush=True)

    oof = np.zeros(len(X))
    tpred = np.zeros(len(X_test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        if mtype == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=4)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=4)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                random_state=SEED, verbosity=0, n_jobs=4, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])], verbose=0)
        elif mtype == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=200, thread_count=4, task_type='CPU')
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=(X.iloc[val_idx], y_log[val_idx]), verbose=0)

        oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))
        tpred += np.expm1(model.predict(X_test)) / N_SPLITS

    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    oofs[mname] = oof
    tests[mname] = tpred
    mae = mean_absolute_error(y, oof)
    print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    # 모델별 중간 저장
    pickle.dump({'oofs': oofs, 'tests': tests}, open(f'{RESULT_DIR}/v24_phase2.pkl', 'wb'))

print(f"  Phase 2 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: Ensemble
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] 앙상블...", flush=True)

names = list(oofs.keys())
oof_list = [oofs[n] for n in names]
test_list = [tests[n] for n in names]

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

# 저장
pd.DataFrame({'ID': test_fe['ID'], TARGET: ens_pred}).to_csv(
    './submission_v24.csv', index=False)

pickle.dump({
    'oofs': oofs, 'tests': tests,
    'weights': w, 'ens_mae': ens_mae,
}, open(f'{RESULT_DIR}/v24_final.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"v24 완료!")
print(f"  전체 피처: {len(all_feature_cols)} -> 선택: {best_n}")
print(f"  SC features: {len(sc_selected)} (slope={len(slope_selected)}, quantile={len(q_selected)})")
print(f"  Extreme AUC: {auc:.4f}")
for n in names:
    print(f"  {n} OOF: {mean_absolute_error(y, oofs[n]):.4f}")
print(f"  앙상블 OOF MAE: {ens_mae:.4f}")
print(f"  v23 대비: {8.5787 - ens_mae:+.4f}")
print(f"  submission_v24.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
