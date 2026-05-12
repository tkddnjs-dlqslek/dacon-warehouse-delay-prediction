"""
v25: v23 피처 150개 보존 + 선별적 SC 확장
  - v23 피처 150개 고정 (base)
  - avg_trip_distance, max_zone_density, unique_sku_15m SC 추가
  - robot_utilization slope 추가
  - 1-fold 검증 후 개선되는 것만 최종 포함
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
print("=== v25: v23 보존 + 선별적 SC 확장 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# v23 피처 리스트 로드
v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
print(f"  v23 피처 로드: {len(v23_selected)}개", flush=True)
print(f"  데이터 로드 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링...", flush=True)

def engineer_features_v25(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'battery_mean', 'fault_count_15m', 'blocked_path_15m',
                'pack_utilization', 'charge_queue_length']

    # === v23 동일: Lag/Rolling (key_cols) ===
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

    # === v23 동일: Forward-looking (key_cols) ===
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_lead2'] = g.shift(-2)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === v23 동일: Extra cols lag/lead ===
    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]

    # === v23 동일: SC Features (8 key_cols) ===
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

    # === NEW: 선별적 SC 확장 (3개 컬럼만) ===
    new_sc_cols = ['avg_trip_distance', 'max_zone_density', 'unique_sku_15m']
    for col in new_sc_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_sc_mean'] = g.transform('mean')
        df[f'{col}_sc_std'] = g.transform('std')
        df[f'{col}_sc_min'] = g.transform('min')
        df[f'{col}_sc_range'] = df[f'{col}_sc_max'] if f'{col}_sc_max' in df.columns else g.transform('max') - g.transform('min')
        # max도 필요
        df[f'{col}_sc_max'] = g.transform('max')
        df[f'{col}_sc_range'] = df[f'{col}_sc_max'] - df[f'{col}_sc_min']

    # === NEW: robot_utilization slope (numpy로 빠르게) ===
    def fast_slope(x):
        n = len(x)
        if n < 3: return 0.0
        t = np.arange(n, dtype=float)
        vals = x.values.astype(float)
        valid = ~np.isnan(vals)
        if valid.sum() < 3: return 0.0
        t_v, v_v = t[valid], vals[valid]
        return np.polyfit(t_v, v_v, 1)[0]

    df['robot_utilization_sc_slope'] = group['robot_utilization'].transform(fast_slope)

    # === v23 동일: 상호작용 피처 ===
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

    # === v23 동일: Layout 비율 피처 ===
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

train_fe = engineer_features_v25(train, layout)
test_fe = engineer_features_v25(test, layout)

y = train_fe[TARGET].values
y_log = np.log1p(y)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_feature_cols = [c for c in train_fe.columns if c not in exclude]
print(f"  전체 피처 수: {len(all_feature_cols)}", flush=True)

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

# 신규 피처 목록
new_features = [c for c in all_feature_cols if c not in v23_selected and c != 'extreme_prob']
print(f"  v23 피처 중 존재: {len([f for f in v23_selected if f in all_feature_cols])}/{len(v23_selected)}")
print(f"  신규 피처: {len(new_features)}개: {new_features}")
print(f"  Phase 0a 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0b: Extreme Probability
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
        random_state=SEED, verbose=-1, n_jobs=-1)
    clf.fit(X_cls.iloc[tr_idx], y_cls[tr_idx],
            eval_set=[(X_cls.iloc[val_idx], y_cls[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    extreme_oof[val_idx] = clf.predict_proba(X_cls.iloc[val_idx])[:, 1]
    extreme_test += clf.predict_proba(test_fe[all_feature_cols])[:, 1] / N_SPLITS

auc = roc_auc_score(y_cls, extreme_oof)
print(f"  Extreme AUC: {auc:.4f}", flush=True)

train_fe['extreme_prob'] = extreme_oof
test_fe['extreme_prob'] = extreme_test

pickle.dump({'auc': auc}, open(f'{RESULT_DIR}/v25_phase0.pkl', 'wb'))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Feature Selection (v23 보존 + 신규 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Feature Selection (v23 보존 + 신규 검증)...", flush=True)

# v23 150개 중 실제 존재하는 것 + extreme_prob
base_features = [f for f in v23_selected if f in train_fe.columns]
if 'extreme_prob' not in base_features:
    base_features.append('extreme_prob')
print(f"  Base (v23): {len(base_features)}개", flush=True)

# 1-fold baseline MAE (v23 피처만)
tr_idx, val_idx = folds[0]
m_base = lgb.LGBMRegressor(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1)
m_base.fit(train_fe[base_features].iloc[tr_idx], y_log[tr_idx],
           eval_set=[(train_fe[base_features].iloc[val_idx], y_log[val_idx])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
base_pred = np.clip(np.expm1(m_base.predict(train_fe[base_features].iloc[val_idx])), 0, None)
base_mae = mean_absolute_error(y[val_idx], base_pred)
print(f"  v23 base fold0 MAE: {base_mae:.4f}", flush=True)

# 신규 피처 하나씩 추가해서 검증
print("  신규 피처 개별 검증:", flush=True)
good_new_features = []
for nf in new_features:
    if nf not in train_fe.columns: continue
    test_feats = base_features + [nf]
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(train_fe[test_feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[test_feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(train_fe[test_feats].iloc[val_idx])), 0, None)
    mae = mean_absolute_error(y[val_idx], pred)
    diff = base_mae - mae
    status = "+" if diff > 0 else "-"
    print(f"    {status} {nf:45s} MAE={mae:.4f} (diff={diff:+.4f})", flush=True)
    if diff > 0.001:  # 0.001 이상 개선되는 것만
        good_new_features.append((nf, diff))

good_new_features.sort(key=lambda x: -x[1])
print(f"\n  개선 피처: {len(good_new_features)}개", flush=True)
for nf, d in good_new_features:
    print(f"    {nf:45s} +{d:.4f}")

# 전체 추가 검증
if good_new_features:
    all_good = [nf for nf, _ in good_new_features]
    final_features = base_features + all_good
    m_final = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m_final.fit(train_fe[final_features].iloc[tr_idx], y_log[tr_idx],
                eval_set=[(train_fe[final_features].iloc[val_idx], y_log[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m_final.predict(train_fe[final_features].iloc[val_idx])), 0, None)
    final_mae = mean_absolute_error(y[val_idx], pred)
    print(f"\n  전체 추가 fold0 MAE: {final_mae:.4f} (base: {base_mae:.4f}, diff: {base_mae-final_mae:+.4f})")

    if final_mae < base_mae:
        selected_features = final_features
        print(f"  -> 신규 피처 포함! 최종: {len(selected_features)}개")
    else:
        selected_features = base_features
        print(f"  -> 전체 추가 시 악화, v23 base 유지: {len(selected_features)}개")
else:
    selected_features = base_features
    print(f"  -> 개선 피처 없음, v23 base 유지: {len(selected_features)}개")

pickle.dump({
    'selected_features': selected_features,
    'base_features': base_features,
    'good_new_features': good_new_features,
    'base_mae': base_mae,
}, open(f'{RESULT_DIR}/v25_phase1.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Training (seed=42)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n[Phase 2] Training (seed=42, {len(selected_features)} features)...", flush=True)

X = train_fe[selected_features]
X_test = test_fe[selected_features]

models_config = [
    ('lgb', 'LGB_MAE'),
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
        if mtype == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], y_log[tr_idx],
                      eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif mtype == 'lgb_huber':
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

    # 중간 저장
    pickle.dump({'oofs': oofs, 'tests': tests_dict}, open(f'{RESULT_DIR}/v25_phase2.pkl', 'wb'))

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
    './submission_v25.csv', index=False)

pickle.dump({
    'oofs': oofs, 'tests': tests_dict,
    'weights': w, 'ens_mae': ens_mae,
    'selected_features': selected_features,
}, open(f'{RESULT_DIR}/v25_final.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"v25 완료!")
print(f"  피처: base {len(base_features)} + new {len(good_new_features)} = {len(selected_features)}")
print(f"  Extreme AUC: {auc:.4f}")
for n in names:
    print(f"  {n} OOF: {mean_absolute_error(y, oofs[n]):.4f}")
print(f"  앙상블 OOF MAE: {ens_mae:.4f}")
print(f"  v23 대비: {8.5787 - ens_mae:+.4f}")
print(f"  submission_v25.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
