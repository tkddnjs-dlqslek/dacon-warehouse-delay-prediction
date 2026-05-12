"""
v17 clean: 순수 학습 + 앙상블 + 저장만. calibration/후처리 없음.
- 3모델: raw_Huber + log_MAE + log_Huber
- weight 없음
- checkpoint 역변환 후 저장
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
CKPT_DIR = './results/v17_clean_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

t0 = time.time()
def elapsed():
    return f"[{(time.time()-t0)/60:.1f}분]"

print(f"=== v17 clean ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

print(f"{elapsed()} 1. 피처 엔지니어링...", flush=True)

def engineer_features(df, layout_df):
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
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    rta = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (rta + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (rta + 1)
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
    group2 = df.groupby(['layout_id', 'scenario_id'])
    for col in ['congestion_score', 'order_inflow_15m', 'robot_utilization', 'pack_utilization']:
        if col not in df.columns: continue
        g = group2[col]
        df[f'{col}_slope3'] = (df[col] - g.shift(2)) / 3
        df[f'{col}_slope5'] = (df[col] - g.shift(4)) / 5
    for col in ['congestion_score', 'order_inflow_15m', 'blocked_path_15m']:
        if col not in df.columns: continue
        g = group2[col]
        cummax = g.transform(lambda x: x.expanding().max())
        cummin = g.transform(lambda x: x.expanding().min())
        df[f'{col}_vs_cummax'] = df[col] / (cummax + 1)
        df[f'{col}_vs_cummin'] = df[col] / (cummin + 1)
        df[f'{col}_cumrange'] = cummax - cummin
    for col in ['order_inflow_15m', 'fault_count_15m', 'blocked_path_15m', 'near_collision_15m']:
        if col not in df.columns: continue
        df[f'{col}_cumsum'] = group2[col].transform(lambda x: x.expanding().sum())
    for col in ['congestion_score', 'order_inflow_15m']:
        diff1_col = f'{col}_diff1'
        if diff1_col in df.columns:
            df[f'{col}_accel'] = group2[diff1_col].shift(0) - group2[diff1_col].shift(1)
    layout_static = ['layout_type', 'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                     'pack_station_count', 'charger_count', 'layout_compactness', 'zone_dispersion',
                     'robot_total', 'building_age_years', 'floor_area_sqm', 'ceiling_height_m',
                     'fire_sprinkler_count', 'emergency_exit_count']
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors='ignore')
    corr_remove = ['battery_mean_rmean3', 'charge_queue_length_rmean3',
                   'battery_mean_rmean5', 'charge_queue_length_rmean5',
                   'pack_utilization_rmean5', 'battery_mean_lag1',
                   'charge_queue_length_lag1', 'congestion_score_rmean3',
                   'order_inflow_15m_cummean', 'robot_utilization_rmean5',
                   'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
                   'battery_risk', 'congestion_score_rmean5',
                   'pack_utilization_rmean3', 'order_inflow_15m_rmean3',
                   'charge_queue_length_lag2', 'blocked_path_15m_rmean5']
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')
    return df

train = engineer_features(train, layout)
test = engineer_features(test, layout)
exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
feature_cols = [c for c in train.columns if c not in exclude]
print(f"   피처 수: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
y_log = np.log1p(y)
X_test = test[feature_cols]
groups = train['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

# 학습
print(f"\n{elapsed()} 2. 학습...", flush=True)

configs = [
    ('lgb_huber', y,     'raw', 'raw_LGB_Huber'),
    ('lgb',       y_log, 'log', 'log_LGB_MAE'),
    ('lgb_huber', y_log, 'log', 'log_LGB_Huber'),
]

all_oofs, all_tests = {}, {}

for mtype, target, transform, name in configs:
    ckpt_oof = f'{CKPT_DIR}/{name}_oof.npy'
    ckpt_test = f'{CKPT_DIR}/{name}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        print(f"  {name}... [checkpoint] {elapsed()}", flush=True)
        o = np.load(ckpt_oof); t = np.load(ckpt_test)
    else:
        print(f"  {name}... [학습] {elapsed()}", flush=True)
        o = np.zeros(len(train)); t = np.zeros(len(test))
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            ft = time.time()
            if mtype == 'lgb_huber':
                model = lgb.LGBMRegressor(
                    objective='huber', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=SEED, verbose=-1, n_jobs=-1)
            else:
                model = lgb.LGBMRegressor(
                    objective='mae', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=SEED, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
            o[val_idx] = model.predict(X.iloc[val_idx])
            t += model.predict(X_test) / N_SPLITS
            print(f"    Fold {fold_idx+1} ({(time.time()-ft)/60:.1f}분) {elapsed()}", flush=True)

        # 역변환
        if transform == 'log':
            o = np.expm1(o); t = np.expm1(t)

        # 저장
        np.save(ckpt_oof, o); np.save(ckpt_test, t)
        print(f"  → 저장 (oof mean={o.mean():.2f}, test mean={t.mean():.2f})", flush=True)

    all_oofs[name] = o; all_tests[name] = t
    print(f"  {name} OOF MAE: {mean_absolute_error(y, o):.4f} {elapsed()}", flush=True)

# 앙상블
print(f"\n{elapsed()} 3. 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[1/3]*3, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
best_mae = res.fun

print(f"  앙상블 OOF MAE: {best_mae:.4f}")
for n, w in zip(names, bw): print(f"    {n}: {w:.3f}")

# 제출 파일 저장
final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: final_pred}).to_csv('./submission_v17_clean.csv', index=False)

print(f"\n{'='*60}")
print(f"v17 clean 완료! {elapsed()}")
print(f"  앙상블 OOF MAE: {best_mae:.4f}")
print(f"  가중치: {dict(zip(names, bw.round(3)))}")
print(f"  pred mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
print(f"  submission_v17_clean.csv 저장")
print(f"{'='*60}", flush=True)
