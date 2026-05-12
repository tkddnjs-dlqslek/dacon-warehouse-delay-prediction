"""
v19: 2-stage 모델 (raw_Huber + log 3개 구조, checkpoint)
- Stage 1: 50분 이상 여부 분류
- Stage 2a/2b: 별도 회귀
- v17/v18 결과와 전체 앙상블
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
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
CKPT_DIR = './results/v19_ckpt'
V17_CKPT = './results/v17_ckpt'
V18_CKPT = './results/v18_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)
THRESHOLD = 50

print("=== v19: 2-stage + 전체 앙상블 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

print("1. 피처 엔지니어링 (v17 동일)...", flush=True)

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
X_test = test[feature_cols]
groups = train['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

# 2-Stage
print(f"\n2. 2-Stage 모델 (threshold={THRESHOLD}분)...", flush=True)
y_cls = (y >= THRESHOLD).astype(int)
print(f"  극단값 비율: {y_cls.mean()*100:.1f}%", flush=True)

ckpt_2s_oof = f'{CKPT_DIR}/2stage_oof.npy'
ckpt_2s_test = f'{CKPT_DIR}/2stage_test.npy'

if os.path.exists(ckpt_2s_oof) and os.path.exists(ckpt_2s_test):
    print("  [checkpoint]", flush=True)
    oof_2s = np.load(ckpt_2s_oof)
    test_2s = np.load(ckpt_2s_test)
else:
    oof_2s = np.zeros(len(train)); test_2s = np.zeros(len(test))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        print(f"  Fold {fold_idx+1}...", flush=True)
        clf = lgb.LGBMClassifier(n_estimators=2000, learning_rate=0.03, num_leaves=63,
            subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, random_state=SEED, verbose=-1, n_jobs=-1)
        clf.fit(X.iloc[tr_idx], y_cls.iloc[tr_idx], eval_set=[(X.iloc[val_idx], y_cls.iloc[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        prob_val = clf.predict_proba(X.iloc[val_idx])[:, 1]
        prob_test = clf.predict_proba(X_test)[:, 1]

        normal_mask = y.iloc[tr_idx] < THRESHOLD
        reg_n = lgb.LGBMRegressor(objective='huber', n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)
        reg_n.fit(X.iloc[tr_idx][normal_mask], np.log1p(y.iloc[tr_idx][normal_mask]),
                  eval_set=[(X.iloc[val_idx], np.log1p(y.iloc[val_idx]))],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        pred_n_val = np.expm1(reg_n.predict(X.iloc[val_idx]))
        pred_n_test = np.expm1(reg_n.predict(X_test))

        extreme_mask = y.iloc[tr_idx] >= THRESHOLD
        reg_e = lgb.LGBMRegressor(objective='huber', n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0, random_state=SEED, verbose=-1, n_jobs=-1)
        reg_e.fit(X.iloc[tr_idx][extreme_mask], np.log1p(y.iloc[tr_idx][extreme_mask]),
                  eval_set=[(X.iloc[val_idx], np.log1p(y.iloc[val_idx]))],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        pred_e_val = np.expm1(reg_e.predict(X.iloc[val_idx]))
        pred_e_test = np.expm1(reg_e.predict(X_test))

        oof_2s[val_idx] = (1 - prob_val) * pred_n_val + prob_val * pred_e_val
        test_2s += ((1 - prob_test) * pred_n_test + prob_test * pred_e_test) / N_SPLITS

    np.save(ckpt_2s_oof, oof_2s); np.save(ckpt_2s_test, test_2s)
    print("  → 저장", flush=True)

print(f"  2-Stage OOF MAE: {mean_absolute_error(y, oof_2s):.4f}", flush=True)

# v17/v18 결과 로드
print("\n3. 이전 결과 로드 + 전체 앙상블...", flush=True)
all_oofs = {'2stage': oof_2s}
all_tests = {'2stage': test_2s}

for name in ['raw_LGB_Huber', 'log_LGB_MAE', 'log_LGB_Huber', 'log_CatBoost']:
    for ckpt_dir in [V17_CKPT, V18_CKPT]:
        ckpt_oof = f'{ckpt_dir}/{name}_s{SEED}_oof.npy'
        ckpt_test = f'{ckpt_dir}/{name}_s{SEED}_test.npy'
        if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
            all_oofs[name] = np.load(ckpt_oof)
            all_tests[name] = np.load(ckpt_test)
            print(f"  {name}: OOF {mean_absolute_error(y, all_oofs[name]):.4f}", flush=True)
            break

for name in ['MLP_raw', 'MLP_log']:
    ckpt_oof = f'{V18_CKPT}/{name}_s{SEED}_oof.npy'
    ckpt_test = f'{V18_CKPT}/{name}_s{SEED}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        all_oofs[name] = np.load(ckpt_oof)
        all_tests[name] = np.load(ckpt_test)
        print(f"  {name}: OOF {mean_absolute_error(y, all_oofs[name]):.4f}", flush=True)

print(f"\n  총 {len(all_oofs)}개 모델 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[1/len(names)]*len(names), method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
best_mae = mean_absolute_error(y, sum(wi*p for wi,p in zip(bw, oof_list)))

print(f"  전체 앙상블 OOF MAE: {best_mae:.4f}")
for n, w in zip(names, bw):
    if w > 0.001: print(f"    {n}: {w:.3f}")

final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: final_pred}).to_csv('./submission_v19.csv', index=False)
pd.DataFrame({'ID': test['ID'], TARGET: np.clip(test_2s, 0, None)}).to_csv('./submission_v19_2stage.csv', index=False)

print(f"\n{'='*60}")
print(f"v19 완료!")
print(f"  2-Stage OOF MAE: {mean_absolute_error(y, oof_2s):.4f}")
print(f"  전체 앙상블 OOF MAE: {best_mae:.4f}")
print(f"  submission_v19.csv 저장")
print(f"  submission_v19_2stage.csv 저장")
print(f"{'='*60}", flush=True)
