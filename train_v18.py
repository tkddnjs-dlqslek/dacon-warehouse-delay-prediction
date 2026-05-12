"""
v18: v17 기반 + MLP 앙상블 (raw_Huber + log 3개 + MLP, checkpoint)
- raw는 Huber만 유지, raw_MAE/raw_CB 제거
- v17 checkpoint 활용
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
CKPT_DIR = './results/v18_ckpt'
V17_CKPT = './results/v17_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

print("=== v18: v17 + MLP (raw_Huber + log 3개) ===", flush=True)
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
y_log = np.log1p(y)
X_test = test[feature_cols]
groups = train['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

# v17 checkpoint 로드 (raw_Huber + log 3개만)
print("\n2. Tree 모델 로드/학습...", flush=True)
tree_models = ['raw_LGB_Huber', 'log_LGB_MAE', 'log_LGB_Huber', 'log_CatBoost']
all_oofs, all_tests = {}, {}

def train_tree(model_type, seed, target, transform='raw'):
    oof = np.zeros(len(train)); tpred = np.zeros(len(test))
    for tr_idx, val_idx in folds:
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx], eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0, random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx], eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif model_type == 'cb':
            model = CatBoostRegressor(loss_function='MAE', eval_metric='MAE', iterations=3000, learning_rate=0.03,
                depth=6, l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=seed, verbose=0, early_stopping_rounds=200)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx], eval_set=(X.iloc[val_idx], target.iloc[val_idx]), verbose=0)
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    if transform == 'log': oof = np.expm1(oof); tpred = np.expm1(tpred)
    return oof, tpred

tree_config = [
    ('lgb_huber', 'raw_LGB_Huber', y, 'raw'),
    ('lgb', 'log_LGB_MAE', y_log, 'log'),
    ('lgb_huber', 'log_LGB_Huber', y_log, 'log'),
    ('cb', 'log_CatBoost', y_log, 'log'),
]

for mtype, mname, target, transform in tree_config:
    # v17 checkpoint 먼저 확인
    loaded = False
    for ckpt_dir in [V17_CKPT, CKPT_DIR]:
        ckpt_oof = f'{ckpt_dir}/{mname}_s{SEED}_oof.npy'
        ckpt_test = f'{ckpt_dir}/{mname}_s{SEED}_test.npy'
        if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
            print(f"  {mname}... [checkpoint from {ckpt_dir}]", flush=True)
            all_oofs[mname] = np.load(ckpt_oof)
            all_tests[mname] = np.load(ckpt_test)
            loaded = True
            break
    if not loaded:
        print(f"  {mname}... [학습]", flush=True)
        o, t = train_tree(mtype, SEED, target, transform)
        ckpt_oof = f'{CKPT_DIR}/{mname}_s{SEED}_oof.npy'
        ckpt_test = f'{CKPT_DIR}/{mname}_s{SEED}_test.npy'
        np.save(ckpt_oof, o); np.save(ckpt_test, t)
        all_oofs[mname] = o; all_tests[mname] = t
        print(f"    → 저장", flush=True)
    print(f"  {mname} OOF MAE: {mean_absolute_error(y, all_oofs[mname]):.4f}", flush=True)

# MLP 학습
print("\n3. MLP 학습...", flush=True)
X_filled = X.fillna(X.median())
X_test_filled = X_test.fillna(X.median())

mlp_configs = [
    {'name': 'MLP_raw', 'target': y, 'transform': 'raw'},
    {'name': 'MLP_log', 'target': y_log, 'transform': 'log'},
]

for cfg in mlp_configs:
    ckpt_oof = f'{CKPT_DIR}/{cfg["name"]}_s{SEED}_oof.npy'
    ckpt_test = f'{CKPT_DIR}/{cfg["name"]}_s{SEED}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        print(f"  {cfg['name']}... [checkpoint]", flush=True)
        all_oofs[cfg['name']] = np.load(ckpt_oof)
        all_tests[cfg['name']] = np.load(ckpt_test)
    else:
        print(f"  {cfg['name']}... [학습]", flush=True)
        oof = np.zeros(len(train)); tpred = np.zeros(len(test))
        for fold_idx, (tr_idx, val_idx) in enumerate(folds):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_filled.iloc[tr_idx])
            X_val = scaler.transform(X_filled.iloc[val_idx])
            X_te = scaler.transform(X_test_filled)
            mlp = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
                alpha=0.01, batch_size=1024, learning_rate='adaptive', learning_rate_init=0.001,
                max_iter=200, early_stopping=True, n_iter_no_change=20,
                validation_fraction=0.1, random_state=SEED, verbose=False)
            mlp.fit(X_tr, cfg['target'].iloc[tr_idx])
            oof[val_idx] = mlp.predict(X_val)
            tpred += mlp.predict(X_te) / N_SPLITS
            print(f"    Fold {fold_idx+1} done", flush=True)
        if cfg['transform'] == 'log': oof = np.expm1(oof); tpred = np.expm1(tpred)
        np.save(ckpt_oof, oof); np.save(ckpt_test, tpred)
        all_oofs[cfg['name']] = oof; all_tests[cfg['name']] = tpred
        print(f"    → 저장", flush=True)
    print(f"  {cfg['name']} OOF MAE: {mean_absolute_error(y, all_oofs[cfg['name']]):.4f}", flush=True)

# 앙상블
print("\n4. 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[1/len(names)]*len(names), method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
best_mae = mean_absolute_error(y, sum(wi*p for wi,p in zip(bw, oof_list)))

print(f"  앙상블 OOF MAE: {best_mae:.4f}")
for n, w in zip(names, bw):
    print(f"    {n}: {w:.3f}")

final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: final_pred}).to_csv('./submission_v18.csv', index=False)

print(f"\n{'='*60}")
print(f"v18 완료!")
print(f"  앙상블 OOF MAE: {best_mae:.4f}")
print(f"  submission_v18.csv 저장")
print(f"{'='*60}", flush=True)
