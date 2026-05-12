"""
v18 MLP 스태킹: v14 베이스 + MLP 메타 학습
- v14 OOF/test를 메타피처로 MLP 스태킹
- LGB 앙상블 × 0.7 + MLP × 0.3 블렌딩
- log 모델 없으면 새로 학습 (checkpoint)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
CKPT_DIR = './results/v18_mlp_ckpt'
V14_CKPT = './results/v14_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

t0 = time.time()
def elapsed():
    return f"[{(time.time()-t0)/60:.1f}분]"

print(f"=== v18: MLP 스태킹 ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# 피처 엔지니어링 (v14 동일 — 시나리오 패턴 없음)
print(f"{elapsed()} 1. 피처 엔지니어링 (v14)...", flush=True)

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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. LGB 모델 (v14 checkpoint 또는 새로 학습)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 2. LGB 모델 (checkpoint/학습)...", flush=True)

def train_lgb(model_type, seed, target, transform, name):
    ckpt_oof = f'{CKPT_DIR}/{name}_oof.npy'
    ckpt_test = f'{CKPT_DIR}/{name}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        print(f"  {name}... [checkpoint] {elapsed()}", flush=True)
        return np.load(ckpt_oof), np.load(ckpt_test)

    # v14 checkpoint 확인
    v14_oof = f'{V14_CKPT}/{name}_s{seed}_oof.npy'
    v14_test = f'{V14_CKPT}/{name}_s{seed}_test.npy'
    if os.path.exists(v14_oof) and os.path.exists(v14_test):
        print(f"  {name}... [v14 checkpoint] {elapsed()}", flush=True)
        o = np.load(v14_oof); t = np.load(v14_test)
        np.save(ckpt_oof, o); np.save(ckpt_test, t)
        return o, t

    print(f"  {name}... [학습] {elapsed()}", flush=True)
    o = np.zeros(len(train)); t = np.zeros(len(test))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        ft = time.time()
        if model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
        else:
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
        model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                  eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        o[val_idx] = model.predict(X.iloc[val_idx])
        t += model.predict(X_test) / N_SPLITS
        print(f"    Fold {fold_idx+1} ({(time.time()-ft)/60:.1f}분) {elapsed()}", flush=True)
    if transform == 'log':
        o = np.expm1(o); t = np.expm1(t)
    np.save(ckpt_oof, o); np.save(ckpt_test, t)
    print(f"  → 저장 (oof mean={o.mean():.2f}) {elapsed()}", flush=True)
    return o, t

lgb_configs = [
    ('lgb_huber', y,     'raw', 'raw_LGB_Huber'),
    ('lgb',       y_log, 'log', 'log_LGB_MAE'),
    ('lgb_huber', y_log, 'log', 'log_LGB_Huber'),
]

lgb_oofs, lgb_tests = {}, {}
for mtype, target, transform, name in lgb_configs:
    o, t = train_lgb(mtype, SEED, target, transform, name)
    lgb_oofs[name] = o; lgb_tests[name] = t
    print(f"  {name} OOF MAE: {mean_absolute_error(y, o):.4f} {elapsed()}", flush=True)

# LGB 앙상블
names_lgb = list(lgb_oofs.keys())
oof_lgb_list = [lgb_oofs[n] for n in names_lgb]
test_lgb_list = [lgb_tests[n] for n in names_lgb]

def ens_lgb(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_lgb_list)))

res_lgb = minimize(ens_lgb, x0=[1/3]*3, method='Nelder-Mead', options={'maxiter': 50000})
bw_lgb = np.array(res_lgb.x); bw_lgb = np.maximum(bw_lgb, 0); bw_lgb = bw_lgb / bw_lgb.sum()
lgb_oof_ens = sum(wi*p for wi,p in zip(bw_lgb, oof_lgb_list))
lgb_test_ens = sum(wi*p for wi,p in zip(bw_lgb, test_lgb_list))
lgb_mae = mean_absolute_error(y, lgb_oof_ens)
print(f"\n  LGB 앙상블 OOF MAE: {lgb_mae:.4f}")
for n, w in zip(names_lgb, bw_lgb): print(f"    {n}: {w:.3f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. MLP 스태킹
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 3. MLP 스태킹...", flush=True)

ckpt_mlp_oof = f'{CKPT_DIR}/mlp_oof.npy'
ckpt_mlp_test = f'{CKPT_DIR}/mlp_test.npy'

if os.path.exists(ckpt_mlp_oof) and os.path.exists(ckpt_mlp_test):
    print(f"  MLP... [checkpoint] {elapsed()}", flush=True)
    mlp_oof = np.load(ckpt_mlp_oof)
    mlp_test_pred = np.load(ckpt_mlp_test)
else:
    # 메타피처: LGB 3개 OOF 예측값 + 원본 주요 피처
    X_filled = X.fillna(X.median())
    X_test_filled = X_test.fillna(X.median())

    mlp_oof = np.zeros(len(train))
    mlp_test_pred = np.zeros(len(test))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        ft = time.time()
        # 메타피처 구성: LGB OOF + 원본 피처
        meta_tr = np.column_stack([lgb_oofs[n][tr_idx] for n in names_lgb] + [X_filled.iloc[tr_idx].values])
        meta_val = np.column_stack([lgb_oofs[n][val_idx] for n in names_lgb] + [X_filled.iloc[val_idx].values])
        meta_test = np.column_stack([lgb_tests[n] for n in names_lgb] + [X_test_filled.values])

        scaler = StandardScaler()
        meta_tr = scaler.fit_transform(meta_tr)
        meta_val = scaler.transform(meta_val)
        meta_test_scaled = scaler.transform(meta_test)

        mlp = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64),
            activation='relu', solver='adam',
            alpha=0.01, batch_size=1024,
            learning_rate='adaptive', learning_rate_init=0.001,
            max_iter=200, early_stopping=True, n_iter_no_change=20,
            validation_fraction=0.1, random_state=SEED, verbose=False,
        )
        mlp.fit(meta_tr, y.iloc[tr_idx])
        mlp_oof[val_idx] = mlp.predict(meta_val)
        mlp_test_pred += mlp.predict(meta_test_scaled) / N_SPLITS
        print(f"    Fold {fold_idx+1} ({(time.time()-ft)/60:.1f}분) {elapsed()}", flush=True)

    np.save(ckpt_mlp_oof, mlp_oof)
    np.save(ckpt_mlp_test, mlp_test_pred)
    print(f"  → 저장 {elapsed()}", flush=True)

mlp_mae = mean_absolute_error(y, mlp_oof)
print(f"  MLP OOF MAE: {mlp_mae:.4f}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 최종 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 4. 최종 블렌딩...", flush=True)

# 다양한 비율 시도
for lgb_ratio in [0.6, 0.7, 0.8, 0.9, 1.0]:
    mlp_ratio = 1 - lgb_ratio
    blended_oof = lgb_oof_ens * lgb_ratio + mlp_oof * mlp_ratio
    blended_mae = mean_absolute_error(y, blended_oof)
    print(f"  LGB {lgb_ratio:.1f} + MLP {mlp_ratio:.1f}: OOF MAE = {blended_mae:.4f}", flush=True)

# 최적 비율 탐색
def blend_mae(alpha):
    return mean_absolute_error(y, lgb_oof_ens * alpha + mlp_oof * (1 - alpha))

from scipy.optimize import minimize_scalar
res_blend = minimize_scalar(blend_mae, bounds=(0, 1), method='bounded')
best_alpha = res_blend.x
best_blend_mae = res_blend.fun
print(f"\n  최적: LGB {best_alpha:.3f} + MLP {1-best_alpha:.3f}: OOF MAE = {best_blend_mae:.4f}")

# 제출 파일
final_pred = np.clip(lgb_oof_ens * best_alpha + mlp_oof * (1 - best_alpha), 0, None)  # OOF 기준
final_test = np.clip(lgb_test_ens * best_alpha + mlp_test_pred * (1 - best_alpha), 0, None)

pd.DataFrame({'ID': test['ID'], TARGET: final_test}).to_csv('./submission_v18_mlp.csv', index=False)

# LGB only도 저장 (비교용)
pd.DataFrame({'ID': test['ID'], TARGET: np.clip(lgb_test_ens, 0, None)}).to_csv('./submission_v18_lgb_only.csv', index=False)

print(f"\n{'='*60}")
print(f"v18 MLP 스태킹 완료! {elapsed()}")
print(f"  LGB 앙상블 OOF MAE: {lgb_mae:.4f}")
print(f"  MLP OOF MAE: {mlp_mae:.4f}")
print(f"  최적 블렌딩 OOF MAE: {best_blend_mae:.4f} (LGB {best_alpha:.3f} + MLP {1-best_alpha:.3f})")
print(f"  submission_v18_mlp.csv 저장")
print(f"  submission_v18_lgb_only.csv 저장")
print(f"{'='*60}", flush=True)
