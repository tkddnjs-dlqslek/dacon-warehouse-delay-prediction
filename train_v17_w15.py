"""
v17 fixed: 3모델(raw_Huber + log_MAE + log_Huber) + sample weight + 후처리
- CatBoost 제거
- sample weight: >=50분에 3.0
- early_stopping 100
- distribution calibration + seen/unseen 분리
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
CKPT_DIR = './results/v17_w15_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)

t0 = time.time()
def elapsed():
    return f"[{(time.time()-t0)/60:.1f}분]"

print(f"=== v17 w1.5: 3모델 + sample weight 1.5 + 후처리 ===", flush=True)
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

# Sample weight
EXTREME_THRESHOLD = 50
EXTREME_WEIGHT = 1.5
sample_weight = np.ones(len(train))
sample_weight[y >= EXTREME_THRESHOLD] = EXTREME_WEIGHT
print(f"   극단값(>={EXTREME_THRESHOLD}분): {(y>=EXTREME_THRESHOLD).sum():,}건, weight={EXTREME_WEIGHT}", flush=True)

# Seen/Unseen mask
train_layouts = set(train['layout_id'].unique())
test_layouts = set(test['layout_id'].unique())
seen_layouts = train_layouts & test_layouts
test_seen_mask = test['layout_id'].isin(seen_layouts).values
print(f"   Test seen: {test_seen_mask.sum():,}, unseen: {(~test_seen_mask).sum():,}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 학습 (3모델, checkpoint)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 2. 학습 (3모델, early_stop=100)...", flush=True)

def train_and_save(model_type, seed, target, transform, name):
    ckpt_oof = f'{CKPT_DIR}/{name}_s{seed}_oof.npy'
    ckpt_test = f'{CKPT_DIR}/{name}_s{seed}_test.npy'
    if os.path.exists(ckpt_oof) and os.path.exists(ckpt_test):
        print(f"  {name}... [checkpoint] {elapsed()}", flush=True)
        return np.load(ckpt_oof), np.load(ckpt_test)

    # raw는 weight 없음, log만 weight 1.5
    use_weight = (transform == 'log')
    print(f"  {name}... [학습, weight={'1.5' if use_weight else 'none'}] {elapsed()}", flush=True)
    oof = np.zeros(len(train)); tpred = np.zeros(len(test))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        ft = time.time()
        fold_w = sample_weight[tr_idx] if use_weight else None
        if model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      sample_weight=fold_w,
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        elif model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1)
            model.fit(X.iloc[tr_idx], target.iloc[tr_idx],
                      sample_weight=fold_w,
                      eval_set=[(X.iloc[val_idx], target.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
        print(f"    Fold {fold_idx+1} ({(time.time()-ft)/60:.1f}분) {elapsed()}", flush=True)

    if transform == 'log':
        oof = np.expm1(oof); tpred = np.expm1(tpred)
    np.save(ckpt_oof, oof); np.save(ckpt_test, tpred)
    print(f"  → 저장", flush=True)
    return oof, tpred

configs = [
    ('lgb_huber', y,     'raw', 'raw_LGB_Huber'),
    ('lgb',       y_log, 'log', 'log_LGB_MAE'),
    ('lgb_huber', y_log, 'log', 'log_LGB_Huber'),
]

all_oofs, all_tests = {}, {}
for mtype, target, transform, name in configs:
    o, t = train_and_save(mtype, SEED, target, transform, name)
    all_oofs[name] = o; all_tests[name] = t
    print(f"  {name} OOF MAE: {mean_absolute_error(y, o):.4f} {elapsed()}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 3. 앙상블...", flush=True)
names = list(all_oofs.keys())
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_list)))

res = minimize(ens_mae, x0=[1/3]*3, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
global_mae = res.fun

print(f"  앙상블 OOF MAE: {global_mae:.4f}")
for n, w in zip(names, bw): print(f"    {n}: {w:.3f}")

global_oof = sum(wi*p for wi,p in zip(bw, oof_list))
global_test = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Distribution Calibration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 4. Distribution Calibration...", flush=True)

def calibrate_distribution(pred, y_train, strength=0.3):
    n = len(pred)
    ranks = pd.Series(pred).rank(method='average') / n
    target_quantiles = np.quantile(y_train, ranks.values)
    calibrated = (1 - strength) * pred + strength * target_quantiles
    return np.clip(calibrated, 0, None)

best_strength = 0.0
best_cal_mae = mean_absolute_error(y, np.clip(global_oof, 0, None))
print(f"  보정 전 OOF MAE: {best_cal_mae:.4f}")

for s in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    cal_oof = calibrate_distribution(global_oof, y.values, strength=s)
    cal_mae = mean_absolute_error(y, cal_oof)
    mark = " ✓" if cal_mae < best_cal_mae else ""
    print(f"  strength={s:.2f}: OOF MAE={cal_mae:.4f}{mark}")
    if cal_mae < best_cal_mae:
        best_cal_mae = cal_mae
        best_strength = s

print(f"  최적 strength: {best_strength} (OOF MAE: {best_cal_mae:.4f})")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. Seen/Unseen 분리 후처리
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{elapsed()} 5. Seen/Unseen 분리...", flush=True)

# seen용: calibration 약하게 (이미 본 layout)
# unseen용: calibration 강하게 (못 본 layout이라 분포 보정 필요)
seen_pred = global_test.copy()
unseen_pred = global_test.copy()

if best_strength > 0:
    seen_pred = calibrate_distribution(global_test, y.values, strength=best_strength * 0.5)
    unseen_pred = calibrate_distribution(global_test, y.values, strength=best_strength)

split_pred = np.where(test_seen_mask, seen_pred, unseen_pred)
print(f"  Seen:   strength={best_strength*0.5:.2f}, mean={seen_pred[test_seen_mask].mean():.2f}")
print(f"  Unseen: strength={best_strength:.2f}, mean={unseen_pred[~test_seen_mask].mean():.2f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. Submission 저장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
pd.DataFrame({'ID': test['ID'], TARGET: global_test}).to_csv('./submission_v17_w15.csv', index=False)
pd.DataFrame({'ID': test['ID'], TARGET: split_pred}).to_csv('./submission_v17_w15_split.csv', index=False)
if best_strength > 0:
    cal_test = calibrate_distribution(global_test, y.values, strength=best_strength)
    pd.DataFrame({'ID': test['ID'], TARGET: cal_test}).to_csv('./submission_v17_w15_calibrated.csv', index=False)

# v17_ckpt 업데이트
import shutil
v17_old = './results/v17_ckpt'
if os.path.exists(v17_old):
    shutil.rmtree(v17_old)
shutil.copytree(CKPT_DIR, v17_old)

# 진단
extreme_mask = y >= EXTREME_THRESHOLD
print(f"\n{'='*60}")
print(f"v17 완료! {elapsed()}")
print(f"\n  [모델별 OOF MAE]")
for n in names:
    mae = mean_absolute_error(y, all_oofs[n])
    e_mae = mean_absolute_error(y[extreme_mask], all_oofs[n][extreme_mask])
    n_mae = mean_absolute_error(y[~extreme_mask], all_oofs[n][~extreme_mask])
    print(f"    {n}: 전체={mae:.4f}, 정상={n_mae:.4f}, 극단={e_mae:.4f}")

print(f"\n  [앙상블]")
print(f"    기본: {global_mae:.4f}")
print(f"    보정(s={best_strength}): {best_cal_mae:.4f}")
print(f"\n  [가중치] {dict(zip(names, bw.round(3)))}")
print(f"\n  [Submission]")
print(f"    submission_v17.csv (기본)")
print(f"    submission_v17_split.csv (seen/unseen 분리)")
if best_strength > 0:
    print(f"    submission_v17_calibrated.csv (분포 보정)")
print(f"{'='*60}", flush=True)
