"""
v7: 피처 선별 + 정규화 강화 + target encoding 제거
- v5 대비: FI 기반 피처 선별, 정규화↑
- v6 대비: target encoding 제거, 노이즈 피처 제거
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEEDS = [42, 123, 2024]

print("1. 데이터 로드...", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 피처 엔지니어링 (선별적)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("2. 피처 엔지니어링...", flush=True)


def engineer_features(df, layout_df):
    df = df.merge(layout_df, on='layout_id', how='left')

    layout_map = {'grid': 0, 'hybrid': 1, 'narrow': 2, 'hub_spoke': 3}
    df['layout_type_enc'] = df['layout_type'].map(layout_map)

    # 타임슬롯
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    df['timeslot_sq'] = df['timeslot'] ** 2
    df['timeslot_norm'] = df['timeslot'] / 24.0

    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    # 핵심 컬럼만 lag/rolling (너무 많으면 과적합)
    key_cols = [
        'order_inflow_15m', 'congestion_score', 'robot_utilization',
        'battery_mean', 'fault_count_15m', 'blocked_path_15m',
        'pack_utilization', 'charge_queue_length',
    ]

    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        # Lag (1, 2만 — lag3은 노이즈)
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        # Diff
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        # Rolling (3만)
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        # Cumulative
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())

    # 핵심 상호작용만 (검증된 것만)
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    robot_total_active = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (robot_total_active + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (robot_total_active + 1)
    df['battery_risk'] = df['low_battery_ratio'] * df['charge_queue_length']
    df['congestion_x_utilization'] = df['congestion_score'] * df['robot_utilization']
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
    df['order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
    df['urgent_order_volume'] = df['order_inflow_15m'] * df['urgent_order_ratio']
    df['dock_pressure'] = df['loading_dock_util'] * df['outbound_truck_wait_min']
    df['staff_per_order'] = df['staff_on_floor'] / (df['order_inflow_15m'] + 1)
    df['total_utilization'] = (
        df['pack_utilization'] + df['staging_area_util'] + df['loading_dock_util']
    ) / 3
    df['fault_x_congestion'] = df['fault_count_15m'] * df['congestion_score']
    df['battery_charge_pressure'] = df['low_battery_ratio'] * df['avg_charge_wait']

    # Layout 기반 (적은 수만)
    if 'floor_area_sqm' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
        df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)

    return df


train = engineer_features(train, layout)
test = engineer_features(test, layout)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
feature_cols = [c for c in train.columns if c not in exclude]
print(f"   피처 수: {len(feature_cols)}", flush=True)

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]
groups = train['scenario_id']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.5 FI 기반 피처 선별 (1 seed로 빠르게)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("2.5. FI 기반 피처 선별...", flush=True)

gkf_fi = GroupKFold(n_splits=N_SPLITS)
fi_sum = np.zeros(len(feature_cols))
for tr_idx, val_idx in gkf_fi.split(X, y, groups):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=1000, learning_rate=0.05,
        num_leaves=63, random_state=42, verbose=-1, n_jobs=-1,
    )
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    fi_sum += m.feature_importances_

fi_df = pd.DataFrame({'feature': feature_cols, 'importance': fi_sum}).sort_values('importance', ascending=False)

# 중요도 0인 피처 제거
important_features = fi_df[fi_df['importance'] > 0]['feature'].tolist()
print(f"   중요 피처: {len(important_features)} / {len(feature_cols)}", flush=True)

X = train[important_features]
X_test = test[important_features]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 1 seed로 각 모델 테스트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n3. 1 seed 테스트 (정규화 강화)...", flush=True)


def train_lgb(seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = lgb.LGBMRegressor(
            objective='mae', n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


def train_xgb(seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
            max_depth=7, min_child_weight=10,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            tree_method='hist', random_state=seed, verbosity=0, n_jobs=-1,
            early_stopping_rounds=200,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


def train_cb(seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups):
        model = CatBoostRegressor(
            loss_function='MAE', eval_metric='MAE',
            iterations=5000, learning_rate=0.03, depth=7,
            l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
            random_seed=seed, verbose=0, early_stopping_rounds=200,
        )
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


# 1 seed 빠른 테스트
print("  LGB seed=42...", flush=True)
lgb_oof_1, lgb_test_1 = train_lgb(42)
lgb_mae_1 = mean_absolute_error(y, lgb_oof_1)
print(f"  LGB OOF MAE: {lgb_mae_1:.4f}", flush=True)

print("  XGB seed=42...", flush=True)
xgb_oof_1, xgb_test_1 = train_xgb(42)
xgb_mae_1 = mean_absolute_error(y, xgb_oof_1)
print(f"  XGB OOF MAE: {xgb_mae_1:.4f}", flush=True)

print("  CB seed=42...", flush=True)
cb_oof_1, cb_test_1 = train_cb(42)
cb_mae_1 = mean_absolute_error(y, cb_oof_1)
print(f"  CB OOF MAE: {cb_mae_1:.4f}", flush=True)

# 1-seed 앙상블
oof_1s = [lgb_oof_1, xgb_oof_1, cb_oof_1]
test_1s = [lgb_test_1, xgb_test_1, cb_test_1]

def ens_mae(w):
    w = np.array(w); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_1s)))

res1 = minimize(ens_mae, x0=[1/3]*3, method='Nelder-Mead')
bw1 = np.array(res1.x); bw1 = bw1 / bw1.sum()
print(f"\n  1-seed 앙상블 OOF MAE: {res1.fun:.4f}", flush=True)
print(f"  가중치: LGB={bw1[0]:.3f}, XGB={bw1[1]:.3f}, CB={bw1[2]:.3f}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 3-seed averaging (1-seed가 괜찮으면)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. 3-seed averaging...", flush=True)

lgb_oofs, lgb_tests = [lgb_oof_1], [lgb_test_1]
xgb_oofs, xgb_tests = [xgb_oof_1], [xgb_test_1]
cb_oofs, cb_tests = [cb_oof_1], [cb_test_1]

for seed in SEEDS[1:]:  # 123, 2024
    print(f"  LGB seed={seed}...", flush=True)
    o, t = train_lgb(seed); lgb_oofs.append(o); lgb_tests.append(t)
    print(f"  XGB seed={seed}...", flush=True)
    o, t = train_xgb(seed); xgb_oofs.append(o); xgb_tests.append(t)
    print(f"  CB seed={seed}...", flush=True)
    o, t = train_cb(seed); cb_oofs.append(o); cb_tests.append(t)

lgb_oof = np.mean(lgb_oofs, axis=0)
xgb_oof = np.mean(xgb_oofs, axis=0)
cb_oof = np.mean(cb_oofs, axis=0)
lgb_test_pred = np.mean(lgb_tests, axis=0)
xgb_test_pred = np.mean(xgb_tests, axis=0)
cb_test_pred = np.mean(cb_tests, axis=0)

print(f"\n  LGB 3-seed OOF: {mean_absolute_error(y, lgb_oof):.4f}", flush=True)
print(f"  XGB 3-seed OOF: {mean_absolute_error(y, xgb_oof):.4f}", flush=True)
print(f"  CB 3-seed OOF: {mean_absolute_error(y, cb_oof):.4f}", flush=True)

# 3-seed 앙상블
oof_3s = [lgb_oof, xgb_oof, cb_oof]
test_3s = [lgb_test_pred, xgb_test_pred, cb_test_pred]

def ens_mae3(w):
    w = np.array(w); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_3s)))

res3 = minimize(ens_mae3, x0=[1/3]*3, method='Nelder-Mead')
bw3 = np.array(res3.x); bw3 = bw3 / bw3.sum()

print(f"\n  3-seed 앙상블 OOF MAE: {res3.fun:.4f}", flush=True)
print(f"  가중치: LGB={bw3[0]:.3f}, XGB={bw3[1]:.3f}, CB={bw3[2]:.3f}", flush=True)

# 둘 다 저장
pred_1s = sum(wi*p for wi,p in zip(bw1, test_1s))
pred_3s = sum(wi*p for wi,p in zip(bw3, test_3s))

pd.DataFrame({'ID': test['ID'], TARGET: np.clip(pred_1s, 0, None)}).to_csv('./submission_v7_1seed.csv', index=False)
pd.DataFrame({'ID': test['ID'], TARGET: np.clip(pred_3s, 0, None)}).to_csv('./submission_v7_3seed.csv', index=False)

print(f"\n{'='*60}")
print(f"v7 완료!")
print(f"  1-seed OOF MAE: {res1.fun:.4f}")
print(f"  3-seed OOF MAE: {res3.fun:.4f}")
print(f"  submission_v7_1seed.csv 저장")
print(f"  submission_v7_3seed.csv 저장")
print(f"{'='*60}", flush=True)
