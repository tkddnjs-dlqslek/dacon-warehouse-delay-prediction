"""
v8: CV 전략 개선 + 피처 정제 + Huber loss + 다양한 정규화
- GroupKFold(layout_id) 시도 — test layout이 train 부분집합
- 피처 수 최적 범위 탐색
- Huber loss (MAE와 MSE 중간)
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
# 2. 피처 엔지니어링 (v7 기반 + 미세 조정)
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
    df['timeslot_sin'] = np.sin(2 * np.pi * df['timeslot'] / 25)
    df['timeslot_cos'] = np.cos(2 * np.pi * df['timeslot'] / 25)

    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    # lag/rolling — v7과 동일한 핵심 8개 + rolling window 5 추가
    key_cols = [
        'order_inflow_15m', 'congestion_score', 'robot_utilization',
        'battery_mean', 'fault_count_15m', 'blocked_path_15m',
        'pack_utilization', 'charge_queue_length',
    ]

    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())

    # 상호작용 (v7과 동일 + 소수 추가)
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
    df['congestion_per_robot'] = df['congestion_score'] / (df['robot_active'] + 1)
    df['order_per_staff'] = df['order_inflow_15m'] / (df['staff_on_floor'] + 1)

    # Layout 기반
    if 'floor_area_sqm' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
        df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)

    return df


train = engineer_features(train, layout)
test = engineer_features(test, layout)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
all_feature_cols = [c for c in train.columns if c not in exclude]
print(f"   전체 피처 수: {len(all_feature_cols)}", flush=True)

y = train[TARGET]
groups_scenario = train['scenario_id']
groups_layout = train['layout_id']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.5 FI 기반 피처 선별
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("2.5. FI 기반 피처 선별...", flush=True)

X_all = train[all_feature_cols]
gkf_fi = GroupKFold(n_splits=N_SPLITS)
fi_sum = np.zeros(len(all_feature_cols))
for tr_idx, val_idx in gkf_fi.split(X_all, y, groups_scenario):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=1000, learning_rate=0.05,
        num_leaves=63, random_state=42, verbose=-1, n_jobs=-1,
    )
    m.fit(X_all.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X_all.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    fi_sum += m.feature_importances_

fi_df = pd.DataFrame({'feature': all_feature_cols, 'importance': fi_sum}).sort_values('importance', ascending=False)
# 상위 150개만 선택 (과적합 방지)
top_n = min(150, len(fi_df[fi_df['importance'] > 0]))
feature_cols = fi_df.head(top_n)['feature'].tolist()
print(f"   선별 피처: {len(feature_cols)} / {len(all_feature_cols)}", flush=True)

X = train[feature_cols]
X_test = test[feature_cols]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. CV 전략 비교 (1-seed 빠른 테스트)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n3. CV 전략 비교 (LGB 1-seed)...", flush=True)

# 공통 LGB 파라미터 (정규화 강화)
lgb_params = dict(
    objective='mae', n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

# GroupKFold(scenario_id)
gkf_s = GroupKFold(n_splits=N_SPLITS)
oof_s = np.zeros(len(train))
for tr_idx, val_idx in gkf_s.split(X, y, groups_scenario):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_s[val_idx] = m.predict(X.iloc[val_idx])
mae_s = mean_absolute_error(y, oof_s)
print(f"  GroupKFold(scenario) OOF MAE: {mae_s:.4f}", flush=True)

# GroupKFold(layout_id)
gkf_l = GroupKFold(n_splits=N_SPLITS)
oof_l = np.zeros(len(train))
for tr_idx, val_idx in gkf_l.split(X, y, groups_layout):
    m = lgb.LGBMRegressor(**lgb_params)
    m.fit(X.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    oof_l[val_idx] = m.predict(X.iloc[val_idx])
mae_l = mean_absolute_error(y, oof_l)
print(f"  GroupKFold(layout)   OOF MAE: {mae_l:.4f}", flush=True)

# 더 나은 CV 선택
if mae_l > mae_s:
    print(f"  → layout CV가 더 보수적 ({mae_l:.4f} > {mae_s:.4f}) — layout 사용", flush=True)
    use_groups = groups_layout
    cv_name = 'layout'
else:
    print(f"  → scenario CV 유지 ({mae_s:.4f})", flush=True)
    use_groups = groups_scenario
    cv_name = 'scenario'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 본 학습 (3-seed × 3모델 + Huber)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n4. 본 학습 (CV={cv_name}, 3-seed)...", flush=True)


def train_model(model_type, seed, objective='mae'):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))

    for tr_idx, val_idx in gkf.split(X, y, use_groups):
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective=objective, n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(
                objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                max_depth=7, min_child_weight=10,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                tree_method='hist', random_state=seed, verbosity=0, n_jobs=-1,
                early_stopping_rounds=200,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], y.iloc[val_idx])], verbose=0)
        elif model_type == 'cb':
            model = CatBoostRegressor(
                loss_function='MAE', eval_metric='MAE',
                iterations=5000, learning_rate=0.03, depth=7,
                l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                random_seed=seed, verbose=0, early_stopping_rounds=200,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=(X.iloc[val_idx], y.iloc[val_idx]), verbose=0)
        elif model_type == 'lgb_huber':
            model = lgb.LGBMRegressor(
                objective='huber', n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=seed, verbose=-1, n_jobs=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
                      eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS

    return oof, tpred


models_config = [
    ('lgb', 'LGB_MAE'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
    ('lgb_huber', 'LGB_Huber'),
]

all_oofs = {}
all_tests = {}

for mtype, mname in models_config:
    print(f"\n  {mname} (3 seeds)...", flush=True)
    oofs, tests = [], []
    for seed in SEEDS:
        print(f"    Seed {seed}...", flush=True)
        o, t = train_model(mtype, seed)
        oofs.append(o)
        tests.append(t)
    all_oofs[mname] = np.mean(oofs, axis=0)
    all_tests[mname] = np.mean(tests, axis=0)
    print(f"  {mname} OOF MAE: {mean_absolute_error(y, all_oofs[mname]):.4f}", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n5. 가중 앙상블 최적화...", flush=True)

oof_list = [all_oofs[n] for _, n in models_config]
test_list = [all_tests[n] for _, n in models_config]
names = [n for _, n in models_config]


def ens_mae(w):
    w = np.array(w); w = w / w.sum()
    return mean_absolute_error(y, sum(wi * p for wi, p in zip(w, oof_list)))


res = minimize(ens_mae, x0=[0.25]*4, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = bw / bw.sum()

# 음수 가중치 제거 후 재최적화
bw = np.maximum(bw, 0)
bw = bw / bw.sum()
final_mae = mean_absolute_error(y, sum(wi * p for wi, p in zip(bw, oof_list)))

print(f"  가중 앙상블 OOF MAE: {final_mae:.4f}")
for n, w in zip(names, bw):
    print(f"    {n}: {w:.3f}")

# 3모델만 앙상블 (Huber 제외)도 비교
oof_3 = [all_oofs['LGB_MAE'], all_oofs['XGB'], all_oofs['CatBoost']]
test_3 = [all_tests['LGB_MAE'], all_tests['XGB'], all_tests['CatBoost']]

def ens3(w):
    w = np.array(w); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof_3)))

res3 = minimize(ens3, x0=[1/3]*3, method='Nelder-Mead')
bw3 = np.array(res3.x); bw3 = bw3 / bw3.sum()
mae3 = res3.fun
print(f"\n  3모델 앙상블 OOF MAE: {mae3:.4f}")
print(f"    LGB={bw3[0]:.3f}, XGB={bw3[1]:.3f}, CB={bw3[2]:.3f}")

# 더 좋은 것 선택
if final_mae <= mae3:
    final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
    best_mae = final_mae
    best_name = '4모델'
else:
    final_pred = np.clip(sum(wi*p for wi,p in zip(bw3, test_3)), 0, None)
    best_mae = mae3
    best_name = '3모델'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. 제출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sub = pd.DataFrame({'ID': test['ID'], TARGET: final_pred})
sub.to_csv('./submission_v8.csv', index=False)

print(f"\n{'='*60}")
print(f"v8 완료!")
print(f"  CV 전략: GroupKFold({cv_name})")
print(f"  피처 수: {len(feature_cols)}")
print(f"  앙상블: {best_name} (OOF MAE: {best_mae:.4f})")
print(f"  예측: mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
print(f"  submission_v8.csv 저장 완료")
print(f"{'='*60}", flush=True)
