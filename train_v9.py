"""
v9: unseen layout 대응 강화
- layout 정적 피처 × 운영 피처 교차
- 운영 피처를 layout 특성으로 정규화
- DART boosting 추가
- GroupKFold(layout_id)
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
# 2. 피처 엔지니어링
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

    # lag/rolling (v7/v8과 동일)
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

    # ── 기본 상호작용 (v8 동일) ──
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
    df['total_utilization'] = (df['pack_utilization'] + df['staging_area_util'] + df['loading_dock_util']) / 3
    df['fault_x_congestion'] = df['fault_count_15m'] * df['congestion_score']
    df['battery_charge_pressure'] = df['low_battery_ratio'] * df['avg_charge_wait']
    df['congestion_per_robot'] = df['congestion_score'] / (df['robot_active'] + 1)
    df['order_per_staff'] = df['order_inflow_15m'] / (df['staff_on_floor'] + 1)

    # ── Layout 기반 비율 (v8 동일) ──
    if 'floor_area_sqm' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
        df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # v9 신규: unseen layout 대응 피처
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # (1) 운영 피처를 layout 특성으로 정규화
    # → unseen layout에서도 "이 창고 규모 대비 얼마나 바쁜가"를 포착
    df['order_per_area'] = df['order_inflow_15m'] / (df['floor_area_sqm'] + 1) * 1000
    df['congestion_per_area'] = df['congestion_score'] / (df['floor_area_sqm'] + 1) * 1000
    df['fault_per_robot_total'] = df['fault_count_15m'] / (df['robot_total'] + 1)
    df['blocked_per_robot_total'] = df['blocked_path_15m'] / (df['robot_total'] + 1)
    df['collision_per_robot_total'] = df['near_collision_15m'] / (df['robot_total'] + 1)
    df['pack_util_per_station'] = df['pack_utilization'] / (df['pack_station_count'] + 1)
    df['charge_queue_per_charger'] = df['charge_queue_length'] / (df['charger_count'] + 1)
    df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)

    # (2) layout 정적 피처 × 운영 피처 교차
    # → "좁은 통로(aisle_width)에서 혼잡도가 높으면 더 위험" 같은 패턴
    df['congestion_x_aisle_width'] = df['congestion_score'] * df['aisle_width_avg']
    df['congestion_x_compactness'] = df['congestion_score'] * df['layout_compactness']
    df['order_x_zone_dispersion'] = df['order_inflow_15m'] * df['zone_dispersion']
    df['blocked_x_one_way'] = df['blocked_path_15m'] * df['one_way_ratio']
    df['utilization_x_compactness'] = df['robot_utilization'] * df['layout_compactness']
    df['battery_risk_x_charger_ratio'] = df['battery_risk'] * df['charger_ratio']

    # (3) layout 피처 간 비율 (추가)
    df['intersection_density'] = df['intersection_count'] / (df['floor_area_sqm'] + 1) * 1000
    df['pack_charger_ratio'] = df['pack_station_count'] / (df['charger_count'] + 1)
    df['exit_density'] = df['emergency_exit_count'] / (df['floor_area_sqm'] + 1) * 10000
    df['sprinkler_density'] = df['fire_sprinkler_count'] / (df['floor_area_sqm'] + 1) * 10000
    df['height_x_area'] = df['ceiling_height_m'] * df['floor_area_sqm']

    return df


train = engineer_features(train, layout)
test = engineer_features(test, layout)

exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
all_feature_cols = [c for c in train.columns if c not in exclude]
print(f"   전체 피처 수: {len(all_feature_cols)}", flush=True)

y = train[TARGET]
groups_layout = train['layout_id']

# ── FI 기반 피처 선별 (상위 150개) ──
print("2.5. FI 기반 피처 선별...", flush=True)
X_all = train[all_feature_cols]
gkf_fi = GroupKFold(n_splits=N_SPLITS)
fi_sum = np.zeros(len(all_feature_cols))
for tr_idx, val_idx in gkf_fi.split(X_all, y, groups_layout):
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=1000, learning_rate=0.05,
        num_leaves=63, random_state=42, verbose=-1, n_jobs=-1,
    )
    m.fit(X_all.iloc[tr_idx], y.iloc[tr_idx],
          eval_set=[(X_all.iloc[val_idx], y.iloc[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    fi_sum += m.feature_importances_

fi_df = pd.DataFrame({'feature': all_feature_cols, 'importance': fi_sum}).sort_values('importance', ascending=False)
top_n = min(150, len(fi_df[fi_df['importance'] > 0]))
feature_cols = fi_df.head(top_n)['feature'].tolist()
print(f"   선별 피처: {len(feature_cols)} / {len(all_feature_cols)}", flush=True)

# v9 신규 피처가 FI 상위에 있는지 확인
v9_new = [c for c in feature_cols if any(k in c for k in ['per_area', 'per_station', 'per_charger', '_x_aisle', '_x_compact', '_x_zone', '_x_one_way', 'intersection_density', 'pack_charger', 'height_x_area'])]
print(f"   v9 신규 피처 중 선별됨: {len(v9_new)}개", flush=True)

X = train[feature_cols]
X_test = test[feature_cols]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 모델 학습 (GroupKFold layout, 3-seed)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n3. 본 학습 (CV=layout, 3-seed)...", flush=True)


def train_model(model_type, seed):
    gkf = GroupKFold(n_splits=N_SPLITS)
    oof = np.zeros(len(train))
    tpred = np.zeros(len(test))
    for tr_idx, val_idx in gkf.split(X, y, groups_layout):
        if model_type == 'lgb':
            model = lgb.LGBMRegressor(
                objective='mae', n_estimators=5000, learning_rate=0.03,
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
        elif model_type == 'lgb_dart':
            model = lgb.LGBMRegressor(
                objective='mae', boosting_type='dart',
                n_estimators=1000, learning_rate=0.05,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                drop_rate=0.1, skip_drop=0.5,
                random_state=seed, verbose=-1, n_jobs=-1,
            )
            model.fit(X.iloc[tr_idx], y.iloc[tr_idx])

        oof[val_idx] = model.predict(X.iloc[val_idx])
        tpred += model.predict(X_test) / N_SPLITS
    return oof, tpred


models_config = [
    ('lgb', 'LGB_MAE'),
    ('lgb_huber', 'LGB_Huber'),
    ('xgb', 'XGB'),
    ('cb', 'CatBoost'),
    ('lgb_dart', 'LGB_DART'),
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
# 4. 앙상블
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n4. 가중 앙상블 최적화...", flush=True)

names = [n for _, n in models_config]
oof_list = [all_oofs[n] for n in names]
test_list = [all_tests[n] for n in names]


def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi * p for wi, p in zip(w, oof_list)))


res = minimize(ens_mae, x0=[0.2]*5, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
final_mae = mean_absolute_error(y, sum(wi * p for wi, p in zip(bw, oof_list)))

print(f"  5모델 앙상블 OOF MAE: {final_mae:.4f}")
for n, w in zip(names, bw):
    print(f"    {n}: {w:.3f}")

# 4모델 (DART 제외) 비교
names4 = ['LGB_MAE', 'LGB_Huber', 'XGB', 'CatBoost']
oof4 = [all_oofs[n] for n in names4]
test4 = [all_tests[n] for n in names4]

def ens4(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oof4)))

res4 = minimize(ens4, x0=[0.25]*4, method='Nelder-Mead', options={'maxiter': 50000})
bw4 = np.array(res4.x); bw4 = np.maximum(bw4, 0); bw4 = bw4 / bw4.sum()
mae4 = mean_absolute_error(y, sum(wi*p for wi,p in zip(bw4, oof4)))
print(f"\n  4모델 앙상블 OOF MAE: {mae4:.4f}")

# 최적 선택
if final_mae <= mae4:
    final_pred = np.clip(sum(wi*p for wi,p in zip(bw, test_list)), 0, None)
    best_mae = final_mae
    best_name = '5모델'
else:
    final_pred = np.clip(sum(wi*p for wi,p in zip(bw4, test4)), 0, None)
    best_mae = mae4
    best_name = '4모델'

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 제출
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
sub = pd.DataFrame({'ID': test['ID'], TARGET: final_pred})
sub.to_csv('./submission_v9.csv', index=False)

print(f"\n{'='*60}")
print(f"v9 완료!")
print(f"  CV: GroupKFold(layout)")
print(f"  피처: {len(feature_cols)}개 (v9 신규 {len(v9_new)}개 포함)")
print(f"  앙상블: {best_name} (OOF MAE: {best_mae:.4f})")
print(f"  예측: mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
print(f"  submission_v9.csv 저장 완료")
print(f"{'='*60}", flush=True)
