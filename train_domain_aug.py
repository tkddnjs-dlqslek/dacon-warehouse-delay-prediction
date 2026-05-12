"""
Domain Features + Data Augmentation
  Strategy A: 미사용 컬럼 lag/rolling (5개 추가)
  Strategy B: 도메인 interaction 피처 (7개)
  Strategy C: MLP with noise augmentation
  → Mega-stacking 확장
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')
torch.manual_seed(42)
np.random.seed(42)

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'

t0 = time.time()
print("=== Domain Features + Data Augmentation ===", flush=True)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

def engineer_v23_base(df, layout_df):
    """v23 동일 피처 생성"""
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
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_lead2'] = g.shift(-2)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]
    extra_cols = ['max_zone_density', 'robot_charging', 'low_battery_ratio',
                  'robot_idle', 'near_collision_15m']
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_lead1'] = g.shift(-1)
        df[f'{col}_diff_lead1'] = df[f'{col}_lead1'] - df[col]
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

    # ===== Strategy A: 미사용 컬럼 lag/rolling =====
    new_cols = ['avg_charge_wait', 'unique_sku_15m', 'avg_recovery_time', 'robot_active', 'task_reassign_15m']
    for col in new_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())

    # ===== Strategy B: 도메인 interaction =====
    df['charge_urgency'] = (df['low_battery_ratio'] * df['charge_queue_length']) / (df['charger_count'] + 1)
    df['pipeline_bottleneck'] = df[['pack_utilization', 'loading_dock_util', 'staging_area_util']].max(axis=1)
    df['robot_efficiency'] = df['robot_active'] / (df['robot_active'] + df['robot_idle'] + df['robot_charging'] + 1)
    df['order_labor_stress'] = (df['order_inflow_15m'] * df['avg_items_per_order']) / (df['staff_on_floor'] + 1)
    df['fault_recovery_load'] = df['fault_count_15m'] * df['avg_recovery_time']
    df['cumulative_fault'] = group['fault_count_15m'].cumsum()
    df['congestion_worst_sofar'] = group['congestion_score'].cummax()

    # 제거
    layout_static = ['layout_type', 'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                     'pack_station_count', 'charger_count', 'layout_compactness', 'zone_dispersion',
                     'robot_total', 'building_age_years', 'floor_area_sqm', 'ceiling_height_m',
                     'fire_sprinkler_count', 'emergency_exit_count']
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors='ignore')
    corr_remove = ['battery_mean_rmean3', 'charge_queue_length_rmean3', 'battery_mean_rmean5',
                   'charge_queue_length_rmean5', 'pack_utilization_rmean5', 'battery_mean_lag1',
                   'charge_queue_length_lag1', 'congestion_score_rmean3', 'order_inflow_15m_cummean',
                   'robot_utilization_rmean5', 'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
                   'battery_risk', 'congestion_score_rmean5', 'pack_utilization_rmean3',
                   'order_inflow_15m_rmean3', 'charge_queue_length_lag2', 'blocked_path_15m_rmean5']
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')
    return df

train_fe = engineer_v23_base(train, layout)
test_fe = engineer_v23_base(test, layout)
y = train_fe[TARGET].values
y_log = np.log1p(y)

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']

exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
all_cols = [c for c in train_fe.columns if c not in exclude]

# v23 기존 150개
v23_feats = [f for f in v23_selected if f in train_fe.columns]

# Strategy A 신규 피처
strategy_a_new = [c for c in all_cols if any(c.startswith(nc + '_') for nc in
                  ['avg_charge_wait', 'unique_sku_15m', 'avg_recovery_time', 'robot_active', 'task_reassign_15m'])]

# Strategy B 신규 피처
strategy_b_new = ['charge_urgency', 'pipeline_bottleneck', 'robot_efficiency',
                  'order_labor_stress', 'fault_recovery_load', 'cumulative_fault', 'congestion_worst_sofar']
strategy_b_new = [c for c in strategy_b_new if c in train_fe.columns]

print(f"  v23 base: {len(v23_feats)}개")
print(f"  Strategy A 신규: {len(strategy_a_new)}개")
print(f"  Strategy B 신규: {len(strategy_b_new)}개")

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1A/B: Feature Selection (개별 검증)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] 신규 피처 개별 검증...", flush=True)

tr_idx, val_idx = folds[0]

# Baseline: v23 only
m_base = lgb.LGBMRegressor(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=4)
m_base.fit(train_fe[v23_feats].iloc[tr_idx], y_log[tr_idx],
           eval_set=[(train_fe[v23_feats].iloc[val_idx], y_log[val_idx])],
           callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
pred_base = np.clip(np.expm1(m_base.predict(train_fe[v23_feats].iloc[val_idx])), 0, None)
base_mae = mean_absolute_error(y[val_idx], pred_base)
print(f"  v23 baseline fold0: {base_mae:.4f}")

# 개별 검증 — Strategy A
print("\n  Strategy A 개별 검증:")
good_a = []
for nf in strategy_a_new:
    feats = v23_feats + [nf]
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(train_fe[feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(train_fe[feats].iloc[val_idx])), 0, None)
    mae = mean_absolute_error(y[val_idx], pred)
    diff = base_mae - mae
    flag = "+" if diff > 0.001 else "-"
    print(f"    {flag} {nf:40s} MAE={mae:.4f} ({diff:+.4f})", flush=True)
    if diff > 0.001:
        good_a.append(nf)

# 개별 검증 — Strategy B
print("\n  Strategy B 개별 검증:")
good_b = []
for nf in strategy_b_new:
    feats = v23_feats + [nf]
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(train_fe[feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(train_fe[feats].iloc[val_idx])), 0, None)
    mae = mean_absolute_error(y[val_idx], pred)
    diff = base_mae - mae
    flag = "+" if diff > 0.001 else "-"
    print(f"    {flag} {nf:40s} MAE={mae:.4f} ({diff:+.4f})", flush=True)
    if diff > 0.001:
        good_b.append(nf)

# 전체 추가 검증
all_good = good_a + good_b
final_feats = v23_feats + all_good
if all_good:
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(train_fe[final_feats].iloc[tr_idx], y_log[tr_idx],
          eval_set=[(train_fe[final_feats].iloc[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    pred = np.clip(np.expm1(m.predict(train_fe[final_feats].iloc[val_idx])), 0, None)
    mae = mean_absolute_error(y[val_idx], pred)
    print(f"\n  전체 추가 fold0: {mae:.4f} (diff {base_mae - mae:+.4f})")
    if mae >= base_mae:
        print(f"  → 전체 추가 시 악화, v23 base 유지")
        final_feats = v23_feats
else:
    print("\n  개선 피처 없음, v23 base 유지")
    final_feats = v23_feats

print(f"  최종 피처: {len(final_feats)}개")
pickle.dump({'final_feats': final_feats, 'good_a': good_a, 'good_b': good_b},
            open(f'{RESULT_DIR}/domain_phase1.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: GBDT 학습 (신규 피처 포함)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] GBDT 학습 (신규 피처 포함)...", flush=True)

X_f = train_fe[final_feats].values
X_f_test = test_fe[final_feats].values
X_f = np.nan_to_num(X_f, 0)
X_f_test = np.nan_to_num(X_f_test, 0)

# v23 base와 동일하면 학습 스킵
skip_gbdt = (final_feats == v23_feats)
if skip_gbdt:
    print("  → v23와 동일, GBDT 재학습 스킵")
    domain_oofs = {}
    domain_tests = {}
else:
    domain_oofs = {}
    domain_tests = {}

    for mtype, mname in [('lgb_huber', 'LGB_Huber'), ('xgb', 'XGB'), ('cb', 'CatBoost')]:
        t1 = time.time()
        print(f"  {mname}...", end='', flush=True)

        oof = np.zeros(len(y))
        tpred = np.zeros(len(X_f_test))

        for tr_idx, val_idx in folds:
            if mtype == 'lgb_huber':
                model = lgb.LGBMRegressor(
                    objective='huber', n_estimators=5000, learning_rate=0.03,
                    num_leaves=63, max_depth=8, min_child_samples=50,
                    subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0,
                    random_state=SEED, verbose=-1, n_jobs=4)
                model.fit(X_f[tr_idx], y_log[tr_idx],
                          eval_set=[(X_f[val_idx], y_log[val_idx])],
                          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            elif mtype == 'xgb':
                model = xgb.XGBRegressor(
                    objective='reg:absoluteerror', n_estimators=5000, learning_rate=0.03,
                    max_depth=7, min_child_weight=10, subsample=0.7, colsample_bytree=0.7,
                    reg_alpha=1.0, reg_lambda=1.0, tree_method='hist',
                    random_state=SEED, verbosity=0, n_jobs=4, early_stopping_rounds=200)
                model.fit(X_f[tr_idx], y_log[tr_idx],
                          eval_set=[(X_f[val_idx], y_log[val_idx])], verbose=0)
            elif mtype == 'cb':
                model = CatBoostRegressor(
                    loss_function='MAE', eval_metric='MAE',
                    iterations=5000, learning_rate=0.03, depth=7,
                    l2_leaf_reg=5.0, random_strength=1.0, bagging_temperature=1.0,
                    random_seed=SEED, verbose=0, early_stopping_rounds=200, thread_count=4, task_type='CPU')
                model.fit(X_f[tr_idx], y_log[tr_idx],
                          eval_set=(X_f[val_idx], y_log[val_idx]), verbose=0)

            oof[val_idx] = np.expm1(model.predict(X_f[val_idx]))
            tpred += np.expm1(model.predict(X_f_test)) / N_SPLITS

        oof = np.clip(oof, 0, None)
        tpred = np.clip(tpred, 0, None)
        domain_oofs[mname] = oof
        domain_tests[mname] = tpred
        mae = mean_absolute_error(y, oof)
        print(f" OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

        pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(
            f'./submission_domain_{mname}.csv', index=False)

    pickle.dump({'oofs': domain_oofs, 'tests': domain_tests}, open(f'{RESULT_DIR}/domain_phase2.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: MLP with Noise Augmentation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 3] MLP with Noise Augmentation...", flush=True)

X_mlp = np.nan_to_num(train_fe[v23_feats].values.astype(np.float32), 0)
X_mlp_test = np.nan_to_num(test_fe[v23_feats].values.astype(np.float32), 0)

class MLPNoise(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

mlp_aug_oof = np.zeros(len(X_mlp))
mlp_aug_test = np.zeros(len(X_mlp_test))

NOISE_STD = 0.05  # 5% of feature std

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    t1 = time.time()
    print(f"\n  Fold {fold_idx}...", flush=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_mlp[tr_idx])
    X_val = scaler.transform(X_mlp[val_idx])
    X_te = scaler.transform(X_mlp_test)

    tr_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_log[tr_idx]))
    tr_loader = DataLoader(tr_dataset, batch_size=4096, shuffle=True)

    val_tensor = torch.FloatTensor(X_val)
    te_tensor = torch.FloatTensor(X_te)

    model = MLPNoise(X_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
    loss_fn = nn.L1Loss()

    best_val_mae = 999
    best_state = None
    no_improve = 0

    for epoch in range(50):
        model.train()
        for xb, yb in tr_loader:
            # NOISE INJECTION
            noise = torch.randn_like(xb) * NOISE_STD
            xb_aug = xb + noise
            optimizer.zero_grad()
            pred = model(xb_aug)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred_log = model(val_tensor).numpy()
            val_pred = np.clip(np.expm1(val_pred_log), 0, None)
            val_mae = mean_absolute_error(y[val_idx], val_pred)

        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: val_MAE={val_mae:.4f}, best={best_val_mae:.4f}", flush=True)

        if no_improve >= 10:
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mlp_aug_oof[val_idx] = np.clip(np.expm1(model(val_tensor).numpy()), 0, None)
        mlp_aug_test += np.clip(np.expm1(model(te_tensor).numpy()), 0, None) / N_SPLITS

    print(f"  Fold {fold_idx} MAE: {best_val_mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    pickle.dump({'mlp_aug_oof': mlp_aug_oof.copy(), 'mlp_aug_test': mlp_aug_test.copy() * N_SPLITS / (fold_idx + 1)},
                open(f'{RESULT_DIR}/mlp_aug_fold{fold_idx}.pkl', 'wb'))
    partial = mlp_aug_test * N_SPLITS / (fold_idx + 1) if fold_idx < N_SPLITS - 1 else mlp_aug_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: partial}).to_csv(
        f'./submission_mlp_aug_fold{fold_idx}.csv', index=False)

mlp_aug_mae = mean_absolute_error(y, mlp_aug_oof)
print(f"\n  MLP_aug OOF: {mlp_aug_mae:.4f}")
pd.DataFrame({'ID': test_fe['ID'], TARGET: mlp_aug_test}).to_csv('./submission_mlp_aug.csv', index=False)
pickle.dump({'mlp_aug_oof': mlp_aug_oof, 'mlp_aug_test': mlp_aug_test, 'mae': mlp_aug_mae},
            open(f'{RESULT_DIR}/mlp_aug_final.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 4: Mega-Stacking 확장
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 4] Mega-Stacking 확장...", flush=True)

selected_oofs = {}
selected_tests = {}

# 기존 20개
for seed in [42, 123, 2024]:
    s = pickle.load(open(f'{RESULT_DIR}/v23_seed{seed}.pkl', 'rb'))
    for name in ['LGB_Huber', 'XGB', 'CatBoost']:
        selected_oofs[f'v23s{seed}_{name}'] = s['oofs'][name]
        selected_tests[f'v23s{seed}_{name}'] = s['tests'][name]

v24 = pickle.load(open(f'{RESULT_DIR}/v24_final.pkl', 'rb'))
for name, oof in v24['oofs'].items():
    selected_oofs[f'v24_{name}'] = oof
    selected_tests[f'v24_{name}'] = v24['tests'][name]

v26 = pickle.load(open(f'{RESULT_DIR}/v26_final.pkl', 'rb'))
for name in ['Tuned_Huber', 'Tuned_sqrt', 'Tuned_pow', 'DART']:
    selected_oofs[f'v26_{name}'] = v26['oofs'][name]
    selected_tests[f'v26_{name}'] = v26['tests'][name]

mlp1 = pickle.load(open(f'{RESULT_DIR}/mlp_final.pkl', 'rb'))
mlp2 = pickle.load(open(f'{RESULT_DIR}/mlp2_final.pkl', 'rb'))
cnn = pickle.load(open(f'{RESULT_DIR}/cnn_final.pkl', 'rb'))
selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']

# 신규 추가
if not skip_gbdt:
    for mname, oof in domain_oofs.items():
        selected_oofs[f'domain_{mname}'] = oof
        selected_tests[f'domain_{mname}'] = domain_tests[mname]

selected_oofs['mlp_aug'] = mlp_aug_oof
selected_tests['mlp_aug'] = mlp_aug_test

print(f"  총 {len(selected_oofs)}개 모델")

stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])

mega_oof = np.zeros(len(y))
mega_test = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=500, learning_rate=0.05,
        num_leaves=15, max_depth=4, min_child_samples=100,
        random_state=SEED, verbose=-1, n_jobs=4)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    mega_oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    mega_test += np.expm1(m.predict(stack_test)) / N_SPLITS
mega_oof = np.clip(mega_oof, 0, None)
mega_test = np.clip(mega_test, 0, None)
mega_mae = mean_absolute_error(y, mega_oof)
print(f"  Mega+domain+aug OOF: {mega_mae:.4f}")

pd.DataFrame({'ID': test_fe['ID'], TARGET: mega_test}).to_csv(
    './submission_megastack_domain.csv', index=False)

# 블렌딩
prev_best = pd.read_csv('./submission_megastack_lgb_15_4.csv')[TARGET].values
for ratio in [0.3, 0.5, 0.7]:
    blend = ratio * mega_test + (1 - ratio) * prev_best
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_megastack_domain_prev_{int(ratio*100)}.csv', index=False)

print(f"\n{'='*60}")
print(f"Domain + Augmentation 완료!")
print(f"  최종 피처: {len(final_feats)}개")
print(f"  good_a: {len(good_a)}, good_b: {len(good_b)}")
if not skip_gbdt:
    for mname, oof in domain_oofs.items():
        print(f"  domain_{mname}: OOF={mean_absolute_error(y, oof):.4f}")
print(f"  MLP_aug OOF: {mlp_aug_mae:.4f}")
print(f"  Mega+domain+aug OOF: {mega_mae:.4f}")
print(f"  v23 대비: {8.5787 - mega_mae:+.4f}")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
