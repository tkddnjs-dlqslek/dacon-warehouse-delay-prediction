"""
MLP 모델: v23 피처 150개로 학습 → GBDT와 블렌딩용
  - PyTorch MLP (3-layer, BatchNorm, Dropout)
  - 5-fold GroupKFold, log1p target
  - fold별 pickle + csv 저장
  - v23/v22_pre와 블렌딩 csv 생성
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
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

torch.manual_seed(SEED)
np.random.seed(SEED)

t0 = time.time()
print("=== MLP: GBDT 블렌딩용 Neural Network ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
print(f"  v23 피처: {len(v23_selected)}개", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: Feature Engineering (v23 동일)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링...", flush=True)

def engineer_features_v23(df, layout_df):
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

train_fe = engineer_features_v23(train, layout)
test_fe = engineer_features_v23(test, layout)
y = train_fe[TARGET].values
y_log = np.log1p(y)

feature_cols = [f for f in v23_selected if f in train_fe.columns]
print(f"  피처: {len(feature_cols)}개")

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

X_all = train_fe[feature_cols].values.astype(np.float32)
X_test_all = test_fe[feature_cols].values.astype(np.float32)

X_all = np.nan_to_num(X_all, 0)
X_test_all = np.nan_to_num(X_test_all, 0)

print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MLP 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MLPModel(nn.Module):
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

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 5-Fold Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] MLP 5-Fold Training...", flush=True)

EPOCHS = 50
BATCH_SIZE = 4096
LR = 0.001

mlp_oof = np.zeros(len(X_all))
mlp_test = np.zeros(len(X_test_all))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    t1 = time.time()
    print(f"\n  Fold {fold_idx}...", flush=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[tr_idx])
    X_val = scaler.transform(X_all[val_idx])
    X_te = scaler.transform(X_test_all)

    y_tr = y_log[tr_idx]
    y_val = y_log[val_idx]

    tr_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_tr))
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_tensor_x = torch.FloatTensor(X_val)
    te_tensor = torch.FloatTensor(X_te)

    model = MLPModel(X_tr.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.L1Loss()

    best_val_mae = 999
    best_state = None
    patience = 10
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(tr_dataset)

        model.eval()
        with torch.no_grad():
            val_pred_log = model(val_tensor_x).numpy()
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
            print(f"    Epoch {epoch+1:2d}: loss={train_loss:.4f}, val_MAE={val_mae:.4f}, best={best_val_mae:.4f}", flush=True)

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        mlp_oof[val_idx] = np.clip(np.expm1(model(val_tensor_x).numpy()), 0, None)
        mlp_test += np.clip(np.expm1(model(te_tensor).numpy()), 0, None) / N_SPLITS

    fold_mae = mean_absolute_error(y[val_idx], mlp_oof[val_idx])
    print(f"  Fold {fold_idx} MAE: {fold_mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    pickle.dump({'mlp_oof': mlp_oof.copy(), 'mlp_test': mlp_test.copy() * N_SPLITS / (fold_idx + 1)},
                open(f'{RESULT_DIR}/mlp_fold{fold_idx}.pkl', 'wb'))

    partial_test = mlp_test * N_SPLITS / (fold_idx + 1) if fold_idx < N_SPLITS - 1 else mlp_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: partial_test}).to_csv(
        f'./submission_mlp_fold{fold_idx}.csv', index=False)

mlp_oof_mae = mean_absolute_error(y, mlp_oof)
print(f"\n  MLP OOF MAE: {mlp_oof_mae:.4f}", flush=True)

pd.DataFrame({'ID': test_fe['ID'], TARGET: mlp_test}).to_csv('./submission_mlp.csv', index=False)
pickle.dump({'mlp_oof': mlp_oof, 'mlp_test': mlp_test, 'mlp_oof_mae': mlp_oof_mae},
            open(f'{RESULT_DIR}/mlp_final.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: GBDT 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] GBDT 블렌딩...", flush=True)

s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
w = s42['weights']
names = list(s42['oofs'].keys())
v23_oof = np.clip(sum(wi*p for wi, p in zip(w, [s42['oofs'][n] for n in names])), 0, None)
v23_test = np.clip(sum(wi*p for wi, p in zip(w, [s42['tests'][n] for n in names])), 0, None)
v22_pre_test = pd.read_csv('submission_v22_pre.csv')['avg_delay_minutes_next_30m'].values
v24_test = pd.read_csv('submission_v24.csv')['avg_delay_minutes_next_30m'].values

from scipy.stats import pearsonr
r_v23, _ = pearsonr(mlp_test, v23_test)
r_v22, _ = pearsonr(mlp_test, v22_pre_test)
print(f"  MLP vs v23 상관: {r_v23:.4f}")
print(f"  MLP vs v22_pre 상관: {r_v22:.4f}")

# v23 + MLP 최적 비율
print("\n  v23 + MLP 블렌딩 (OOF):")
best_blend_mae = 999
best_ratio = 0
for ratio in np.arange(0, 0.51, 0.02):
    blended_oof = (1 - ratio) * v23_oof + ratio * mlp_oof
    mae = mean_absolute_error(y, blended_oof)
    if mae < best_blend_mae:
        best_blend_mae = mae
        best_ratio = ratio
    if ratio in [0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        print(f"    MLP {ratio:.0%}: OOF MAE={mae:.4f}", flush=True)

print(f"  -> 최적: MLP {best_ratio:.0%} (OOF MAE={best_blend_mae:.4f})")
print(f"  -> v23 단독 대비: {mean_absolute_error(y, v23_oof) - best_blend_mae:+.4f}")

# 블렌딩 csv
for ratio in [0.05, 0.10, 0.15, 0.20, best_ratio]:
    ratio = round(ratio, 2)
    blend = (1 - ratio) * v23_test + ratio * mlp_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_v23_mlp_{int(ratio*100)}.csv', index=False)

# 3-way: v23 + v22_pre + MLP
for r_mlp in [0.05, 0.10, 0.15]:
    r_v23_val = 1.0 - 0.20 - r_mlp
    blend3 = r_v23_val * v23_test + 0.20 * v22_pre_test + r_mlp * mlp_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend3}).to_csv(
        f'./submission_3way_mlp_{int(r_mlp*100)}.csv', index=False)

# 4-way: v23 + v22_pre + v24 + MLP
blend4 = 0.55 * v23_test + 0.15 * v22_pre_test + 0.15 * v24_test + 0.15 * mlp_test
pd.DataFrame({'ID': test_fe['ID'], TARGET: blend4}).to_csv(
    './submission_4way_blend.csv', index=False)

# best 3-way 기반 + MLP
# 현재 best: v23 65% + v22 20% + v24 15% = 9.930
# 여기에 MLP 추가
for r_mlp in [0.05, 0.10]:
    scale = (1 - r_mlp)
    blend_best = scale * (0.65 * v23_test + 0.20 * v22_pre_test + 0.15 * v24_test) + r_mlp * mlp_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend_best}).to_csv(
        f'./submission_best3way_mlp_{int(r_mlp*100)}.csv', index=False)

pickle.dump({
    'mlp_oof': mlp_oof, 'mlp_test': mlp_test,
    'r_v23': r_v23, 'r_v22': r_v22,
    'best_ratio': best_ratio, 'best_blend_mae': best_blend_mae,
}, open(f'{RESULT_DIR}/mlp_blend.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"MLP 완료!")
print(f"  MLP OOF MAE: {mlp_oof_mae:.4f}")
print(f"  MLP vs v23 상관: {r_v23:.4f}")
print(f"  MLP vs v22 상관: {r_v22:.4f}")
print(f"  최적 v23+MLP: MLP {best_ratio:.0%} (OOF {best_blend_mae:.4f})")
print(f"  submission_mlp.csv (단독)")
print(f"  submission_v23_mlp_*.csv (2-way)")
print(f"  submission_3way_mlp_*.csv (3-way)")
print(f"  submission_4way_blend.csv (4-way)")
print(f"  submission_best3way_mlp_*.csv (best+MLP)")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
