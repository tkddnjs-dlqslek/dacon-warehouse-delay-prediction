"""
MLP Deep Army — 4 focused 모델 추가 → mega37 stacking
  1. mlp_deep_s2 (seed=123)
  2. mlp_deep_s3 (seed=2024)
  3. mlp_deep_gelu (activation 변경)
  4. Transformer Encoder (proper config)
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
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
RESULT_DIR = './results'

t0 = time.time()
print("=== MLP Deep Army (4 모델) ===", flush=True)

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

def engineer_v23(df, layout_df):
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

train_fe = engineer_v23(train, layout)
test_fe = engineer_v23(test, layout)
y = train_fe[TARGET].values
y_log = np.log1p(y)

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']
feature_cols = [f for f in v23_selected if f in train_fe.columns]
print(f"  v23 features: {len(feature_cols)}")

X_all = np.nan_to_num(train_fe[feature_cols].values.astype(np.float32), 0)
X_test_all = np.nan_to_num(test_fe[feature_cols].values.astype(np.float32), 0)

groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 모델 정의
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class MLPDeep(nn.Module):
    """기존 mlp_deep와 동일 구조 (seed만 다름)"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

class MLPDeepGELU(nn.Module):
    """mlp_deep + GELU activation"""
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)

# Transformer Encoder for sequences
class TransformerEncoder(nn.Module):
    def __init__(self, n_feat, d_model=128, n_heads=4, n_layers=4, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(n_feat, d_model)
        self.pos_emb = nn.Parameter(torch.randn(1, 25, d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*4, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
    def forward(self, x):
        # x: (batch, 25, n_feat)
        x = self.input_proj(x) + self.pos_emb
        x = self.transformer(x)
        x = self.norm(x)
        return self.head(x).squeeze(-1)  # (batch, 25)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Row-level 학습 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def train_row(name, model_cls, seed=42, EPOCHS=50):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t1 = time.time()
    print(f"\n  [{name}] training (seed={seed})...", flush=True)

    oof = np.zeros(len(X_all))
    tpred = np.zeros(len(X_test_all))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[tr_idx])
        X_val = scaler.transform(X_all[val_idx])
        X_te = scaler.transform(X_test_all)

        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_log[tr_idx]))
        tr_ld = DataLoader(tr_ds, batch_size=4096, shuffle=True)

        val_x = torch.FloatTensor(X_val)
        te_x = torch.FloatTensor(X_te)

        model = model_cls(X_tr.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.L1Loss()

        best_mae = 999
        best_state = None
        no_imp = 0

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                opt.step()

            model.eval()
            with torch.no_grad():
                vp = np.clip(np.expm1(model(val_x).numpy()), 0, None)
                vm = mean_absolute_error(y[val_idx], vp)
            sch.step()

            if vm < best_mae:
                best_mae = vm
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 8: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oof[val_idx] = np.clip(np.expm1(model(val_x).numpy()), 0, None)
            tpred += np.clip(np.expm1(model(te_x).numpy()), 0, None) / N_SPLITS

        print(f"    Fold {fold_idx}: {best_mae:.4f}", flush=True)
        pickle.dump({'oof': oof.copy(), 'test': tpred.copy() * N_SPLITS / (fold_idx + 1)},
                    open(f'{RESULT_DIR}/{name}_fold{fold_idx}.pkl', 'wb'))

    mae = mean_absolute_error(y, oof)
    print(f"  [{name}] OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred}).to_csv(f'./submission_{name}.csv', index=False)
    pickle.dump({'oof': oof, 'test': tpred, 'mae': mae},
                open(f'{RESULT_DIR}/{name}_final.pkl', 'wb'))
    return oof, tpred, mae

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Sequence 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n  시퀀스 변환...", flush=True)
seq_cols = [c for c in train.columns if c not in ['ID', 'layout_id', 'scenario_id', TARGET]]

train_raw_sorted = train.sort_values(['layout_id','scenario_id']).reset_index(drop=True)
train_raw_sorted['timeslot'] = train_raw_sorted.groupby(['layout_id','scenario_id']).cumcount()
test_raw_sorted = test.sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_raw_sorted['timeslot'] = test_raw_sorted.groupby(['layout_id','scenario_id']).cumcount()

def make_sequences(df, features):
    groups_gp = df.groupby(['layout_id','scenario_id'])
    X_seqs = []
    meta = []
    for (lid, sid), g in groups_gp:
        vals = g[features].values.astype(np.float32)
        if len(vals) < 25:
            vals = np.vstack([vals, np.zeros((25-len(vals), len(features)), dtype=np.float32)])
        elif len(vals) > 25:
            vals = vals[:25]
        X_seqs.append(vals)
        meta.append((lid, sid, g.index.values))
    return np.array(X_seqs), meta

X_train_seq, train_meta = make_sequences(train_raw_sorted, seq_cols)
X_test_seq, test_meta = make_sequences(test_raw_sorted, seq_cols)
X_train_seq = np.nan_to_num(X_train_seq, 0)
X_test_seq = np.nan_to_num(X_test_seq, 0)
print(f"  Sequence: train {X_train_seq.shape}, test {X_test_seq.shape}")

train_layout_ids = np.array([m[0] for m in train_meta])
gkf_seq = GroupKFold(n_splits=N_SPLITS)
folds_seq = list(gkf_seq.split(X_train_seq, np.arange(len(X_train_seq)), groups=train_layout_ids))

def train_seq(name, model_cls, seed=42, EPOCHS=30):
    torch.manual_seed(seed)
    np.random.seed(seed)
    t1 = time.time()
    print(f"\n  [{name}] sequence training (seed={seed})...", flush=True)

    n_feat = X_train_seq.shape[2]
    y_train_seq = np.zeros((len(X_train_seq), 25))
    for i, m in enumerate(train_meta):
        row_idx = m[2]
        vals = y[row_idx] if len(row_idx) <= 25 else y[row_idx[:25]]
        if len(vals) < 25:
            vals = np.pad(vals, (0, 25 - len(vals)), constant_values=0)
        y_train_seq[i] = vals
    y_train_seq_log = np.log1p(y_train_seq).astype(np.float32)

    oof_rows = np.zeros(len(X_all))
    tpred_rows = np.zeros(len(X_test_all))

    for fold_idx, (tr_idx, val_idx) in enumerate(folds_seq):
        X_tr = X_train_seq[tr_idx]
        X_val = X_train_seq[val_idx]

        scaler = StandardScaler()
        scaler.fit(X_tr.reshape(-1, n_feat))
        X_tr = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape).astype(np.float32)
        X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape).astype(np.float32)
        X_te = scaler.transform(X_test_seq.reshape(-1, n_feat)).reshape(X_test_seq.shape).astype(np.float32)

        tr_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_train_seq_log[tr_idx]))
        tr_ld = DataLoader(tr_ds, batch_size=256, shuffle=True)

        model = model_cls(n_feat)
        opt = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        loss_fn = nn.L1Loss()

        best_mae = 999
        best_state = None
        no_imp = 0
        val_x = torch.FloatTensor(X_val)

        for epoch in range(EPOCHS):
            model.train()
            for xb, yb in tr_ld:
                opt.zero_grad()
                loss = loss_fn(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            model.eval()
            with torch.no_grad():
                vp = np.clip(np.expm1(model(val_x).numpy()), 0, None)
                val_row_idx = np.concatenate([train_meta[i][2] for i in val_idx])
                val_row_preds = []
                for i, vi in enumerate(val_idx):
                    n_r = len(train_meta[vi][2])
                    val_row_preds.append(vp[i, :n_r])
                val_row_preds = np.concatenate(val_row_preds)
                vm = mean_absolute_error(y[val_row_idx], val_row_preds)
            sch.step()

            if vm < best_mae:
                best_mae = vm
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
            if no_imp >= 7: break

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            vp = np.clip(np.expm1(model(val_x).numpy()), 0, None)
            for i, vi in enumerate(val_idx):
                row_idx = train_meta[vi][2]
                n_r = len(row_idx)
                oof_rows[row_idx] = vp[i, :n_r]

            te_x = torch.FloatTensor(X_te)
            tp = np.clip(np.expm1(model(te_x).numpy()), 0, None)
            for i, m in enumerate(test_meta):
                row_idx = m[2]
                n_r = len(row_idx)
                tpred_rows[row_idx] += tp[i, :n_r] / N_SPLITS

        print(f"    Fold {fold_idx}: {best_mae:.4f}", flush=True)
        pickle.dump({'oof': oof_rows.copy(), 'test': tpred_rows.copy() * N_SPLITS / (fold_idx + 1)},
                    open(f'{RESULT_DIR}/{name}_fold{fold_idx}.pkl', 'wb'))

    mae = mean_absolute_error(y, oof_rows)
    print(f"  [{name}] OOF: {mae:.4f} ({time.time()-t1:.0f}s)", flush=True)
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tpred_rows}).to_csv(f'./submission_{name}.csv', index=False)
    pickle.dump({'oof': oof_rows, 'test': tpred_rows, 'mae': mae},
                open(f'{RESULT_DIR}/{name}_final.pkl', 'wb'))
    return oof_rows, tpred_rows, mae

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 4개 모델 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] 4개 모델 순차 학습...", flush=True)
new_models = {}

# 1. mlp_deep seed=123
oof, tpred, mae = train_row('mlp_deep_s2', MLPDeep, seed=123)
new_models['mlp_deep_s2'] = {'oof': oof, 'test': tpred, 'mae': mae}

# 2. mlp_deep seed=2024
oof, tpred, mae = train_row('mlp_deep_s3', MLPDeep, seed=2024)
new_models['mlp_deep_s3'] = {'oof': oof, 'test': tpred, 'mae': mae}

# 3. mlp_deep_gelu seed=42
oof, tpred, mae = train_row('mlp_deep_gelu', MLPDeepGELU, seed=42)
new_models['mlp_deep_gelu'] = {'oof': oof, 'test': tpred, 'mae': mae}

# 4. Transformer Encoder
oof, tpred, mae = train_seq('transformer_enc', TransformerEncoder, seed=42)
new_models['transformer_enc'] = {'oof': oof, 'test': tpred, 'mae': mae}

pickle.dump(new_models, open(f'{RESULT_DIR}/mlp_army.pkl', 'wb'))
print(f"\n  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 37-Model Mega-Stacking
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] 37-model Mega-Stacking...", flush=True)

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

selected_oofs = {}
selected_tests = {}

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
domain = pickle.load(open(f'{RESULT_DIR}/domain_phase2.pkl', 'rb'))
mlp_aug = pickle.load(open(f'{RESULT_DIR}/mlp_aug_final.pkl', 'rb'))
offset_p3 = pickle.load(open(f'{RESULT_DIR}/offset_phase3.pkl', 'rb'))
na = pickle.load(open(f'{RESULT_DIR}/neural_army.pkl', 'rb'))

selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']
for n in domain['oofs']:
    selected_oofs[f'domain_{n}'] = domain['oofs'][n]
    selected_tests[f'domain_{n}'] = domain['tests'][n]
selected_oofs['mlp_aug'] = mlp_aug['mlp_aug_oof']
selected_tests['mlp_aug'] = mlp_aug['mlp_aug_test']
for n, data in offset_p3.items():
    selected_oofs[f'offset_{n}'] = data['oof']
    selected_tests[f'offset_{n}'] = data['test']
for n, data in na.items():
    selected_oofs[f'na_{n}'] = data['oof']
    selected_tests[f'na_{n}'] = data['test']

# 신규 4개 추가
for n, data in new_models.items():
    selected_oofs[f'new_{n}'] = data['oof']
    selected_tests[f'new_{n}'] = data['test']

print(f"  총 {len(selected_oofs)}개 모델")

stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])

meta_oofs_all = {}
meta_tests_all = {}

# LGB meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(objective='mae', n_estimators=500, learning_rate=0.05,
                           num_leaves=15, max_depth=4, min_child_samples=100,
                           random_state=42, verbose=-1, n_jobs=-1)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['lgb'] = oof; meta_tests_all['lgb'] = tpred
print(f"  LGB meta: {mean_absolute_error(y, oof):.4f}")

# XGB meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=500, learning_rate=0.05,
                          max_depth=4, min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', random_state=42, verbosity=0, n_jobs=-1, early_stopping_rounds=50)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])], verbose=0)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['xgb'] = oof; meta_tests_all['xgb'] = tpred
print(f"  XGB meta: {mean_absolute_error(y, oof):.4f}")

# CatBoost meta
oof = np.zeros(len(y))
tpred = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = CatBoostRegressor(loss_function='MAE', iterations=500, learning_rate=0.05,
                           depth=4, l2_leaf_reg=5, random_seed=42, verbose=0, early_stopping_rounds=50)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=(stack_train[val_idx], y_log[val_idx]), verbose=0)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['cb'] = oof; meta_tests_all['cb'] = tpred
print(f"  CatBoost meta: {mean_absolute_error(y, oof):.4f}")

meta_avg_oof = (meta_oofs_all['lgb'] + meta_oofs_all['xgb'] + meta_oofs_all['cb']) / 3
meta_avg_test = (meta_tests_all['lgb'] + meta_tests_all['xgb'] + meta_tests_all['cb']) / 3
print(f"  3-meta avg: {mean_absolute_error(y, meta_avg_oof):.4f}")

# CSV
for name, tp in meta_tests_all.items():
    pd.DataFrame({'ID': test_fe['ID'], TARGET: tp}).to_csv(
        f'./submission_mega37_{name}.csv', index=False)
pd.DataFrame({'ID': test_fe['ID'], TARGET: meta_avg_test}).to_csv(
    './submission_mega37_avg.csv', index=False)

# 블렌딩
prev = pd.read_csv('./submission_mega33_avg.csv')[TARGET].values
for ratio in [0.3, 0.5, 0.7]:
    blend = ratio * meta_avg_test + (1 - ratio) * prev
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_mega37_avg_prev_{int(ratio*100)}.csv', index=False)

pickle.dump({
    'meta_oofs': meta_oofs_all, 'meta_tests': meta_tests_all,
    'meta_avg_oof': meta_avg_oof, 'meta_avg_test': meta_avg_test,
}, open(f'{RESULT_DIR}/mega37_final.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"MLP Army 완료!")
for name, data in new_models.items():
    print(f"  new_{name}: OOF={data['mae']:.4f}")
print(f"  mega37 LGB: {mean_absolute_error(y, meta_oofs_all['lgb']):.4f}")
print(f"  mega37 XGB: {mean_absolute_error(y, meta_oofs_all['xgb']):.4f}")
print(f"  mega37 CB:  {mean_absolute_error(y, meta_oofs_all['cb']):.4f}")
print(f"  mega37 avg: {mean_absolute_error(y, meta_avg_oof):.4f}")
print(f"  (이전 mega33 avg: 8.3989)")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
