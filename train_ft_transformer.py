"""
FT-Transformer (간소화 버전): Tabular Transformer
  - 각 피처를 token으로 embedding → Transformer encoder → prediction
  - GBDT/MLP와 다른 학습 방식
  - 5-fold GroupKFold, log target
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
print("=== FT-Transformer (간소화) ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']

# v23 동일 피처 (코드 재사용)
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
groups = train_fe['layout_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(train_fe, y, groups=groups))

X_all = np.nan_to_num(train_fe[feature_cols].values.astype(np.float32), 0)
X_test_all = np.nan_to_num(test_fe[feature_cols].values.astype(np.float32), 0)

n_features = X_all.shape[1]
print(f"  피처: {n_features}개, Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FT-Transformer (간소화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class FeatureTokenizer(nn.Module):
    """각 numerical feature를 d_model 차원의 token으로 변환"""
    def __init__(self, n_features, d_model):
        super().__init__()
        # 각 피처마다 학습 가능한 weight + bias
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def forward(self, x):
        # x: (batch, n_features)
        # token = x.unsqueeze(-1) * weight + bias
        x = x.unsqueeze(-1) * self.weight + self.bias  # (batch, n_features, d_model)
        # CLS token 추가
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # (batch, n_features+1, d_model)
        return x

class FTTransformer(nn.Module):
    def __init__(self, n_features, d_model=64, n_heads=4, n_layers=3, dropout=0.2):
        super().__init__()
        self.tokenizer = FeatureTokenizer(n_features, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_model*2, dropout=dropout,
            activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (batch, n_features)
        tokens = self.tokenizer(x)  # (batch, n_features+1, d_model)
        encoded = self.transformer(tokens)
        cls_out = self.norm(encoded[:, 0])  # CLS token only
        return self.head(cls_out).squeeze(-1)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 5-Fold Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] FT-Transformer 5-Fold Training...", flush=True)

EPOCHS = 30
BATCH_SIZE = 1024
LR = 0.0005

ftt_oof = np.zeros(len(X_all))
ftt_test = np.zeros(len(X_test_all))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    t1 = time.time()
    print(f"\n  Fold {fold_idx}...", flush=True)

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_all[tr_idx])
    X_val = scaler.transform(X_all[val_idx])
    X_te = scaler.transform(X_test_all)

    tr_dataset = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(y_log[tr_idx]))
    tr_loader = DataLoader(tr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_tensor = torch.FloatTensor(X_val)
    te_tensor = torch.FloatTensor(X_te)

    model = FTTransformer(n_features=n_features, d_model=64, n_heads=4, n_layers=3, dropout=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    loss_fn = nn.L1Loss()

    best_val_mae = 999
    best_state = None
    no_improve = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        n_samples = 0
        for xb, yb in tr_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_samples += len(xb)
        train_loss /= n_samples

        model.eval()
        with torch.no_grad():
            # 큰 데이터 → 배치로 나눠서 예측
            val_preds = []
            for i in range(0, len(val_tensor), 4096):
                val_preds.append(model(val_tensor[i:i+4096]).numpy())
            val_pred_log = np.concatenate(val_preds)
            val_pred = np.clip(np.expm1(val_pred_log), 0, None)
            val_mae = mean_absolute_error(y[val_idx], val_pred)

        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: loss={train_loss:.4f}, val_MAE={val_mae:.4f}, best={best_val_mae:.4f}", flush=True)

        if no_improve >= 7:
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_preds = []
        for i in range(0, len(val_tensor), 4096):
            val_preds.append(model(val_tensor[i:i+4096]).numpy())
        ftt_oof[val_idx] = np.clip(np.expm1(np.concatenate(val_preds)), 0, None)

        te_preds = []
        for i in range(0, len(te_tensor), 4096):
            te_preds.append(model(te_tensor[i:i+4096]).numpy())
        ftt_test += np.clip(np.expm1(np.concatenate(te_preds)), 0, None) / N_SPLITS

    fold_mae = mean_absolute_error(y[val_idx], ftt_oof[val_idx])
    print(f"  Fold {fold_idx} MAE: {fold_mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    pickle.dump({'ftt_oof': ftt_oof.copy(), 'ftt_test': ftt_test.copy() * N_SPLITS / (fold_idx + 1)},
                open(f'{RESULT_DIR}/ftt_fold{fold_idx}.pkl', 'wb'))

    partial_test = ftt_test * N_SPLITS / (fold_idx + 1) if fold_idx < N_SPLITS - 1 else ftt_test
    pd.DataFrame({'ID': test_fe['ID'], TARGET: partial_test}).to_csv(
        f'./submission_ftt_fold{fold_idx}.csv', index=False)

ftt_oof_mae = mean_absolute_error(y, ftt_oof)
print(f"\n  FT-Transformer OOF MAE: {ftt_oof_mae:.4f}", flush=True)

pd.DataFrame({'ID': test_fe['ID'], TARGET: ftt_test}).to_csv('./submission_ftt.csv', index=False)
pickle.dump({'ftt_oof': ftt_oof, 'ftt_test': ftt_test, 'ftt_oof_mae': ftt_oof_mae},
            open(f'{RESULT_DIR}/ftt_final.pkl', 'wb'))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: Mega-stacking에 FTT 추가
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] Mega-Stacking + FTT...", flush=True)

import lightgbm as lgb

# 기존 20개 + adversarial reweighted (있으면) + FTT
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
selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']

# Adversarial reweighted (chain 후 있으면)
try:
    adv_p2 = pickle.load(open(f'{RESULT_DIR}/adversarial_phase2.pkl', 'rb'))
    for alpha_name, data in adv_p2.items():
        for mname, oof in data['oofs'].items():
            selected_oofs[f'rw{alpha_name}_{mname}'] = oof
            selected_tests[f'rw{alpha_name}_{mname}'] = data['tests'][mname]
    print(f"  Adversarial 모델 추가됨")
except:
    print(f"  Adversarial pickle 없음, 스킵")

# FTT 추가
selected_oofs['ftt'] = ftt_oof
selected_tests['ftt'] = ftt_test

print(f"  총 {len(selected_oofs)}개 모델")

stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])

# LGB meta
mega_oof = np.zeros(len(y))
mega_test = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(
        objective='mae', n_estimators=500, learning_rate=0.05,
        num_leaves=15, max_depth=4, min_child_samples=100,
        random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    mega_oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    mega_test += np.expm1(m.predict(stack_test)) / N_SPLITS
mega_oof = np.clip(mega_oof, 0, None)
mega_test = np.clip(mega_test, 0, None)
print(f"  Mega+FTT OOF: {mean_absolute_error(y, mega_oof):.4f}")

pd.DataFrame({'ID': test_fe['ID'], TARGET: mega_test}).to_csv(
    './submission_megastack_ftt.csv', index=False)

# 블렌딩
prev_best = pd.read_csv('./submission_megastack_lgb_15_4.csv')[TARGET].values
for ratio in [0.3, 0.5, 0.7]:
    blend = ratio * mega_test + (1 - ratio) * prev_best
    pd.DataFrame({'ID': test_fe['ID'], TARGET: blend}).to_csv(
        f'./submission_megastack_ftt_prev_{int(ratio*100)}.csv', index=False)

print(f"\n{'='*60}")
print(f"FT-Transformer 완료!")
print(f"  FTT OOF: {ftt_oof_mae:.4f}")
print(f"  Mega+FTT OOF: {mean_absolute_error(y, mega_oof):.4f}")
print(f"  submission_ftt*.csv, submission_megastack_ftt*.csv")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
