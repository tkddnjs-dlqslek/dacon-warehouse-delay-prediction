"""
1D-CNN: 시나리오 25 timestep을 시퀀스로 학습
  - Input: (batch, 25, n_features) → Conv1D → prediction per timestep
  - 5-fold GroupKFold, log1p target
  - fold별 pickle + csv 저장
  - GBDT/MLP와 블렌딩
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
from torch.utils.data import DataLoader, Dataset
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
print("=== 1D-CNN: Sequence Model ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

v23_phase1 = pickle.load(open(f'{RESULT_DIR}/v23_phase1.pkl', 'rb'))
v23_selected = v23_phase1['selected_features']

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: 피처 생성 + 시퀀스 구조 변환
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 피처 엔지니어링 + 시퀀스 변환...", flush=True)

# SC features 제외한 raw 피처만 사용 (SC는 시퀀스에서 자동 학습됨)
# CNN이 직접 시퀀스 패턴을 학습하므로 lag/lead/SC 불필요
# 대신 원본 90개 운영 피처를 그대로 사용

train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

train = train.sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
train['timeslot'] = train.groupby(['layout_id', 'scenario_id']).cumcount()
test = test.sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
test['timeslot'] = test.groupby(['layout_id', 'scenario_id']).cumcount()

y = train[TARGET].values
y_log = np.log1p(y)

# CNN용 피처: layout 정적 제거, ID/meta 제거
exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type',
           'aisle_width_avg', 'intersection_count', 'one_way_ratio',
           'pack_station_count', 'charger_count', 'layout_compactness',
           'zone_dispersion', 'robot_total', 'building_age_years',
           'floor_area_sqm', 'ceiling_height_m', 'fire_sprinkler_count',
           'emergency_exit_count', 'timeslot']
cnn_features = [c for c in train.columns if c not in exclude]
print(f"  CNN 피처: {len(cnn_features)}개")

# 시나리오별 시퀀스로 변환 (25 timesteps × n_features)
train_groups = train.groupby(['layout_id', 'scenario_id'])
test_groups = test.groupby(['layout_id', 'scenario_id'])

def make_sequences(df, groups, features, target_col=None):
    """시나리오별 (25, n_features) 시퀀스 생성"""
    X_seqs = []
    y_seqs = []
    meta = []  # (layout_id, scenario_id, row_indices)

    for (lid, sid), group_df in groups:
        vals = group_df[features].values.astype(np.float32)
        # 25 timestep이 아닌 경우 패딩
        if len(vals) < 25:
            pad = np.zeros((25 - len(vals), len(features)), dtype=np.float32)
            vals = np.vstack([vals, pad])
        elif len(vals) > 25:
            vals = vals[:25]

        X_seqs.append(vals)

        if target_col is not None:
            t = group_df[target_col].values
            if len(t) < 25:
                t = np.pad(t, (0, 25 - len(t)), constant_values=0)
            elif len(t) > 25:
                t = t[:25]
            y_seqs.append(t)

        meta.append((lid, sid, group_df.index.values))

    X_seqs = np.array(X_seqs)  # (n_scenarios, 25, n_features)
    y_seqs = np.array(y_seqs) if y_seqs else None
    return X_seqs, y_seqs, meta

print("  시퀀스 변환 중...", flush=True)
X_train_seq, y_train_seq, train_meta = make_sequences(train, train_groups, cnn_features, TARGET)
X_test_seq, _, test_meta = make_sequences(test, test_groups, cnn_features)

y_train_seq_log = np.log1p(y_train_seq)  # (n_scenarios, 25)

print(f"  Train: {X_train_seq.shape}, Test: {X_test_seq.shape}")

# NaN → 0
X_train_seq = np.nan_to_num(X_train_seq, 0)
X_test_seq = np.nan_to_num(X_test_seq, 0)

# layout_id 매핑 (GroupKFold용)
train_layout_ids = np.array([m[0] for m in train_meta])

groups_arr = train_layout_ids
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X_train_seq, y_train_seq, groups=groups_arr))

print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1D-CNN 모델
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class CNN1DModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        # Input: (batch, 25, n_features) → transpose → (batch, n_features, 25)
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        # Output per timestep: (batch, 64, 25) → (batch, 25, 64) → linear → (batch, 25, 1)
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, 25, n_features)
        x = x.transpose(1, 2)  # (batch, n_features, 25)
        x = self.conv(x)       # (batch, 64, 25)
        x = x.transpose(1, 2)  # (batch, 25, 64)
        x = self.head(x)       # (batch, 25, 1)
        return x.squeeze(-1)   # (batch, 25)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Custom Dataset
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
class SeqDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: 5-Fold Training
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] CNN 5-Fold Training...", flush=True)

EPOCHS = 60
BATCH_SIZE = 256
LR = 0.001

# 결과: row 단위로 펼쳐야 함 (시나리오 → 행)
cnn_oof_rows = np.zeros(len(train))
cnn_test_rows = np.zeros(len(test))

for fold_idx, (tr_idx, val_idx) in enumerate(folds):
    t1 = time.time()
    print(f"\n  Fold {fold_idx}...", flush=True)

    # Scaler (시나리오 레벨에서 피처별 정규화)
    # tr_idx는 시나리오 인덱스
    X_tr = X_train_seq[tr_idx]  # (n_tr, 25, feat)
    X_val = X_train_seq[val_idx]
    y_tr = y_train_seq_log[tr_idx]
    y_val = y_train_seq_log[val_idx]

    # 피처별 scaler (시나리오를 flatten해서 fit)
    n_feat = X_tr.shape[2]
    scaler = StandardScaler()
    X_tr_flat = X_tr.reshape(-1, n_feat)
    scaler.fit(X_tr_flat)
    X_tr = scaler.transform(X_tr.reshape(-1, n_feat)).reshape(X_tr.shape)
    X_val = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_te = scaler.transform(X_test_seq.reshape(-1, n_feat)).reshape(X_test_seq.shape)

    tr_loader = DataLoader(SeqDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    model = CNN1DModel(n_feat)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
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
            pred = model(xb)  # (batch, 25)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_samples += len(xb)
        train_loss /= n_samples

        model.eval()
        with torch.no_grad():
            val_pred_log = model(torch.FloatTensor(X_val)).numpy()  # (n_val, 25)
            val_pred = np.clip(np.expm1(val_pred_log), 0, None)
            # row 단위 MAE
            y_val_real = np.expm1(y_val)
            val_mae = mean_absolute_error(y_val_real.flatten(), val_pred.flatten())

        scheduler.step()

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1:2d}: loss={train_loss:.4f}, val_MAE={val_mae:.4f}, best={best_val_mae:.4f}", flush=True)

        if no_improve >= 12:
            print(f"    Early stop at epoch {epoch+1}", flush=True)
            break

    # Best model 예측
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = np.clip(np.expm1(model(torch.FloatTensor(X_val)).numpy()), 0, None)
        te_pred = np.clip(np.expm1(model(torch.FloatTensor(X_te)).numpy()), 0, None)

    # 시나리오 → row 매핑
    for i, sc_idx in enumerate(val_idx):
        row_indices = train_meta[sc_idx][2]
        n_rows = len(row_indices)
        cnn_oof_rows[row_indices] = val_pred[i, :n_rows]

    for i, meta in enumerate(test_meta):
        row_indices = meta[2]
        n_rows = len(row_indices)
        cnn_test_rows[row_indices] += te_pred[i, :n_rows] / N_SPLITS

    fold_mae = mean_absolute_error(y[np.concatenate([train_meta[si][2] for si in val_idx])],
                                    cnn_oof_rows[np.concatenate([train_meta[si][2] for si in val_idx])])
    print(f"  Fold {fold_idx} MAE: {fold_mae:.4f} ({time.time()-t1:.0f}s)", flush=True)

    pickle.dump({'cnn_oof': cnn_oof_rows.copy(), 'cnn_test': cnn_test_rows.copy()},
                open(f'{RESULT_DIR}/cnn_fold{fold_idx}.pkl', 'wb'))
    pd.DataFrame({'ID': test['ID'], TARGET: cnn_test_rows}).to_csv(
        f'./submission_cnn_fold{fold_idx}.csv', index=False)

cnn_oof_mae = mean_absolute_error(y, cnn_oof_rows)
print(f"\n  CNN OOF MAE: {cnn_oof_mae:.4f}", flush=True)

pd.DataFrame({'ID': test['ID'], TARGET: cnn_test_rows}).to_csv('./submission_cnn.csv', index=False)
pickle.dump({'cnn_oof': cnn_oof_rows, 'cnn_test': cnn_test_rows, 'cnn_oof_mae': cnn_oof_mae},
            open(f'{RESULT_DIR}/cnn_final.pkl', 'wb'))
print(f"  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] 블렌딩...", flush=True)

# 기존 모델 로드
s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
w = s42['weights']
names = list(s42['oofs'].keys())
v23_oof = np.clip(sum(wi*p for wi, p in zip(w, [s42['oofs'][n] for n in names])), 0, None)
v23_test = np.clip(sum(wi*p for wi, p in zip(w, [s42['tests'][n] for n in names])), 0, None)

mlp1 = pickle.load(open(f'{RESULT_DIR}/mlp_final.pkl', 'rb'))
mlp2 = pickle.load(open(f'{RESULT_DIR}/mlp2_final.pkl', 'rb'))
mlp_avg_test = (mlp1['mlp_test'] + mlp2['mlp2_test']) / 2
mlp_avg_oof = (mlp1['mlp_oof'] + mlp2['mlp2_oof']) / 2

v22_pre_test = pd.read_csv('submission_v22_pre.csv')[TARGET].values
v24_test = pd.read_csv('submission_v24.csv')[TARGET].values

from scipy.stats import pearsonr
r_cnn_v23, _ = pearsonr(cnn_test_rows, v23_test)
r_cnn_mlp, _ = pearsonr(cnn_test_rows, mlp_avg_test)
r_cnn_v22, _ = pearsonr(cnn_test_rows, v22_pre_test)
print(f"  CNN vs v23 상관: {r_cnn_v23:.4f}")
print(f"  CNN vs MLP_avg 상관: {r_cnn_mlp:.4f}")
print(f"  CNN vs v22_pre 상관: {r_cnn_v22:.4f}")

# v23 + CNN 블렌딩
print("\n  v23 + CNN 블렌딩 (OOF):")
best_mae = 999
best_r = 0
for ratio in np.arange(0, 0.51, 0.02):
    blended = (1 - ratio) * v23_oof + ratio * cnn_oof_rows
    mae = mean_absolute_error(y, blended)
    if mae < best_mae:
        best_mae = mae
        best_r = ratio
    if ratio in [0, 0.10, 0.20, 0.30, 0.50]:
        print(f"    CNN {ratio:.0%}: OOF MAE={mae:.4f}", flush=True)
print(f"  -> 최적: CNN {best_r:.0%} (OOF {best_mae:.4f})")

# 6-way: v23 + v22 + v24 + MLP1 + MLP2 + CNN
for r_cnn in [0.05, 0.08, 0.10, 0.12, 0.15]:
    scale = 1 - r_cnn
    # 기존 5-way 비율 유지하면서 CNN 추가
    blend6 = scale * (0.50 * v23_test + 0.15 * v22_pre_test + 0.10 * v24_test +
                       0.125 * mlp1['mlp_test'] + 0.125 * mlp2['mlp2_test']) + r_cnn * cnn_test_rows
    pd.DataFrame({'ID': test['ID'], TARGET: blend6}).to_csv(
        f'./submission_6way_cnn_{int(r_cnn*100)}.csv', index=False)

# best 3-way + MLP_avg + CNN
for r_cnn in [0.05, 0.10]:
    r_mlp = 0.10
    r_base = 1 - r_mlp - r_cnn
    blend = r_base * (0.65 * v23_test + 0.20 * v22_pre_test + 0.15 * v24_test) + r_mlp * mlp_avg_test + r_cnn * cnn_test_rows
    pd.DataFrame({'ID': test['ID'], TARGET: blend}).to_csv(
        f'./submission_best_mlp_cnn_{int(r_cnn*100)}.csv', index=False)

pickle.dump({
    'cnn_oof': cnn_oof_rows, 'cnn_test': cnn_test_rows,
    'r_cnn_v23': r_cnn_v23, 'r_cnn_mlp': r_cnn_mlp,
}, open(f'{RESULT_DIR}/cnn_blend.pkl', 'wb'))

print(f"\n{'='*60}")
print(f"1D-CNN 완료!")
print(f"  CNN OOF MAE: {cnn_oof_mae:.4f}")
print(f"  CNN vs v23: {r_cnn_v23:.4f}")
print(f"  CNN vs MLP: {r_cnn_mlp:.4f}")
print(f"  CNN vs v22: {r_cnn_v22:.4f}")
print(f"  v23+CNN 최적: CNN {best_r:.0%} (OOF {best_mae:.4f})")
print(f"  submission_cnn.csv")
print(f"  submission_6way_cnn_*.csv")
print(f"  submission_best_mlp_cnn_*.csv")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
