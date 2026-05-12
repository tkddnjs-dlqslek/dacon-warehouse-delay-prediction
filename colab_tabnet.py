# ================================================================
# colab_tabnet.py — TabNet tabular model for Dacon warehouse delay
# Google Colab (T4 GPU) 전용
#
# 사용법:
#   1. colab_data.zip 을 Colab에 업로드
#   2. 아래 순서대로 실행
# ================================================================

# ── [셀 1] 패키지 설치 ──────────────────────────────────────────
# !pip install -q pytorch-tabnet

# ── [셀 2] 전체 실행 코드 ────────────────────────────────────────
import numpy as np
import pandas as pd
import os, zipfile, time, warnings
warnings.filterwarnings('ignore')

from pytorch_tabnet.tab_model import TabNetRegressor
import torch

# ── 데이터 로드 ──────────────────────────────────────────────────
BASE = '/content'

if not os.path.exists(f'{BASE}/X_train.npy'):
    with zipfile.ZipFile(f'{BASE}/colab_data.zip') as z:
        z.extractall(BASE)
    print("Extracted colab_data.zip")

X_tr  = np.load(f'{BASE}/X_train.npy').astype(np.float32)
y_ls  = np.load(f'{BASE}/y_train.npy').astype(np.float32)
X_te  = np.load(f'{BASE}/X_test.npy').astype(np.float32)
fold_ids   = np.load(f'{BASE}/fold_ids.npy')
oracle_oof = np.load(f'{BASE}/oracle_oof.npy')
oracle_t   = np.load(f'{BASE}/oracle_test.npy')
id2        = np.load(f'{BASE}/id2.npy')
te_rid     = np.load(f'{BASE}/te_rid_to_ls.npy')
unseen     = np.load(f'{BASE}/unseen_mask.npy')
y_rid      = np.load(f'{BASE}/y_true_rid.npy')

oracle_mae    = float(np.mean(np.abs(y_rid - oracle_oof)))
oracle_unseen = float(oracle_t[unseen].mean())
print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

# ── NaN 처리 + 표준화 ────────────────────────────────────────────
col_mean = np.nanmean(X_tr, axis=0)
col_std  = np.nanstd(X_tr, axis=0)
col_std  = np.where(col_std < 1e-8, 1.0, col_std)

def preprocess(X, mean, std):
    X = X.copy()
    nan_mask = np.isnan(X)
    X[nan_mask] = np.take(mean, np.where(nan_mask)[1])
    return (X - mean) / std

X_tr_n = preprocess(X_tr, col_mean, col_std)
X_te_n = preprocess(X_te, col_mean, col_std)

# log1p target
y_log = np.log1p(y_ls).astype(np.float64)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")

# ── TabNet 학습 (5-fold) ─────────────────────────────────────────
N_FOLDS  = 5
oof_preds  = np.zeros(len(y_ls))
test_preds = np.zeros(len(X_te))

# TabNet 하이퍼파라미터
TABNET_PARAMS = dict(
    n_d=32, n_a=32,
    n_steps=5,
    gamma=1.3,
    n_independent=2, n_shared=2,
    lambda_sparse=1e-4,
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-3, weight_decay=1e-5),
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    scheduler_params=dict(step_size=10, gamma=0.9),
    mask_type='entmax',
    verbose=0,
    device_name=DEVICE,
)

print(f"\n=== Training TabNet ({N_FOLDS}-fold) ===")

for fold in range(N_FOLDS):
    t0 = time.time()
    val_mask = fold_ids == fold
    tr_mask  = ~val_mask

    model = TabNetRegressor(**TABNET_PARAMS)
    model.fit(
        X_tr_n[tr_mask], y_log[tr_mask].reshape(-1, 1),
        eval_set=[(X_tr_n[val_mask], y_log[val_mask].reshape(-1, 1))],
        eval_metric=['mae'],
        max_epochs=200,
        patience=20,
        batch_size=4096,
        virtual_batch_size=512,
        num_workers=2,
    )

    val_pred_log = model.predict(X_tr_n[val_mask]).flatten()
    oof_preds[val_mask] = np.clip(np.expm1(val_pred_log), 0, None)
    test_preds += np.clip(np.expm1(model.predict(X_te_n).flatten()), 0, None) / N_FOLDS

    fold_mae = float(np.mean(np.abs(y_ls[val_mask] - oof_preds[val_mask])))
    print(f"  fold {fold}: MAE={fold_mae:.4f}  best_ep={model.best_epoch} ({time.time()-t0:.0f}s)")

oof_preds  = np.clip(oof_preds, 0, None)
oof_rid    = oof_preds[id2]
test_rid   = test_preds[te_rid]

oof_mae  = float(np.mean(np.abs(y_rid - oof_rid)))
corr     = float(np.corrcoef(oracle_oof, oof_rid)[0, 1])
print(f"\nTabNet OOF (rid): {oof_mae:.4f}  corr={corr:.4f}")
print(f"test_unseen: {test_rid[unseen].mean():.3f}  test_seen: {test_rid[~unseen].mean():.3f}")

# ── Blend 분석 ──────────────────────────────────────────────────
print(f"\n=== Blend with oracle_NEW ===")
best_w, best_bl = 0, oracle_mae
for w in np.arange(0.01, 0.51, 0.01):
    bl = np.clip((1-w)*oracle_oof + w*oof_rid, 0, None)
    m  = float(np.mean(np.abs(y_rid - bl)))
    if m < best_bl:
        best_bl, best_w = m, w

print(f"Best blend: w={best_w:.2f}  OOF={best_bl:.4f}  gain={oracle_mae-best_bl:+.4f}")
for w in [0.02, 0.05, 0.10, 0.15, 0.20]:
    bl   = np.clip((1-w)*oracle_oof + w*oof_rid, 0, None)
    bl_t = np.clip((1-w)*oracle_t   + w*test_rid, 0, None)
    m    = float(np.mean(np.abs(y_rid - bl)))
    print(f"  w={w:.2f}: OOF={m:.4f} ({m-oracle_mae:+.4f})  unseen={bl_t[unseen].mean():.3f} ({bl_t[unseen].mean()-oracle_unseen:+.3f})")

# ── 저장 ────────────────────────────────────────────────────────
np.save(f'{BASE}/tabnet_oof.npy',  oof_rid.astype(np.float32))
np.save(f'{BASE}/tabnet_test.npy', test_rid.astype(np.float32))
print(f"\nSaved: tabnet_oof.npy, tabnet_test.npy")

if best_w > 0:
    bl_t = np.clip((1-best_w)*oracle_t + best_w*test_rid, 0, None)
    sample = pd.read_csv(f'{BASE}/sample_submission.csv')
    fname  = f'/content/COLAB_tabnet_w{int(best_w*100):02d}_OOF{best_bl:.4f}.csv'
    sample['avg_delay_minutes_next_30m'] = bl_t
    sample.to_csv(fname, index=False)
    print(f"Submission saved: {fname}  (gain={oracle_mae-best_bl:+.4f})")
else:
    print("No blend improvement. Do not submit.")
