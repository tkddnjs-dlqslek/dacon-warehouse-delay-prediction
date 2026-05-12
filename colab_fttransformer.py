# ================================================================
# colab_fttransformer.py — FT-Transformer for Dacon warehouse delay
# Google Colab (T4 GPU) 전용
#
# 사용법:
#   1. colab_data.zip 을 Colab에 업로드
#   2. 순서대로 실행
# ================================================================

# ── [셀 1] 패키지 설치 ──────────────────────────────────────────
# !pip install -q rtdl

# ── [셀 2] 전체 실행 코드 ────────────────────────────────────────
import numpy as np
import pandas as pd
import os, zipfile, time, warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import rtdl

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

y_log = np.log1p(y_ls).astype(np.float32)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {DEVICE}")
n_features = X_tr.shape[1]

# ── FT-Transformer 설정 ──────────────────────────────────────────
N_FOLDS  = 5
EPOCHS   = 100
BS       = 2048
PATIENCE = 15
LR       = 1e-4

oof_preds  = np.zeros(len(y_ls))
test_preds = np.zeros(len(X_te))

print(f"\n=== Training FT-Transformer ({N_FOLDS}-fold) ===")

for fold in range(N_FOLDS):
    t0 = time.time()
    val_mask = fold_ids == fold
    tr_mask  = ~val_mask

    Xtr = torch.tensor(X_tr_n[tr_mask])
    ytr = torch.tensor(y_log[tr_mask])
    Xvl = torch.tensor(X_tr_n[val_mask])
    yvl = torch.tensor(y_log[val_mask])
    Xte = torch.tensor(X_te_n)

    train_dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True, num_workers=2)
    val_dl   = DataLoader(TensorDataset(Xvl, yvl), batch_size=BS*4, shuffle=False, num_workers=2)

    model = rtdl.FTTransformer.make_default(
        n_num_features=n_features,
        cat_cardinalities=None,
        last_layer_query_idx=[-1],
        d_out=1,
    ).to(DEVICE)

    opt   = model.make_default_optimizer()
    loss_fn = nn.HuberLoss(delta=1.0)

    best_val, best_ep, patience_cnt = 1e9, 0, 0
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb, None).squeeze(-1)
            loss_fn(pred, yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_log = []
        with torch.no_grad():
            for xb, _ in val_dl:
                val_log.append(model(xb.to(DEVICE), None).squeeze(-1).cpu().numpy())
        vp = np.clip(np.expm1(np.concatenate(val_log)), 0, None)
        val_mae = float(np.mean(np.abs(np.expm1(yvl.numpy()) - vp)))

        if val_mae < best_val:
            best_val, best_ep, patience_cnt = val_mae, ep, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vl_log = []
        for xb, _ in val_dl:
            vl_log.append(model(xb.to(DEVICE), None).squeeze(-1).cpu().numpy())
        oof_preds[val_mask] = np.clip(np.expm1(np.concatenate(vl_log)), 0, None)

        te_log = []
        te_dl  = DataLoader(TensorDataset(Xte), batch_size=BS*4, shuffle=False)
        for (xb,) in te_dl:
            te_log.append(model(xb.to(DEVICE), None).squeeze(-1).cpu().numpy())
        test_preds += np.clip(np.expm1(np.concatenate(te_log)), 0, None) / N_FOLDS

    print(f"  fold {fold}: best_val={best_val:.4f} ep={best_ep} ({time.time()-t0:.0f}s)")

oof_preds  = np.clip(oof_preds, 0, None)
oof_rid    = oof_preds[id2]
test_rid   = test_preds[te_rid]

oof_mae = float(np.mean(np.abs(y_rid - oof_rid)))
corr    = float(np.corrcoef(oracle_oof, oof_rid)[0, 1])
print(f"\nFT-Transformer OOF (rid): {oof_mae:.4f}  corr={corr:.4f}")
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
np.save(f'{BASE}/ftt_oof.npy',  oof_rid.astype(np.float32))
np.save(f'{BASE}/ftt_test.npy', test_rid.astype(np.float32))
print(f"\nSaved: ftt_oof.npy, ftt_test.npy")

if best_w > 0:
    bl_t = np.clip((1-best_w)*oracle_t + best_w*test_rid, 0, None)
    sample = pd.read_csv(f'{BASE}/sample_submission.csv')
    fname  = f'/content/COLAB_ftt_w{int(best_w*100):02d}_OOF{best_bl:.4f}.csv'
    sample['avg_delay_minutes_next_30m'] = bl_t
    sample.to_csv(fname, index=False)
    print(f"Submission saved: {fname}  (gain={oracle_mae-best_bl:+.4f})")
else:
    print("No blend improvement. Do not submit.")
