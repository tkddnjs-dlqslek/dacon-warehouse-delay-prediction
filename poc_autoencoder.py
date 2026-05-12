"""
POC B: Auto-encoder reconstruction error as a feature.

Train AE on v23 149 features (train+test combined for better representation).
Per-row reconstruction error (L2 norm over 149 dims) = "anomaly" signal.
Also extract bottleneck embedding (16 dims) as features.

Add reconstruction error + 16 embeddings = 17 new features to v23.
Retrain LGB_Huber, check corr.
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import torch
import torch.nn as nn

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "poc_autoencoder")
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)


class AE(nn.Module):
    def __init__(self, d_in=149, d_bottle=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, d_bottle),
        )
        self.dec = nn.Sequential(
            nn.Linear(d_bottle, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, d_in),
        )

    def forward(self, x):
        z = self.enc(x)
        xh = self.dec(z)
        return z, xh


def main():
    print("=" * 60, flush=True)
    print("POC B: Auto-encoder reconstruction error + embeddings", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    y_log = np.log1p(np.clip(y, 0, None))

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)

    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # Stack features
    X_train = train_fe[feat_cols].values.astype(np.float32)
    X_test = test_fe[feat_cols].values.astype(np.float32)

    # Fill NaN with column median, standardize
    print(f"NaN count train: {np.isnan(X_train).sum()}, test: {np.isnan(X_test).sum()}", flush=True)
    med = np.nanmedian(X_train, axis=0)
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_test = np.where(np.isnan(X_test), med, X_test)
    scaler = StandardScaler().fit(X_train)
    Xt_std = scaler.transform(X_train).astype(np.float32)
    Xe_std = scaler.transform(X_test).astype(np.float32)

    # Train AE on train+test combined (unsupervised, no leak)
    X_all = np.vstack([Xt_std, Xe_std])
    print(f"AE training set: {X_all.shape}", flush=True)

    torch.manual_seed(42)
    ae = AE(d_in=X_all.shape[1], d_bottle=16).to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    Xa = torch.from_numpy(X_all).to(DEVICE)
    batch_size = 4096
    n_epochs = 20
    print(f"Training AE: {n_epochs} epochs, batch={batch_size}", flush=True)
    for epoch in range(n_epochs):
        ae.train()
        perm = torch.randperm(len(Xa), device=DEVICE)
        total_loss = 0
        n_batches = 0
        for i in range(0, len(Xa), batch_size):
            idx = perm[i:i+batch_size]
            xb = Xa[idx]
            z, xh = ae(xb)
            loss = loss_fn(xh, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1}: loss {total_loss/n_batches:.6f}", flush=True)

    # Compute reconstruction error + embeddings
    ae.eval()
    with torch.no_grad():
        z_train, xh_train = ae(torch.from_numpy(Xt_std).to(DEVICE))
        z_test, xh_test = ae(torch.from_numpy(Xe_std).to(DEVICE))
        rec_err_train = ((xh_train - torch.from_numpy(Xt_std).to(DEVICE)) ** 2).mean(dim=1).cpu().numpy()
        rec_err_test = ((xh_test - torch.from_numpy(Xe_std).to(DEVICE)) ** 2).mean(dim=1).cpu().numpy()
        z_train = z_train.cpu().numpy()
        z_test = z_test.cpu().numpy()

    print(f"rec_err train stats: mean={rec_err_train.mean():.4f}, std={rec_err_train.std():.4f}", flush=True)

    # Build new feature matrix: v23 149 + 16 embeddings + 1 rec_err
    X_train_ext = np.hstack([X_train, z_train, rec_err_train.reshape(-1, 1)])
    X_test_ext = np.hstack([X_test, z_test, rec_err_test.reshape(-1, 1)])
    print(f"Extended feature matrix: {X_train_ext.shape}", flush=True)

    # Compare with existing
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"\nOld v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)

    # Train LGB_Huber
    print("\nTraining LGB_Huber on 149 + 17 AE features (5-fold)...", flush=True)
    new_oof = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        model = lgb.LGBMRegressor(
            objective="huber", n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X_train_ext[tr_mask], y_log[tr_mask],
            eval_set=[(X_train_ext[val_mask], y_log[val_mask])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        new_oof[val_mask] = np.expm1(model.predict(X_train_ext[val_mask]))
        test_preds.append(np.expm1(model.predict(X_test_ext)))
        fold_mae = np.mean(np.abs(new_oof[val_mask] - y[val_mask]))
        print(f"  fold {f}: best_iter={model.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    new_test = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))
    corr = float(np.corrcoef(y - old_oof, y - new_oof)[0, 1])

    print(f"\n=== POC B Comparison ===", flush=True)
    print(f"Old OOF MAE:   {old_mae:.5f}", flush=True)
    print(f"New OOF MAE:   {new_mae:.5f}", flush=True)
    print(f"Delta:         {new_mae - old_mae:+.5f}", flush=True)
    print(f"residual_corr: {corr:.5f}", flush=True)

    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95
    print("\nKill gates:", flush=True)
    print(f"  MAE hurt > 0.05: {hurt}", flush=True)
    print(f"  residual_corr ≥ 0.95: {too_similar}", flush=True)

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test)
    np.save(os.path.join(OUT, "rec_err_train.npy"), rec_err_train)
    np.save(os.path.join(OUT, "rec_err_test.npy"), rec_err_test)
    np.save(os.path.join(OUT, "z_train.npy"), z_train)
    np.save(os.path.join(OUT, "z_test.npy"), z_test)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae,
        delta=new_mae - old_mae, corr=corr,
        rec_err_train_mean=float(rec_err_train.mean()),
        rec_err_train_std=float(rec_err_train.std()),
        gates=dict(hurt=bool(hurt), too_similar=bool(too_similar)),
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if hurt or too_similar:
        print("\nVERDICT: NO_GO", flush=True)
    else:
        print("\nVERDICT: PROCEED", flush=True)


if __name__ == "__main__":
    main()
