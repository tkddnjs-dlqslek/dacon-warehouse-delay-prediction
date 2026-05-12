"""
Approach 4: Sequence anomaly detection.

Build scenario trajectory vectors (25 ts × 6 key columns = 150-dim per scenario).
Fit Isolation Forest + LOF on train+test combined (unsupervised).
Anomaly scores = new features.
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
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "approach_4_anom")
os.makedirs(OUT, exist_ok=True)


def build_trajectory_matrix(df, cols):
    """Return 10000 (or 2000) × (25*len(cols)) trajectory matrix, sorted by (layout_id, scenario_id)."""
    df = df.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
    traj_list = []
    keys = []
    for (lid, sid), grp in df.groupby(["layout_id", "scenario_id"], sort=False):
        if len(grp) == 25:
            traj = np.concatenate([grp[c].fillna(0).values for c in cols])
            traj_list.append(traj)
            keys.append((lid, sid))
    return np.array(traj_list), keys


def main():
    print("=" * 60, flush=True)
    print("Approach 4: Sequence anomaly detection", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    test = (
        pd.read_csv(os.path.join(ROOT, "test.csv"))
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

    traj_cols = [
        "order_inflow_15m", "congestion_score", "robot_utilization",
        "pack_utilization", "fault_count_15m", "blocked_path_15m",
    ]
    # Ensure all exist
    traj_cols = [c for c in traj_cols if c in train.columns and c in test.columns]
    print(f"Trajectory cols: {traj_cols}", flush=True)

    print("Building train/test trajectory matrices...", flush=True)
    train_traj, train_keys = build_trajectory_matrix(train, traj_cols)
    test_traj, test_keys = build_trajectory_matrix(test, traj_cols)
    print(f"train_traj: {train_traj.shape}, test_traj: {test_traj.shape}", flush=True)

    # Standardize (fit on train + test combined to match distribution)
    all_traj = np.vstack([train_traj, test_traj])
    scaler = StandardScaler().fit(all_traj)
    train_std = scaler.transform(train_traj)
    test_std = scaler.transform(test_traj)

    # ─── Isolation Forest ───
    print("\nTraining Isolation Forest...", flush=True)
    iso = IsolationForest(
        n_estimators=200, contamination=0.1, random_state=42, n_jobs=-1,
    )
    iso.fit(np.vstack([train_std, test_std]))
    train_iso_score = -iso.decision_function(train_std)  # higher = more anomalous
    test_iso_score = -iso.decision_function(test_std)
    print(f"  train iso score: mean={train_iso_score.mean():.3f}, std={train_iso_score.std():.3f}", flush=True)

    # ─── LOF (unfortunately sklearn LOF doesn't have .predict on new data; use novelty=True) ───
    print("\nTraining LOF (novelty=True)...", flush=True)
    # Subsample for LOF fit speed
    n_fit = 5000
    np.random.seed(42)
    fit_idx = np.random.choice(len(train_std), size=min(n_fit, len(train_std)), replace=False)
    lof = LocalOutlierFactor(n_neighbors=30, contamination=0.1, novelty=True, n_jobs=-1)
    lof.fit(train_std[fit_idx])
    train_lof_score = -lof.decision_function(train_std)
    test_lof_score = -lof.decision_function(test_std)
    print(f"  train lof score: mean={train_lof_score.mean():.3f}, std={train_lof_score.std():.3f}", flush=True)

    # ─── Map scores to rows (each scenario has 25 rows) ───
    train_score_map = dict(zip(train_keys, zip(train_iso_score, train_lof_score)))
    test_score_map = dict(zip(test_keys, zip(test_iso_score, test_lof_score)))

    train_iso_row = np.array([train_score_map[(lid, sid)][0]
                               for lid, sid in zip(train["layout_id"], train["scenario_id"])])
    train_lof_row = np.array([train_score_map[(lid, sid)][1]
                               for lid, sid in zip(train["layout_id"], train["scenario_id"])])
    test_iso_row = np.array([test_score_map[(lid, sid)][0]
                              for lid, sid in zip(test["layout_id"], test["scenario_id"])])
    test_lof_row = np.array([test_score_map[(lid, sid)][1]
                              for lid, sid in zip(test["layout_id"], test["scenario_id"])])

    # corr with y and mega33 residual
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    resid = y - mega_oof

    print(f"\ncorr(iso_score, y): {np.corrcoef(train_iso_row, y)[0,1]:+.4f}", flush=True)
    print(f"corr(iso_score, resid): {np.corrcoef(train_iso_row, resid)[0,1]:+.4f}", flush=True)
    print(f"corr(lof_score, y): {np.corrcoef(train_lof_row, y)[0,1]:+.4f}", flush=True)
    print(f"corr(lof_score, resid): {np.corrcoef(train_lof_row, resid)[0,1]:+.4f}", flush=True)

    # Stack with v23
    X = np.hstack([train_fe[feat_cols].values,
                    train_iso_row.reshape(-1, 1),
                    train_lof_row.reshape(-1, 1)])
    X_test = np.hstack([test_fe[feat_cols].values,
                         test_iso_row.reshape(-1, 1),
                         test_lof_row.reshape(-1, 1)])
    print(f"\nCombined X: {X.shape}", flush=True)

    # Train LGB_Huber
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"Reference v23 LGB_Huber OOF: {old_mae:.5f}", flush=True)

    print("\n=== Training LGB_Huber on v23 + 2 anomaly features ===", flush=True)
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
        model.fit(X[tr_mask], y_log[tr_mask],
                  eval_set=[(X[val_mask], y_log[val_mask])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        new_oof[val_mask] = np.expm1(model.predict(X[val_mask]))
        test_preds.append(np.expm1(model.predict(X_test)))
        fold_mae = np.mean(np.abs(new_oof[val_mask] - y[val_mask]))
        print(f"  fold {f}: best_iter={model.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    new_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))
    corr = float(np.corrcoef(y - old_oof, y - new_oof)[0, 1])
    mega_mae = float(mean_absolute_error(y, mega_oof))
    corr_mega = float(np.corrcoef(y - mega_oof, y - new_oof)[0, 1])

    print(f"\n=== Results ===", flush=True)
    print(f"Old OOF: {old_mae:.5f}", flush=True)
    print(f"New OOF: {new_mae:.5f}  delta={new_mae-old_mae:+.5f}", flush=True)
    print(f"corr (v23 old): {corr:.5f}", flush=True)
    print(f"corr (mega33):  {corr_mega:.5f}", flush=True)

    best_mae = mega_mae; best_w = 0
    for w in np.linspace(0, 0.3, 31):
        pred = (1 - w) * mega_oof + w * new_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m; best_w = w
    print(f"Blend: w={best_w:.3f}, MAE={best_mae:.5f}, delta={best_mae-mega_mae:+.5f}", flush=True)

    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test_pred)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae, delta=new_mae-old_mae,
        corr=corr, corr_mega=corr_mega,
        blend_w=best_w, blend_mae=best_mae, blend_delta=best_mae-mega_mae,
        iso_corr_y=float(np.corrcoef(train_iso_row, y)[0,1]),
        lof_corr_y=float(np.corrcoef(train_lof_row, y)[0,1]),
        gates=dict(hurt=hurt, too_similar=too_similar),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVERDICT: {'NO_GO' if (hurt or too_similar) else 'PROCEED'}", flush=True)


if __name__ == "__main__":
    main()
