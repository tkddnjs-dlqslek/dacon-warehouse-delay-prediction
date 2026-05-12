"""
Approach 2: Target encoding with strict CV.

4 variants:
  A. Layout-level: mean y per layout_id
  B. Scenario-level: mean y per scenario_id
  C. Bucket-level (layout_type × ts_bucket × congestion_bin): 3-way mean y
  D. Shape-cluster: cluster scenarios by trajectory, mean y per cluster

For each fold:
  - Compute encoding from train-fold rows only
  - Apply to val-fold rows
  - For test: use full-train encoding
Smoothing: global mean shrinkage to handle small groups.

All 4 encodings → add as features → retrain v23 LGB_Huber → check corr/MAE.
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
from sklearn.cluster import KMeans
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "approach_2_te")
os.makedirs(OUT, exist_ok=True)


def smooth_target_encoding(train_df, test_df, group_cols, y, fold_idx, smoothing=10):
    """Fold-safe target encoding.
    train_df and test_df must have group_cols.
    y is only for training.
    Returns: (train_encoding array, test_encoding array)
    """
    global_mean = y.mean()
    train_enc = np.zeros(len(train_df), dtype=np.float64)
    n_folds = 5
    for f in range(n_folds):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        # Compute group mean and count on train fold
        tmp = train_df[tr_mask].copy()
        tmp["_y"] = y[tr_mask]
        g = tmp.groupby(group_cols)["_y"].agg(["mean", "count"]).reset_index()
        # Smooth: (count * mean + smoothing * global_mean) / (count + smoothing)
        g["_enc"] = (g["count"] * g["mean"] + smoothing * global_mean) / (g["count"] + smoothing)
        # Map to val rows
        enc_map = g.set_index(group_cols)["_enc"]
        val_keys = list(train_df[val_mask][group_cols].itertuples(index=False, name=None)) if len(group_cols) > 1 else train_df[val_mask][group_cols[0]].values
        val_enc = []
        for k in val_keys:
            if k in enc_map.index:
                val_enc.append(enc_map.loc[k])
            else:
                val_enc.append(global_mean)
        train_enc[val_mask] = np.array(val_enc)

    # Test: use full-train encoding
    tmp = train_df.copy()
    tmp["_y"] = y
    g = tmp.groupby(group_cols)["_y"].agg(["mean", "count"]).reset_index()
    g["_enc"] = (g["count"] * g["mean"] + smoothing * global_mean) / (g["count"] + smoothing)
    enc_map = g.set_index(group_cols)["_enc"]
    test_keys = list(test_df[group_cols].itertuples(index=False, name=None)) if len(group_cols) > 1 else test_df[group_cols[0]].values
    test_enc = []
    for k in test_keys:
        if k in enc_map.index:
            test_enc.append(enc_map.loc[k])
        else:
            test_enc.append(global_mean)
    return train_enc, np.array(test_enc)


def main():
    print("=" * 60, flush=True)
    print("Approach 2: Target Encoding (4 variants)", flush=True)
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

    # Setup layout_type, ts_idx, ts_bucket, congestion_bin
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))[["layout_id", "layout_type"]]
    train = train.merge(layout_info, on="layout_id", how="left")
    test = test.merge(layout_info, on="layout_id", how="left")
    train["ts_idx"] = np.tile(np.arange(25), len(train) // 25)
    test["ts_idx"] = np.tile(np.arange(25), len(test) // 25)
    train["ts_bucket"] = pd.cut(train["ts_idx"], bins=[-1, 7, 15, 24], labels=["early", "mid", "late"]).astype(str)
    test["ts_bucket"] = pd.cut(test["ts_idx"], bins=[-1, 7, 15, 24], labels=["early", "mid", "late"]).astype(str)

    # congestion_bin: use layoutwise quantile-based bin
    # For safety: compute quantiles on train-fold only for fold-based (we'll use global quantiles on train — small leak)
    cong_q25 = np.nanquantile(train["congestion_score"], 0.33)
    cong_q66 = np.nanquantile(train["congestion_score"], 0.67)
    def bin_cong(x):
        if np.isnan(x): return "mid"
        if x < cong_q25: return "low"
        if x > cong_q66: return "high"
        return "mid"
    train["cong_bin"] = train["congestion_score"].apply(bin_cong)
    test["cong_bin"] = test["congestion_score"].apply(bin_cong)

    # ─── Variant A: layout_id ───
    print("\n[A] Layout-level target encoding...", flush=True)
    ta_enc, te_enc = smooth_target_encoding(train, test, ["layout_id"], y, fold_idx, smoothing=20)
    print(f"  train enc stats: mean={ta_enc.mean():.3f}, std={ta_enc.std():.3f}", flush=True)
    print(f"  corr(enc, y): {np.corrcoef(ta_enc, y)[0,1]:+.4f}", flush=True)
    te_A_train = ta_enc
    te_A_test = te_enc

    # ─── Variant B: scenario_id (may have low granularity — each scenario is 25 rows) ───
    print("\n[B] Scenario-level target encoding...", flush=True)
    ta_enc, te_enc = smooth_target_encoding(train, test, ["layout_id", "scenario_id"], y, fold_idx, smoothing=5)
    print(f"  corr(enc, y): {np.corrcoef(ta_enc, y)[0,1]:+.4f}", flush=True)
    te_B_train = ta_enc
    te_B_test = te_enc

    # ─── Variant C: bucket-level (layout_type × ts_bucket × cong_bin) ───
    print("\n[C] Bucket-level (layout_type × ts_bucket × cong_bin) encoding...", flush=True)
    ta_enc, te_enc = smooth_target_encoding(train, test, ["layout_type", "ts_bucket", "cong_bin"], y, fold_idx, smoothing=50)
    print(f"  corr(enc, y): {np.corrcoef(ta_enc, y)[0,1]:+.4f}", flush=True)
    te_C_train = ta_enc
    te_C_test = te_enc

    # ─── Variant D: Shape-cluster (KMeans on scenario y trajectory proxy) ───
    # Use scenario-level features (order_inflow, congestion, robot_util trajectories) for clustering
    print("\n[D] Shape-cluster encoding (KMeans on trajectory)...", flush=True)
    # Build 25-dim trajectory per (layout_id, scenario_id) for order_inflow + congestion + robot_util
    sc_traj = []
    sc_keys = []
    for (lid, sid), grp in train.groupby(["layout_id", "scenario_id"]):
        if len(grp) == 25:
            traj = np.concatenate([
                grp["order_inflow_15m"].fillna(0).values,
                grp["congestion_score"].fillna(0).values,
                grp["robot_utilization"].fillna(0).values,
            ])
            sc_traj.append(traj)
            sc_keys.append((lid, sid))
    sc_traj = np.array(sc_traj)
    # Train KMeans on train trajectories
    from sklearn.preprocessing import StandardScaler
    scaler_traj = StandardScaler().fit(sc_traj)
    sc_traj_std = scaler_traj.transform(sc_traj)
    n_clusters = 50
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit(sc_traj_std)
    sc_cluster = pd.DataFrame({"layout_id": [k[0] for k in sc_keys],
                                "scenario_id": [k[1] for k in sc_keys],
                                "cluster": km.labels_})
    train = train.merge(sc_cluster, on=["layout_id", "scenario_id"], how="left")
    # Test: same clustering
    sc_traj_test = []
    sc_keys_test = []
    for (lid, sid), grp in test.groupby(["layout_id", "scenario_id"]):
        if len(grp) == 25:
            traj = np.concatenate([
                grp["order_inflow_15m"].fillna(0).values,
                grp["congestion_score"].fillna(0).values,
                grp["robot_utilization"].fillna(0).values,
            ])
            sc_traj_test.append(traj)
            sc_keys_test.append((lid, sid))
    sc_traj_test = np.array(sc_traj_test)
    sc_traj_test_std = scaler_traj.transform(sc_traj_test)
    test_cluster_labels = km.predict(sc_traj_test_std)
    sc_cluster_test = pd.DataFrame({"layout_id": [k[0] for k in sc_keys_test],
                                     "scenario_id": [k[1] for k in sc_keys_test],
                                     "cluster": test_cluster_labels})
    test = test.merge(sc_cluster_test, on=["layout_id", "scenario_id"], how="left")
    # Train encoding on cluster
    ta_enc, te_enc = smooth_target_encoding(train, test, ["cluster"], y, fold_idx, smoothing=100)
    print(f"  n_clusters: {n_clusters}, corr(enc, y): {np.corrcoef(ta_enc, y)[0,1]:+.4f}", flush=True)
    te_D_train = ta_enc
    te_D_test = te_enc

    # ─── Combine and train ───
    te_cols = np.column_stack([te_A_train, te_B_train, te_C_train, te_D_train])
    te_cols_test = np.column_stack([te_A_test, te_B_test, te_C_test, te_D_test])

    X = np.hstack([train_fe[feat_cols].values, te_cols])
    X_test = np.hstack([test_fe[feat_cols].values, te_cols_test])
    print(f"\nCombined X shape: {X.shape}", flush=True)

    # Train LGB_Huber
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"Reference v23 LGB_Huber OOF: {old_mae:.5f}", flush=True)

    print("\n=== Training LGB_Huber on v23 + 4 TE features ===", flush=True)
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
    new_test = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))
    corr = float(np.corrcoef(y - old_oof, y - new_oof)[0, 1])

    print(f"\n=== Results ===", flush=True)
    print(f"Old: {old_mae:.5f}", flush=True)
    print(f"New: {new_mae:.5f}", flush=True)
    print(f"Delta: {new_mae - old_mae:+.5f}", flush=True)
    print(f"corr (old v23 LGB_Huber): {corr:.5f}", flush=True)

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))
    corr_mega = float(np.corrcoef(y - mega_oof, y - new_oof)[0, 1])
    print(f"corr vs mega33: {corr_mega:.5f}", flush=True)

    # Blend
    best_mae = mega_mae; best_w = 0
    for w in np.linspace(0, 0.3, 31):
        pred = (1 - w) * mega_oof + w * new_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m; best_w = w
    print(f"Blend mega33+new: w={best_w:.3f} MAE={best_mae:.5f} delta={best_mae-mega_mae:+.5f}", flush=True)

    # Kill gates
    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95
    print(f"\nKill gates: hurt={hurt}, too_similar={too_similar}", flush=True)

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test)
    np.save(os.path.join(OUT, "te_train.npy"), te_cols)
    np.save(os.path.join(OUT, "te_test.npy"), te_cols_test)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae, delta=new_mae - old_mae,
        corr=corr, corr_mega=corr_mega,
        blend_w=best_w, blend_mae=best_mae, blend_delta=best_mae - mega_mae,
        gates=dict(hurt=bool(hurt), too_similar=bool(too_similar)),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\nVERDICT: " + ("NO_GO" if (hurt or too_similar) else "PROCEED"), flush=True)


if __name__ == "__main__":
    main()
