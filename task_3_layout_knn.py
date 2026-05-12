"""
Task 3: Layout-level k-NN base.

Use layout_info static features (aisle_width_avg, intersection_count, one_way_ratio,
pack_station_count, charger_count, layout_compactness, floor_area_sqm, robot_total, ...)
to compute layout-similarity.

For each (train row, test row):
  - Find top-K similar train layouts via layout_info feature distance
  - Prediction = weighted mean/median of y within those layouts (filtered to similar scenarios)

5-fold GroupKFold: for each fold, train layouts = fold-train layouts, val = fold-val layouts.

Two variants:
  Variant A: layout-level mean y (simple per-layout baseline)
  Variant B: layout-level + scenario-level: within top-K layouts, find most similar scenarios
             by scenario-level feature similarity (order_mean, congestion_mean, ...)
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
from sklearn.neighbors import NearestNeighbors

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "task_3_knn")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Task 3: Layout-level k-NN base", flush=True)
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
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # Load layout_info
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    print(f"layout_info columns: {list(layout_info.columns)}", flush=True)

    # Layout-level features (numeric only)
    li_cols = [c for c in layout_info.columns if c not in ["layout_id", "layout_type"]]
    # add one-hot for layout_type
    layout_info_num = layout_info[["layout_id"] + li_cols].copy()
    # layout_type onehot
    lt = pd.get_dummies(layout_info["layout_type"], prefix="lt")
    layout_info_num = pd.concat([layout_info_num, lt], axis=1)
    layout_info_num = layout_info_num.fillna(0)
    print(f"Layout features: {len(li_cols) + len(lt.columns)}", flush=True)

    # Build layout feature matrix indexed by layout_id
    layout_features = layout_info_num.set_index("layout_id")
    feat_cols = [c for c in layout_features.columns]

    # Scenario-level features from raw train/test (per-scenario aggregates)
    sc_agg_cols = ["order_inflow_15m", "congestion_score", "robot_utilization",
                   "battery_mean", "fault_count_15m", "blocked_path_15m"]
    def build_sc_features(df):
        agg = df.groupby(["layout_id", "scenario_id"])[sc_agg_cols].agg(["mean", "std", "max"])
        agg.columns = [f"{c}_{s}" for c, s in agg.columns]
        return agg.reset_index()

    train_sc = build_sc_features(train)
    test_sc = build_sc_features(test)
    sc_feat_cols = [c for c in train_sc.columns if c not in ["layout_id", "scenario_id"]]

    # Merge layout_features into sc
    train_sc = train_sc.merge(layout_features.reset_index(), on="layout_id", how="left")
    test_sc = test_sc.merge(layout_features.reset_index(), on="layout_id", how="left")

    all_feat = feat_cols + sc_feat_cols
    train_sc[all_feat] = train_sc[all_feat].fillna(0)
    test_sc[all_feat] = test_sc[all_feat].fillna(0)

    # Scenario-level y (mean y in that scenario)
    sc_y = train.groupby(["layout_id", "scenario_id"])["avg_delay_minutes_next_30m"].mean().reset_index()
    sc_y.columns = ["layout_id", "scenario_id", "y_sc_mean"]
    train_sc = train_sc.merge(sc_y, on=["layout_id", "scenario_id"], how="left")

    # For each train row, link to its scenario index in train_sc
    train_scenario_idx = train.groupby(["layout_id", "scenario_id"]).ngroup()
    # scenario-level y_sc for each training row
    sc_y_map = dict(zip(zip(train_sc["layout_id"], train_sc["scenario_id"]), train_sc["y_sc_mean"]))
    # We'll predict y_sc_mean for each scenario and broadcast to rows
    # Then optionally adjust with per-ts pattern

    K = 30  # Number of nearest scenarios
    print(f"k-NN k={K}", flush=True)

    # Build OOF predictions at scenario level, then broadcast to row level
    sc_oof = np.zeros(len(train_sc), dtype=np.float64)  # per scenario
    sc_test_pred = np.zeros(len(test_sc), dtype=np.float64)

    # 5-fold: for each fold, train = other folds' scenarios, val = this fold's scenarios
    # Map each train_sc row to fold via layout_id
    layout_to_fold = {}
    for layout_id, fold in zip(train["layout_id"], fold_idx):
        layout_to_fold[layout_id] = fold
    train_sc["_fold"] = train_sc["layout_id"].map(layout_to_fold)
    print(f"train_sc fold counts: {train_sc['_fold'].value_counts().sort_index().to_dict()}", flush=True)

    scaler = StandardScaler()
    Xtrain_sc = scaler.fit_transform(train_sc[all_feat].values)
    Xtest_sc = scaler.transform(test_sc[all_feat].values)

    # Precompute for test: kNN uses ALL train scenarios (no val exclusion needed)
    # But for test kNN, we must use all train scenarios, per-fold to honor fold structure not strictly
    # Actually for test prediction we just use full train.

    # Per-fold OOF
    for fold in range(5):
        val_mask = train_sc["_fold"].values == fold
        tr_mask = ~val_mask
        nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
        nn.fit(Xtrain_sc[tr_mask])
        dist, ind = nn.kneighbors(Xtrain_sc[val_mask])
        y_tr_sc = train_sc.loc[tr_mask, "y_sc_mean"].values
        # Weighted prediction: inverse distance weights (avoid 0)
        w = 1.0 / (dist + 1e-3)
        w /= w.sum(axis=1, keepdims=True)
        preds = (y_tr_sc[ind] * w).sum(axis=1)
        sc_oof[val_mask] = preds
        print(f"  fold {fold}: val scenarios={val_mask.sum()}", flush=True)

    # Test prediction: fit on ALL train scenarios (out-of-time use)
    nn = NearestNeighbors(n_neighbors=K, metric="euclidean")
    nn.fit(Xtrain_sc)
    dist, ind = nn.kneighbors(Xtest_sc)
    y_tr_sc_full = train_sc["y_sc_mean"].values
    w = 1.0 / (dist + 1e-3)
    w /= w.sum(axis=1, keepdims=True)
    sc_test_pred = (y_tr_sc_full[ind] * w).sum(axis=1)

    # Broadcast scenario predictions to rows (each scenario has 25 rows)
    train_sc_pred_map = dict(zip(zip(train_sc["layout_id"], train_sc["scenario_id"]), sc_oof))
    test_sc_pred_map = dict(zip(zip(test_sc["layout_id"], test_sc["scenario_id"]), sc_test_pred))
    knn_oof = np.array([train_sc_pred_map[(lid, sid)]
                         for lid, sid in zip(train["layout_id"], train["scenario_id"])])
    knn_test = np.array([test_sc_pred_map[(lid, sid)]
                          for lid, sid in zip(test["layout_id"], test["scenario_id"])])

    knn_mae = float(mean_absolute_error(y, knn_oof))

    # Compare
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))

    corr = float(np.corrcoef(y - mega_oof, y - knn_oof)[0, 1])

    print(f"\n=== Layout kNN results ===", flush=True)
    print(f"Single OOF MAE: {knn_mae:.5f}", flush=True)
    print(f"Mega33 OOF MAE: {mega_mae:.5f}", flush=True)
    print(f"residual_corr vs mega33: {corr:.5f}", flush=True)

    # Blend test
    best_mae = mega_mae
    best_w = 0
    for w in np.linspace(0, 0.5, 51):
        pred = (1-w) * mega_oof + w * knn_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m
            best_w = w
    print(f"Best 2-way blend: w={best_w:.3f}, MAE={best_mae:.5f}, delta={best_mae - mega_mae:+.5f}", flush=True)

    # Kill gates
    bad_quality = knn_mae > 10
    too_similar = corr >= 0.95
    print("\nKill gates:", flush=True)
    print(f"  single MAE > 10: {bad_quality}", flush=True)
    print(f"  corr(mega) >= 0.95: {too_similar}", flush=True)

    np.save(os.path.join(OUT, "knn_oof.npy"), knn_oof)
    np.save(os.path.join(OUT, "knn_test.npy"), knn_test)
    summary = dict(
        single_mae=knn_mae,
        corr_vs_mega=corr,
        best_blend_w=best_w,
        best_blend_mae=best_mae,
        delta=best_mae - mega_mae,
        gates=dict(bad_quality=bad_quality, too_similar=too_similar),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if bad_quality or too_similar:
        print("\nVERDICT: NO_GO", flush=True)
    else:
        print("\nVERDICT: PROCEED — include in multi-blend", flush=True)


if __name__ == "__main__":
    main()
