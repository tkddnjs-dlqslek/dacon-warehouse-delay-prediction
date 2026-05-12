"""
V_ORTH: Orthogonal Feature Set (v23와 완전히 다른 철학).

v23 is built on: lag/rolling, SC (scenario aggregates), forward-looking.
V_ORTH is built on: raw columns + cyclical + PCA + clustering + rank + layout static.

Goal: create features that lead a GBDT to fundamentally different splits.
Target: residual_corr(mega33, V_ORTH_lgb) < 0.95 -> genuine stacking value.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import lightgbm as lgb

OUT = "results/vorth"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("V_ORTH: orthogonal feature set experiment")
print("=" * 64)
t0 = time.time()

# --- load raw data (NOT v30 cache) ---
print("loading raw data...")
train_raw = pd.read_csv("train.csv")
test_raw = pd.read_csv("test.csv")
layout = pd.read_csv("layout_info.csv")
print(f"  train: {train_raw.shape}, test: {test_raw.shape}, layout: {layout.shape}")

# sort to match mega33 oof / fold_ids
train_raw = train_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
# test sort: engineer_features_v23 sorts by layout_id+scenario_id+timeslot; we match that too
test_raw = test_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)

# timeslot
train_raw["timeslot"] = train_raw.groupby(["layout_id", "scenario_id"]).cumcount()
test_raw["timeslot"] = test_raw.groupby(["layout_id", "scenario_id"]).cumcount()

# merge layout (keep layout_static unlike v23)
train_raw = train_raw.merge(layout, on="layout_id", how="left")
test_raw = test_raw.merge(layout, on="layout_id", how="left")

# need fold_ids aligned with train_raw (sorted order)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_raw[TARGET].values.astype(np.float64)
y_log = np.log1p(y)

# sanity: mega33 baseline (alignment check)
with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
assert abs(baseline_mae - 8.3989) < 0.005, f"baseline drift: {baseline_mae}"
print(f"  mega33 baseline: {baseline_mae:.5f}")

# --- V_ORTH feature engineering ---
print("\n[1] V_ORTH feature engineering...")

def engineer_vorth(df, is_train=True, pca_model=None, kmeans10=None, kmeans20=None,
                  col_mean=None, col_std=None, raw_num_cols=None, train_y=None):
    """Engineer orthogonal features. Returns (df_out, feat_cols, fitted_transforms)."""

    # base: ALL numeric columns except id/target/timeslot-aux
    exclude = {"ID", "layout_id", "scenario_id", TARGET, "layout_type",
               "timeslot_sq", "timeslot_norm"}
    # identify numeric cols (raw)
    if raw_num_cols is None:
        raw_num_cols = [c for c in df.columns if c not in exclude
                        and pd.api.types.is_numeric_dtype(df[c])]
    df = df.copy()

    # 1) cyclical timeslot
    df["ts_sin"] = np.sin(2 * np.pi * df["timeslot"] / 25.0)
    df["ts_cos"] = np.cos(2 * np.pi * df["timeslot"] / 25.0)

    # 2) timeslot position buckets (not sequential!)
    df["ts_is_start"] = (df["timeslot"] < 5).astype(np.int8)
    df["ts_is_mid"] = ((df["timeslot"] >= 5) & (df["timeslot"] < 20)).astype(np.int8)
    df["ts_is_end"] = (df["timeslot"] >= 20).astype(np.int8)

    # 3) deviation from scenario mean (simple, not cumulative)
    #    different from v23's cummean/sc_dev
    key8 = ["order_inflow_15m", "congestion_score", "robot_utilization",
            "battery_mean", "fault_count_15m", "blocked_path_15m",
            "pack_utilization", "charge_queue_length"]
    grp = df.groupby(["layout_id", "scenario_id"], sort=False)
    for c in key8:
        if c in df.columns:
            sc_mean = grp[c].transform("mean")
            df[f"{c}_abs_dev_from_mean"] = np.abs(df[c] - sc_mean)

    # 4) rank within scenario (row rank on y is leakage, so use feature ranks)
    for c in key8:
        if c in df.columns:
            df[f"{c}_rank_in_sc"] = grp[c].rank(pct=True)

    # 5) layout type one-hot (v23 dropped layout_type!)
    for t in ["narrow", "grid", "hybrid", "hub_spoke"]:
        df[f"layout_type_{t}"] = (df["layout_type"] == t).astype(np.int8)

    # build numeric feature matrix for PCA / KMeans (on raw 90 + timeslot)
    mat_cols = raw_num_cols + ["timeslot", "ts_sin", "ts_cos"]
    mat_cols = [c for c in mat_cols if c in df.columns]
    mat = df[mat_cols].values.astype(np.float64)
    # impute with column mean (fit on train, apply on test)
    if col_mean is None:
        col_mean = np.nanmean(mat, axis=0)
    mat = np.where(np.isnan(mat), col_mean, mat)
    if col_std is None:
        col_std = mat.std(axis=0) + 1e-6
    mat_std = (mat - mat.mean(axis=0)) / col_std

    # 6) PCA(20)
    if pca_model is None:
        pca_model = PCA(n_components=20, random_state=42)
        pca_model.fit(mat_std)
    pca_feats = pca_model.transform(mat_std)
    for i in range(20):
        df[f"pca_{i:02d}"] = pca_feats[:, i].astype(np.float32)

    # 7) KMeans cluster labels (k=10, k=20) — as CATEGORICAL integers
    if kmeans10 is None:
        kmeans10 = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=4096, n_init="auto")
        kmeans10.fit(mat_std)
    df["kmeans10"] = kmeans10.predict(mat_std).astype(np.int16)

    if kmeans20 is None:
        kmeans20 = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=4096, n_init="auto")
        kmeans20.fit(mat_std)
    df["kmeans20"] = kmeans20.predict(mat_std).astype(np.int16)

    # 8) drop columns we don't want in final feature set
    # Remove v23-style engineered stuff if any (they're not present here since we started from raw)
    # Also remove non-feature cols
    drop_cols = {"ID", "layout_id", "scenario_id", TARGET, "layout_type"}
    feat_cols_out = [c for c in df.columns if c not in drop_cols]

    # Ensure only numeric in feat_cols
    feat_cols_out = [c for c in feat_cols_out if pd.api.types.is_numeric_dtype(df[c])]

    return df, feat_cols_out, {
        "pca": pca_model, "k10": kmeans10, "k20": kmeans20,
        "col_mean": col_mean, "col_std": col_std, "raw_num_cols": raw_num_cols,
    }


train_vo, feat_cols, fitted = engineer_vorth(train_raw, is_train=True)
print(f"  train V_ORTH: {train_vo.shape}, feat_cols: {len(feat_cols)}")

test_vo, _, _ = engineer_vorth(
    test_raw,
    is_train=False,
    pca_model=fitted["pca"], kmeans10=fitted["k10"], kmeans20=fitted["k20"],
    col_mean=fitted["col_mean"], col_std=fitted["col_std"], raw_num_cols=fitted["raw_num_cols"],
)
print(f"  test V_ORTH: {test_vo.shape}")

# align feat_cols across train/test
feat_cols = [c for c in feat_cols if c in test_vo.columns]
print(f"  final feat_cols: {len(feat_cols)}")

# check a few new features
print("\n  sample new feature stats (train):")
for c in ["ts_sin", "ts_cos", "layout_type_hub_spoke", "pca_00", "kmeans10",
         "fault_count_15m_abs_dev_from_mean", "fault_count_15m_rank_in_sc"]:
    if c in train_vo.columns:
        v = train_vo[c].values
        print(f"    {c}: min={v.min():.3f} max={v.max():.3f} mean={v.mean():.3f}")

# --- train LGB ---
print("\n[2] Train LGB-Huber (V_ORTH) 5-fold...")
X_tr = train_vo[feat_cols].values.astype(np.float32)
X_te = test_vo[feat_cols].values.astype(np.float32)

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

vo_oof = np.zeros(len(y))
vo_test = np.zeros(len(X_te))
fold_rows = []

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr], y_log[tr],
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    vo_oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    vo_test += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

    fm = mean_absolute_error(y[val], vo_oof[val])
    fold_rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tv:.0f}s)")

vo_mae = mean_absolute_error(y, vo_oof)
print(f"\nV_ORTH OOF: {vo_mae:.5f}")

np.save(f"{OUT}/vorth_oof.npy", vo_oof)
np.save(f"{OUT}/vorth_test.npy", vo_test)
pd.DataFrame(fold_rows).to_csv(f"{OUT}/vorth_fold_metrics.csv", index=False)

# --- diversity check ---
r_mega = y - mega33_oof
r_vo = y - vo_oof
corr = float(np.corrcoef(r_mega, r_vo)[0, 1])
print(f"\n[3] residual_corr(mega33, V_ORTH): {corr:.4f}  (goal <0.95)")

best_w, best_mae = 0, float(baseline_mae)
scan = []
for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    blend = (1 - w) * mega33_oof + w * vo_oof
    mae = float(mean_absolute_error(y, blend))
    scan.append({"w": w, "mae": mae, "delta": mae - baseline_mae})
    tag = " ***" if mae - baseline_mae < -0.01 else (" **" if mae - baseline_mae < -0.005 else "")
    print(f"  blend w={w:.2f}: mae={mae:.5f} delta={mae - baseline_mae:+.5f}{tag}")
    if mae < best_mae:
        best_w, best_mae = w, mae

verdict = "STRONG_GO" if best_mae - baseline_mae < -0.02 else (
    "GO" if best_mae - baseline_mae < -0.008 else (
    "WEAK" if best_mae - baseline_mae < -0.003 else "NO_GO"))

if best_w > 0 and best_mae < baseline_mae:
    blend_test = (1 - best_w) * mega33_test + best_w * vo_test
    # test ID order
    test_ids = pd.read_csv("test.csv")["ID"].values  # raw test order
    # we sorted test_raw by layout_id+scenario_id; need to re-map
    sorted_test_ids = test_vo["ID"].values
    # create (ID -> prediction) map, then output in raw order
    pred_map = dict(zip(sorted_test_ids, blend_test))
    sub_preds = np.array([pred_map[id_] for id_ in test_ids])
    pd.DataFrame({"ID": test_ids, TARGET: np.clip(sub_preds, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_vorth_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission saved: {OUT}/submission_mega33_vorth_w{int(best_w*100)}.csv")

summary = {
    "vorth_oof_mae": float(vo_mae),
    "mega33_baseline": float(baseline_mae),
    "n_features": len(feat_cols),
    "residual_corr": corr,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(best_mae - baseline_mae),
    "verdict": verdict,
    "fold_rows": fold_rows,
    "blend_scan": scan,
    "elapsed": round(time.time() - t0, 1),
}
with open(f"{OUT}/vorth_summary.json", "w") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}")
print(f"  corr={corr:.4f}  single OOF={vo_mae:.5f}  best_delta={best_mae - baseline_mae:+.5f}")
print(f"  elapsed: {time.time() - t0:.0f}s")
print(f"{'='*64}")
