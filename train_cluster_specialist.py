"""
Test-only clustering + specialized layout models (D2).

1. Build LAYOUT profile (mean of features per layout) for train and test together
2. KMeans cluster layouts into N groups (k=5..10)
3. For each cluster:
   - Find train layouts in this cluster
   - Train LGB on their data
   - Predict test rows whose layout is in this cluster
4. Aggregate OOF (with GroupKFold) and test predictions
5. Check residual_corr + blend with mega33

Different because each cluster-specific model sees smaller, more homogeneous data.
Residuals should differ from global mega33.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

OUT = "results/cluster_spec"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"
N_CLUSTERS = 6

print("=" * 64)
print("Test-only Layout Clustering + Specialized Models")
print("=" * 64)
t0 = time.time()

# Load
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

# --- Build layout profile (mean features per layout) ---
print("\n[1] Building layout profiles...")
train_layouts = train_fe.groupby("layout_id", sort=False)[feat_cols].mean()
test_layouts = test_fe.groupby("layout_id", sort=False)[feat_cols].mean()
print(f"  train layouts: {train_layouts.shape}, test layouts: {test_layouts.shape}")

all_layouts = pd.concat([train_layouts, test_layouts], axis=0)
all_layouts = all_layouts.fillna(all_layouts.median())

scaler = StandardScaler()
all_std = scaler.fit_transform(all_layouts.values)

# --- KMeans clustering ---
print(f"\n[2] KMeans clustering (k={N_CLUSTERS})...")
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
all_clusters = km.fit_predict(all_std)

n_train_layouts = len(train_layouts)
train_layout_clusters = all_clusters[:n_train_layouts]
test_layout_clusters = all_clusters[n_train_layouts:]

# map cluster counts
print("  cluster sizes (train | test):")
for c in range(N_CLUSTERS):
    tr_c = (train_layout_clusters == c).sum()
    te_c = (test_layout_clusters == c).sum()
    print(f"    cluster {c}: train {tr_c}, test {te_c}")

# map row-level clusters
train_layout_to_cluster = dict(zip(train_layouts.index.values, train_layout_clusters))
test_layout_to_cluster = dict(zip(test_layouts.index.values, test_layout_clusters))

train_row_cluster = train_fe["layout_id"].map(train_layout_to_cluster).values.astype(np.int32)
test_row_cluster = test_fe["layout_id"].map(test_layout_to_cluster).values.astype(np.int32)
print(f"\n  train row clusters dist: {np.bincount(train_row_cluster)}")
print(f"  test row clusters dist:  {np.bincount(test_row_cluster)}")

# --- Train cluster-specific models ---
print(f"\n[3] Training cluster-specific LGB models (5-fold each)...")
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

PARAMS = dict(
    objective="huber", n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

cluster_oof = np.zeros(len(y))
cluster_test = np.zeros(len(X_te))

for c in range(N_CLUSTERS):
    tv = time.time()
    train_mask = train_row_cluster == c
    test_mask = test_row_cluster == c
    n_tr = train_mask.sum()
    n_te = test_mask.sum()
    print(f"\n  === cluster {c}: n_train={n_tr} n_test={n_te} ===")
    if n_tr < 1000:
        print(f"    too few, fallback to mega33 for this cluster")
        cluster_oof[train_mask] = mega33_oof[train_mask]
        cluster_test[test_mask] = mega33_test[test_mask]
        continue

    # within this cluster, run 5-fold using fold_ids
    X_tr_c = X_tr[train_mask]
    y_log_c = y_log[train_mask]
    fold_ids_c = fold_ids[train_mask]
    y_c = y[train_mask]

    c_oof = np.zeros(n_tr)
    c_test_sum = np.zeros(n_te)

    for f in range(5):
        tr = np.where(fold_ids_c != f)[0]
        val = np.where(fold_ids_c == f)[0]
        if len(tr) < 500 or len(val) < 100:
            print(f"    fold {f} too small, skip")
            continue
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_tr_c[tr], y_log_c[tr],
              eval_set=[(X_tr_c[val], y_log_c[val])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        c_oof[val] = np.clip(np.expm1(m.predict(X_tr_c[val])), 0, None)
        c_test_sum += np.clip(np.expm1(m.predict(X_te[test_mask])), 0, None) / 5
        del m
        gc.collect()

    c_mae = mean_absolute_error(y_c, c_oof)
    print(f"    cluster {c} OOF mae: {c_mae:.5f} ({time.time()-tv:.0f}s)")

    cluster_oof[train_mask] = c_oof
    cluster_test[test_mask] = c_test_sum

# --- Overall metrics ---
cl_mae = mean_absolute_error(y, cluster_oof)
print(f"\n[4] Overall cluster-specialized OOF: {cl_mae:.5f}")

r_mega = y - mega33_oof
r_cl = y - cluster_oof
corr_cl = float(np.corrcoef(r_mega, r_cl)[0, 1])
print(f"  residual_corr(mega33, cluster_spec): {corr_cl:.4f}")

# --- Blend with mega33 ---
best_w, best_mae = 0, float(baseline_mae)
for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]:
    blend = (1 - w) * mega33_oof + w * cluster_oof
    m = float(mean_absolute_error(y, blend))
    print(f"    blend w={w:.2f}: mae={m:.5f} delta={m - baseline_mae:+.5f}")
    if m < best_mae:
        best_w, best_mae = w, m

delta = best_mae - baseline_mae
verdict = "STRONG_GO" if delta < -0.01 else ("GO" if delta < -0.005 else ("WEAK" if delta < -0.002 else "NO_GO"))

np.save(f"{OUT}/cluster_oof.npy", cluster_oof)
np.save(f"{OUT}/cluster_test.npy", cluster_test)

if delta < -0.003:
    blend_test = (1 - best_w) * mega33_test + best_w * cluster_test
    test_raw = pd.read_csv("test.csv")
    pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_cluster_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission: {OUT}/submission_mega33_cluster_w{int(best_w*100)}.csv")

json.dump({
    "baseline_mae": float(baseline_mae),
    "cluster_oof_mae": float(cl_mae),
    "residual_corr": corr_cl,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(delta),
    "verdict": verdict,
    "n_clusters": N_CLUSTERS,
    "elapsed": round(time.time()-t0, 1),
}, open(f"{OUT}/cluster_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}  corr={corr_cl:.4f}  delta={delta:+.5f}")
print(f"elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
