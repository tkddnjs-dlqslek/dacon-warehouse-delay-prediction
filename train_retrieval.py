"""
Retrieval-Augmented Features Experiment.

For each scenario, find k-NN in train scenarios (GroupKFold-safe),
add neighbor y-stats as new features, train LGB-Huber.

Hypothesis: retrieval features add a new information axis (explicit "similar
past situations") that neither v23 nor mega33 captures -> residual_corr < 0.95.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb

OUT = "results/retrieval"
import os
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"
K_NEIGHBORS = 10

print("=" * 64)
print("Retrieval-Augmented Features")
print("=" * 64)
t0 = time.time()

# --- load ---
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
print(f"mega33 baseline: {baseline_mae:.5f}  (loaded {time.time()-t0:.0f}s)")

# --- build scenario-level feature matrix ---
# group by (layout_id, scenario_id), mean of features
print("\n[1] Building scenario-level feature matrices...")
train_grouped = train_fe.groupby(["layout_id", "scenario_id"], sort=False)
train_scenario_feats = train_grouped[feat_cols].mean()  # ~10K scenarios
train_scenario_y = train_grouped[TARGET].mean()
# scenario fold_ids: pick first row of each group
train_scenario_fold = train_grouped.apply(lambda g: fold_ids[g.index[0]])

print(f"  train scenarios: {len(train_scenario_feats)}")
assert (train_scenario_fold.values >= 0).all()

test_grouped = test_fe.groupby(["layout_id", "scenario_id"], sort=False)
test_scenario_feats = test_grouped[feat_cols].mean()
print(f"  test scenarios: {len(test_scenario_feats)}")

# --- standardize scenario features (Euclidean distance) ---
# fill NaN with col mean, standardize
print("\n[2] Standardizing scenario features for kNN...")
tr_mat = train_scenario_feats.values.astype(np.float32)
te_mat = test_scenario_feats.values.astype(np.float32)
col_mean = np.nanmean(np.vstack([tr_mat, te_mat]), axis=0)
tr_mat = np.where(np.isnan(tr_mat), col_mean, tr_mat)
te_mat = np.where(np.isnan(te_mat), col_mean, te_mat)
col_std = tr_mat.std(axis=0) + 1e-6
tr_mat = (tr_mat - tr_mat.mean(axis=0)) / col_std
te_mat = (te_mat - tr_mat.mean(axis=0)) / col_std  # use tr stats
# Note: te uses tr statistics (standard practice)

# --- build scenario-level retrieval features, GroupKFold-safe ---
print(f"\n[3] Building retrieval features (GroupKFold-safe, k={K_NEIGHBORS})...")
retrieval_scenario = np.zeros((len(tr_mat), 5), dtype=np.float64)  # 5 new features per scenario

sc_fold = train_scenario_fold.values.astype(np.int8)
tr_y = train_scenario_y.values.astype(np.float64)

for f in range(5):
    tv = time.time()
    tr_src_mask = sc_fold != f     # training scenarios for this fold
    val_mask = sc_fold == f        # val scenarios

    nn = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto", n_jobs=-1)
    nn.fit(tr_mat[tr_src_mask])
    dist, idx = nn.kneighbors(tr_mat[val_mask])
    # map neighbor indices back to global
    tr_src_global = np.where(tr_src_mask)[0]
    neighbor_y = tr_y[tr_src_global[idx]]  # (n_val, k)

    val_idx_global = np.where(val_mask)[0]
    retrieval_scenario[val_idx_global, 0] = neighbor_y.mean(axis=1)
    retrieval_scenario[val_idx_global, 1] = neighbor_y.std(axis=1)
    retrieval_scenario[val_idx_global, 2] = np.median(neighbor_y, axis=1)
    retrieval_scenario[val_idx_global, 3] = dist[:, 0]  # nearest dist
    retrieval_scenario[val_idx_global, 4] = dist.mean(axis=1)  # mean dist
    print(f"  fold {f}: n_tr={tr_src_mask.sum()} n_val={val_mask.sum()} ({time.time()-tv:.0f}s)")
    del nn, dist, idx, neighbor_y

# test retrieval: use ALL train
tv = time.time()
nn_full = NearestNeighbors(n_neighbors=K_NEIGHBORS, algorithm="auto", n_jobs=-1)
nn_full.fit(tr_mat)
dist_te, idx_te = nn_full.kneighbors(te_mat)
retrieval_scenario_test = np.column_stack([
    tr_y[idx_te].mean(axis=1),
    tr_y[idx_te].std(axis=1),
    np.median(tr_y[idx_te], axis=1),
    dist_te[:, 0],
    dist_te.mean(axis=1),
])
print(f"  test retrieval ({time.time()-tv:.0f}s)")
del nn_full, dist_te, idx_te

# --- broadcast scenario retrieval to row level ---
print("\n[4] Broadcasting scenario retrieval to row level...")
# map row -> scenario index
train_scenario_keys = list(train_scenario_feats.index)
key_to_idx_tr = {k: i for i, k in enumerate(train_scenario_keys)}
train_row_scenario_idx = train_fe[["layout_id", "scenario_id"]].apply(
    lambda r: key_to_idx_tr[(r["layout_id"], r["scenario_id"])], axis=1
).values.astype(np.int32)
retrieval_tr_row = retrieval_scenario[train_row_scenario_idx]

test_scenario_keys = list(test_scenario_feats.index)
key_to_idx_te = {k: i for i, k in enumerate(test_scenario_keys)}
test_row_scenario_idx = test_fe[["layout_id", "scenario_id"]].apply(
    lambda r: key_to_idx_te[(r["layout_id"], r["scenario_id"])], axis=1
).values.astype(np.int32)
retrieval_te_row = retrieval_scenario_test[test_row_scenario_idx]

new_feat_names = ["retr_neigh_y_mean", "retr_neigh_y_std", "retr_neigh_y_median",
                  "retr_nearest_dist", "retr_mean_dist"]

# Sanity: retrieval stats
print(f"  retrieval_tr_row shape: {retrieval_tr_row.shape}")
for i, name in enumerate(new_feat_names):
    vals = retrieval_tr_row[:, i]
    print(f"  {name}: mean={vals.mean():.4f} std={vals.std():.4f}")

# --- corr check: how does retrieval_neigh_y_mean correlate with y? ---
from scipy.stats import spearmanr
rho = spearmanr(retrieval_tr_row[:, 0], y).statistic
print(f"\n  Spearman(retr_neigh_y_mean, y) = {rho:.4f}")

# --- build feature matrices for LGB ---
print("\n[5] Training LGB-Huber (v23 + retrieval) 5-fold...")
X_tr_base = train_fe[feat_cols].values.astype(np.float32)
X_te_base = test_fe[feat_cols].values.astype(np.float32)
X_tr = np.hstack([X_tr_base, retrieval_tr_row.astype(np.float32)])
X_te = np.hstack([X_te_base, retrieval_te_row.astype(np.float32)])
del X_tr_base, X_te_base
gc.collect()
print(f"  X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

retr_oof = np.zeros(len(y))
retr_test = np.zeros(len(X_te))
fold_rows = []
for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr], y_log[tr],
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    retr_oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    retr_test += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

    fm = mean_absolute_error(y[val], retr_oof[val])
    fold_rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tv:.0f}s)")

retr_mae = mean_absolute_error(y, retr_oof)
print(f"\nRetrieval-LGB OOF: {retr_mae:.5f}")

np.save(f"{OUT}/retr_oof.npy", retr_oof)
np.save(f"{OUT}/retr_test.npy", retr_test)
pd.DataFrame(fold_rows).to_csv(f"{OUT}/retr_fold_metrics.csv", index=False)

# --- diversity check ---
r_mega = y - mega33_oof
r_retr = y - retr_oof
corr = float(np.corrcoef(r_mega, r_retr)[0, 1])
print(f"\n[6] Diversity check")
print(f"  residual_corr(mega33, retrieval): {corr:.4f}  (goal < 0.95)")

best_w, best_mae = 0, float(baseline_mae)
scan = []
for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]:
    blend = (1 - w) * mega33_oof + w * retr_oof
    mae = float(mean_absolute_error(y, blend))
    scan.append({"w": w, "mae": mae, "delta": mae - baseline_mae})
    tag = " ***" if mae - baseline_mae < -0.01 else (" **" if mae - baseline_mae < -0.005 else "")
    print(f"  blend w={w:.2f}: mae={mae:.5f} delta={mae - baseline_mae:+.5f}{tag}")
    if mae < best_mae:
        best_w, best_mae = w, mae

verdict = "STRONG GO" if best_mae - baseline_mae < -0.02 else (
    "GO" if best_mae - baseline_mae < -0.008 else (
    "WEAK" if best_mae - baseline_mae < -0.003 else "NO_GO"))

# generate submission if helpful
if best_w > 0 and best_mae < baseline_mae:
    blend_test = (1 - best_w) * mega33_test + best_w * retr_test
    test_raw = pd.read_csv("test.csv")
    pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_retr_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission saved: {OUT}/submission_mega33_retr_w{int(best_w*100)}.csv")

summary = {
    "retr_oof_mae": float(retr_mae),
    "mega33_baseline": float(baseline_mae),
    "residual_corr": corr,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(best_mae - baseline_mae),
    "verdict": verdict,
    "fold_rows": fold_rows,
    "blend_scan": scan,
    "elapsed": round(time.time() - t0, 1),
}
with open(f"{OUT}/retrieval_summary.json", "w") as fp:
    json.dump(summary, fp, indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}")
print(f"  corr={corr:.4f}  best_delta={best_mae - baseline_mae:+.5f}  best_w={best_w}")
print(f"  elapsed: {time.time() - t0:.0f}s")
print(f"{'='*64}")
