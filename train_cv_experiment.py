"""
CV Strategy A/B Test: GroupKFold vs RowKFold vs Full-fit.

Hypothesis: if shared layouts (50/100) in test benefit more from memorization,
RowKFold-trained model could perform better on LB than GroupKFold.

3 variants:
A. GroupKFold(layout_id) - our standard (already done, use mega33's approach)
B. RowKFold(5-fold random) - allows layout memorization
C. Full-fit on all train - pure memorization + generalization mix

All use v23 features + LGB-Huber.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import lightgbm as lgb

OUT = "results/cv_experiment"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("CV Strategy A/B Test")
print("=" * 64)
t0 = time.time()

# Load v30 cache
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)

y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

# Load mega33 for comparison
with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline = mean_absolute_error(y, mega33_oof)

# Existing GroupKFold fold IDs (used in all our prior work)
group_fold_ids = np.load("results/eda_v30/fold_idx.npy")

# Also load layout info for adversarial estimation
train_layouts_set = set(train_fe["layout_id"].unique())
test_layouts_set = set(test_fe["layout_id"].unique())
shared_in_test = len(train_layouts_set & test_layouts_set)
unseen_in_test = len(test_layouts_set - train_layouts_set)
print(f"test layouts: shared={shared_in_test}, unseen={unseen_in_test}")

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)


def train_kfold(fold_ids_arr, name):
    """5-fold training with given fold IDs."""
    print(f"\n=== Training {name} (5-fold) ===")
    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    fold_metrics = []
    for f in range(5):
        tf = time.time()
        tr = np.where(fold_ids_arr != f)[0]
        val = np.where(fold_ids_arr == f)[0]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_tr[tr], y_log[tr],
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
        test_pred += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
        fm = mean_absolute_error(y[val], oof[val])
        fold_metrics.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
        print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tf:.0f}s)")
        del m
    overall_mae = mean_absolute_error(y, oof)
    print(f"  {name} OOF: {overall_mae:.5f}")
    return oof, test_pred, overall_mae, fold_metrics


# --- A. GroupKFold (reuse existing) - train fresh to get test predictions ---
# We don't have direct test preds from v23 GroupKFold only, but we can re-train
print("\n[A] GroupKFold(layout_id) - v23 features, LGB-Huber")
group_oof, group_test, group_mae, group_folds = train_kfold(group_fold_ids, "GroupKFold")
np.save(f"{OUT}/group_oof.npy", group_oof)
np.save(f"{OUT}/group_test.npy", group_test)

# --- B. RowKFold (random 5-fold, layouts can appear in both) ---
print("\n[B] RowKFold(random 5-fold)")
row_kf = KFold(n_splits=5, shuffle=True, random_state=42)
row_fold_ids = np.zeros(len(y), dtype=np.int8)
for fi, (_, val_idx) in enumerate(row_kf.split(X_tr)):
    row_fold_ids[val_idx] = fi
row_oof, row_test, row_mae, row_folds = train_kfold(row_fold_ids, "RowKFold")
np.save(f"{OUT}/row_oof.npy", row_oof)
np.save(f"{OUT}/row_test.npy", row_test)

# --- C. Full-fit (no CV, trains on all 250K) ---
print("\n[C] Full-fit on ALL 250K rows (no holdout)")
tv = time.time()
m_full = lgb.LGBMRegressor(**PARAMS)
# no early stopping since no val set; use fixed n_estimators (median of CV best_iter)
best_its = [f["best_iter"] for f in group_folds]
n_est_use = int(np.median(best_its))
print(f"  using n_estimators={n_est_use} (median of GroupKFold best_iter)")
params_full = PARAMS.copy()
params_full["n_estimators"] = n_est_use
m_full = lgb.LGBMRegressor(**params_full)
m_full.fit(X_tr, y_log)
full_test = np.clip(np.expm1(m_full.predict(X_te)), 0, None)
print(f"  full-fit train time: {time.time()-tv:.0f}s")
np.save(f"{OUT}/full_test.npy", full_test)

# --- Compare predictions ---
print(f"\n{'='*64}")
print("Comparison (single LGB models):")
print(f"{'='*64}")
print(f"  GroupKFold OOF: {group_mae:.5f}")
print(f"  RowKFold   OOF: {row_mae:.5f}  (likely LOWER due to memorization)")
print(f"  mega33 baseline: {baseline:.5f}")

print("\nresidual_corr with mega33 (on train):")
cg = float(np.corrcoef(y - mega33_oof, y - group_oof)[0,1])
cr = float(np.corrcoef(y - mega33_oof, y - row_oof)[0,1])
print(f"  GroupKFold: {cg:.4f}")
print(f"  RowKFold:   {cr:.4f}")
# Note: RowKFold OOF is biased (memorization leak), so corr interpretation is tricky

print("\nTest predictions difference:")
diff_g_r = np.abs(group_test - row_test).mean()
diff_g_mega = np.abs(group_test - mega33_test).mean()
diff_r_mega = np.abs(row_test - mega33_test).mean()
diff_full_mega = np.abs(full_test - mega33_test).mean()
print(f"  |Group - Row|     mean: {diff_g_r:.4f}")
print(f"  |Group - mega33|  mean: {diff_g_mega:.4f}")
print(f"  |Row - mega33|    mean: {diff_r_mega:.4f}")
print(f"  |Full - mega33|   mean: {diff_full_mega:.4f}")

# --- Generate submission CSVs blending with current best (fixed) ---
print("\n--- Generating submissions ---")
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
fixed_test = pd.read_csv("results/final_blend/submission_final_multiblend_FIXED.csv")
fixed_test = fixed_test.set_index("ID").loc[sorted_ids][TARGET].values

# Variant: blend fixed + row_test (RowKFold model, potential memorization)
for name, preds in [("row_lgb", row_test), ("full_lgb", full_test)]:
    for w in [0.05, 0.10, 0.15, 0.20]:
        b = (1-w) * fixed_test + w * preds
        pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(b, 0, None)}).to_csv(
            f"{OUT}/submission_fixed_plus_{name}_{int(w*100)}.csv", index=False)
    # also pure model
    pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(preds, 0, None)}).to_csv(
        f"{OUT}/submission_{name}_pure.csv", index=False)

# Also: blend fixed + group_test as sanity check
for w in [0.05, 0.10]:
    b = (1-w) * fixed_test + w * group_test
    pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(b, 0, None)}).to_csv(
        f"{OUT}/submission_fixed_plus_group_lgb_{int(w*100)}.csv", index=False)

print("\nAll submissions saved.")
print(f"\n{'='*64}")
print(f"Recommendation:")
print(f"  Most likely to improve LB: submission_fixed_plus_row_lgb_10.csv")
print(f"  ← tests whether RowKFold-trained LGB adds signal on shared layouts")
print(f"{'='*64}")

json.dump({
    "group_oof_mae": float(group_mae),
    "row_oof_mae": float(row_mae),
    "mega33_baseline": float(baseline),
    "group_corr_with_mega33": cg,
    "row_corr_with_mega33": cr,
    "diff_group_vs_mega": float(diff_g_mega),
    "diff_row_vs_mega": float(diff_r_mega),
    "diff_full_vs_mega": float(diff_full_mega),
    "full_fit_n_estimators": n_est_use,
    "shared_test_layouts": shared_in_test,
    "unseen_test_layouts": unseen_in_test,
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/cv_experiment_summary.json","w"), indent=2)
