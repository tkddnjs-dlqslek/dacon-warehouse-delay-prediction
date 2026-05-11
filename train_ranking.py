"""
Ranking-based learning (B4): LightGBM lambdarank within scenario groups.

Each scenario (25 rows) = 1 group. Model learns to rank rows by delay.
Then combine rank predictions with mega33 for diverse blend.

This is a fundamentally different loss landscape than MAE/Huber.
Ranking-based predictions have different residual patterns.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb

OUT = "results/ranking"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Ranking Loss Experiment")
print("=" * 64)
t0 = time.time()

with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

# For lambdarank, we need:
# - group information (size of each group)
# - target: higher value = higher rank relevance
# lambdarank wants integer relevance; y is continuous so we use qcut to bins

# Build scenario groups
scenario_key = (train_fe["layout_id"].astype(str) + "_" + train_fe["scenario_id"].astype(str)).values
scenario_key_te = (test_fe["layout_id"].astype(str) + "_" + test_fe["scenario_id"].astype(str)).values

# For lambdarank: group by scenario, relevance = y binned (higher y = higher rank = more "relevant")
# lambdarank needs int relevance [0, max_rel]
# Use percentile ranking within scenario, then convert to ints 0-24
print("\n[1] Building rank-based target within scenarios...")
train_fe["_sc"] = scenario_key
grp = train_fe.groupby("_sc", sort=False)
rank_within = grp[TARGET].rank(method="average", ascending=True).values  # 1..25
# convert to integer relevance [0..24]
rel = (rank_within - 1).astype(np.int32)
# Group sizes (sorted by scenario order)
# The data is already sorted by (layout_id, scenario_id) + timeslot
# Verify: consecutive same scenario_key
group_sizes = [25] * (len(y) // 25)  # should be uniform 25
assert sum(group_sizes) == len(y), f"non-uniform groups: {sum(group_sizes)} vs {len(y)}"

print(f"  rel values: min={rel.min()} max={rel.max()}")
print(f"  n_groups: {len(group_sizes)} (each size 25)")

# Train lambdarank per fold
print("\n[2] Training LGB-lambdarank 5-fold...")
PARAMS_RANK = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=2000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    min_child_samples=50,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
    label_gain=list(range(25)),  # linear gain
)

rank_oof_raw = np.zeros(len(y))
rank_test_raw = np.zeros(len(X_te))
fold_rows = []

# group sizes per fold
def _group_sizes_for_idx(idx):
    # assume idx elements come in contiguous scenario blocks of 25
    # so just return [25]*(len(idx)//25)
    sizes = []
    count = 0
    prev = None
    for i in idx:
        key = scenario_key[i]
        if key != prev:
            if count > 0:
                sizes.append(count)
            count = 1
            prev = key
        else:
            count += 1
    if count > 0:
        sizes.append(count)
    return sizes

for f in range(5):
    tf = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]

    tr_sizes = _group_sizes_for_idx(tr)
    val_sizes = _group_sizes_for_idx(val)
    print(f"  fold {f}: n_tr_groups={len(tr_sizes)} n_val_groups={len(val_sizes)}")

    model = lgb.LGBMRanker(**PARAMS_RANK)
    model.fit(
        X_tr[tr], rel[tr],
        group=tr_sizes,
        eval_set=[(X_tr[val], rel[val])],
        eval_group=[val_sizes],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    pred_val = model.predict(X_tr[val])   # raw scores
    pred_test = model.predict(X_te)
    rank_oof_raw[val] = pred_val
    rank_test_raw += pred_test / 5

    # quick fold-level metric: within-scenario Spearman of predicted rank vs true y
    # per scenario
    val_scenarios = np.unique(scenario_key[val])
    rhos = []
    for sc in val_scenarios[:200]:  # sample for speed
        mask = scenario_key[val] == sc
        if mask.sum() >= 10:
            r = spearmanr(pred_val[mask], y[val][mask]).statistic
            if not np.isnan(r):
                rhos.append(r)
    mean_spearman = np.mean(rhos)
    fold_rows.append({"fold": f, "best_iter": int(model.best_iteration_ or 2000),
                      "spearman_pred_y": float(mean_spearman),
                      "time_s": round(time.time()-tf, 1)})
    print(f"    within-scenario Spearman(pred, y): {mean_spearman:.4f} ({time.time()-tf:.0f}s)")
    del model

print(f"\n[3] Converting rank predictions to regression prediction (blend with mega33)...")
# rank_oof_raw is in rank-score scale, not delay scale. To use for stacking,
# we should see if residuals are different from mega33.
# But rank prediction can't directly be used as a delay prediction.
# Instead, we use it as a FEATURE when combined with mega33.

# Alternative: within each scenario, use ranked predictions to REORDER mega33 predictions
# Method: within scenario, compute percentile rank from rank_oof_raw, then use to adjust mega33
# This is a "rank-guided adjustment"
scenario_to_mega33 = {}
print("  building rank-adjusted predictions within scenarios...")
rank_adj_oof = mega33_oof.copy()
rank_adj_test = mega33_test.copy()

# Within each scenario, replace mega33 ranks with ranking model's ranks while preserving mean
def rank_adjust(raw_rank_scores, mega_preds, scenario_keys):
    """Within each scenario, redistribute mega33 predictions by rank model's ordering."""
    out = mega_preds.copy()
    df = pd.DataFrame({
        "sc": scenario_keys,
        "rank_score": raw_rank_scores,
        "mega": mega_preds,
        "idx": np.arange(len(mega_preds)),
    })
    for sc, grp_df in df.groupby("sc", sort=False):
        if len(grp_df) < 2:
            continue
        # sort mega values ascending (sorted_mega)
        sorted_mega = np.sort(grp_df["mega"].values)
        # rank of rank_score ascending
        rank_order = np.argsort(grp_df["rank_score"].values)
        # assign sorted_mega in rank_order positions
        new_vals = np.empty(len(grp_df))
        for pos, original_idx in enumerate(rank_order):
            new_vals[original_idx] = sorted_mega[pos]
        out[grp_df["idx"].values] = new_vals
    return out

print("  adjusting OOF...")
rank_adj_oof = rank_adjust(rank_oof_raw, mega33_oof, scenario_key)
print("  adjusting test...")
rank_adj_test = rank_adjust(rank_test_raw, mega33_test, scenario_key_te)

# compare
adj_mae = mean_absolute_error(y, rank_adj_oof)
print(f"\n  mega33 OOF mae: {baseline_mae:.5f}")
print(f"  rank-adjusted OOF mae: {adj_mae:.5f}  delta: {adj_mae - baseline_mae:+.5f}")

# residual corr
r_mega = y - mega33_oof
r_adj = y - rank_adj_oof
corr_adj = float(np.corrcoef(r_mega, r_adj)[0, 1])
print(f"  residual_corr(mega33, rank_adj): {corr_adj:.4f}")

# blend scan
best_w, best_mae = 0, float(baseline_mae)
for w in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    b = (1 - w) * mega33_oof + w * rank_adj_oof
    m = float(mean_absolute_error(y, b))
    print(f"    blend w={w:.2f}: mae={m:.5f} delta={m - baseline_mae:+.5f}")
    if m < best_mae:
        best_w, best_mae = w, m

delta = best_mae - baseline_mae
verdict = "STRONG_GO" if delta < -0.01 else ("GO" if delta < -0.005 else ("WEAK" if delta < -0.002 else "NO_GO"))

np.save(f"{OUT}/rank_raw_oof.npy", rank_oof_raw)
np.save(f"{OUT}/rank_raw_test.npy", rank_test_raw)
np.save(f"{OUT}/rank_adj_oof.npy", rank_adj_oof)
np.save(f"{OUT}/rank_adj_test.npy", rank_adj_test)

# submission if helpful
if delta < -0.003:
    blend_test = (1 - best_w) * mega33_test + best_w * rank_adj_test
    test_raw = pd.read_csv("test.csv")
    pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_rank_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission: {OUT}/submission_mega33_rank_w{int(best_w*100)}.csv")

json.dump({
    "baseline_mae": float(baseline_mae),
    "rank_adj_oof_mae": float(adj_mae),
    "residual_corr": corr_adj,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(delta),
    "verdict": verdict,
    "fold_rows": fold_rows,
    "elapsed": round(time.time()-t0, 1),
}, open(f"{OUT}/ranking_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}  corr={corr_adj:.4f}  delta={delta:+.5f}")
print(f"elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
