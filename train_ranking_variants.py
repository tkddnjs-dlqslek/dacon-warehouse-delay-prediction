"""
Ranking variants ensemble - extend what worked.

6 variants:
- 3 seeds (42, 123, 2024)
- 2 objectives (lambdarank, rank_xendcg)
= 6 ranking models, each 5-fold

Average their rank-adjusted OOF/test predictions -> 'rank_ens' new base.
Hopefully corr with mega33 drops below 0.97 (more diversity than single rank).
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
import lightgbm as lgb

OUT = "results/ranking_variants"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Ranking Variants Ensemble (6 variants)")
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
baseline = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline:.5f}")

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

scenario_key = (train_fe["layout_id"].astype(str) + "_" + train_fe["scenario_id"].astype(str)).values
scenario_key_te = (test_fe["layout_id"].astype(str) + "_" + test_fe["scenario_id"].astype(str)).values

grp = train_fe.groupby(
    (train_fe["layout_id"].astype(str) + "_" + train_fe["scenario_id"].astype(str)),
    sort=False,
)
rank_within = grp[TARGET].rank(method="average", ascending=True).values
rel = (rank_within - 1).astype(np.int32)

del blob, train_fe, test_fe
gc.collect()


def _group_sizes(idx):
    sizes = []
    count = 0
    prev = None
    for i in idx:
        k = scenario_key[i]
        if k != prev:
            if count > 0:
                sizes.append(count)
            count = 1
            prev = k
        else:
            count += 1
    if count > 0:
        sizes.append(count)
    return sizes


def rank_adjust(raw_scores, mega_preds, sc_keys):
    """Within each scenario, redistribute mega predictions by rank scores' ordering."""
    out = mega_preds.copy().astype(np.float64)
    df = pd.DataFrame({"sc": sc_keys, "rs": raw_scores, "m": mega_preds, "i": np.arange(len(mega_preds))})
    for sc, g in df.groupby("sc", sort=False):
        if len(g) < 2:
            continue
        sorted_m = np.sort(g["m"].values)
        order = np.argsort(g["rs"].values)
        new = np.empty(len(g))
        for pos, idx_ in enumerate(order):
            new[idx_] = sorted_m[pos]
        out[g["i"].values] = new
    return out


# 6 variants
variants = []
for seed in [42, 123, 2024]:
    for obj in ["lambdarank", "rank_xendcg"]:
        variants.append((seed, obj))

print(f"\n[Running {len(variants)} variants]")

all_rank_adj_oofs = []
all_rank_adj_tests = []
summary_rows = []

for vi, (seed, obj) in enumerate(variants):
    tag = f"v{vi}_s{seed}_{obj}"
    tv = time.time()
    print(f"\n--- variant {tag} ---")

    PARAMS = dict(
        objective=obj, metric="ndcg",
        n_estimators=1500, learning_rate=0.05,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=seed, verbose=-1, n_jobs=-1,
        label_gain=list(range(25)),
    )

    rank_oof_raw = np.zeros(len(y))
    rank_test_raw = np.zeros(len(X_te))

    for f in range(5):
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]
        tr_sizes = _group_sizes(tr)
        val_sizes = _group_sizes(val)
        m = lgb.LGBMRanker(**PARAMS)
        m.fit(X_tr[tr], rel[tr], group=tr_sizes,
              eval_set=[(X_tr[val], rel[val])],
              eval_group=[val_sizes],
              callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(0)])
        rank_oof_raw[val] = m.predict(X_tr[val])
        rank_test_raw += m.predict(X_te) / 5
        del m

    rank_adj_oof = rank_adjust(rank_oof_raw, mega33_oof, scenario_key)
    rank_adj_test = rank_adjust(rank_test_raw, mega33_test, scenario_key_te)
    all_rank_adj_oofs.append(rank_adj_oof)
    all_rank_adj_tests.append(rank_adj_test)

    mae_v = mean_absolute_error(y, rank_adj_oof)
    corr_v = float(np.corrcoef(y - mega33_oof, y - rank_adj_oof)[0, 1])
    best_w, best_mae = 0, float(baseline)
    for w in [0.05, 0.10, 0.15, 0.20]:
        b = (1 - w) * mega33_oof + w * rank_adj_oof
        mae = float(mean_absolute_error(y, b))
        if mae < best_mae: best_w, best_mae = w, mae

    delta = best_mae - baseline
    print(f"  {tag}: single_mae={mae_v:.5f} corr={corr_v:.4f} best_blend_delta={delta:+.5f} ({time.time()-tv:.0f}s)")
    summary_rows.append({"variant": tag, "seed": seed, "obj": obj,
                         "single_mae": float(mae_v), "corr": corr_v,
                         "best_w": float(best_w), "best_delta": float(delta)})
    np.save(f"{OUT}/rank_adj_oof_{tag}.npy", rank_adj_oof)
    np.save(f"{OUT}/rank_adj_test_{tag}.npy", rank_adj_test)

# --- Ensemble: average the 6 rank-adjusted predictions ---
print(f"\n[Ensemble] Averaging {len(variants)} variants...")
rank_ens_oof = np.mean(all_rank_adj_oofs, axis=0)
rank_ens_test = np.mean(all_rank_adj_tests, axis=0)
ens_mae = mean_absolute_error(y, rank_ens_oof)
ens_corr = float(np.corrcoef(y - mega33_oof, y - rank_ens_oof)[0, 1])
print(f"  ensemble single_mae: {ens_mae:.5f}")
print(f"  ensemble corr vs mega33: {ens_corr:.4f}")

# blend scan
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1 - w) * mega33_oof + w * rank_ens_oof
    mae = float(mean_absolute_error(y, b))
    d = mae - baseline
    tag_d = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"    blend w={w:.2f}: mae={mae:.5f} delta={d:+.5f}{tag_d}")
    if mae < best_mae:
        best_w, best_mae = w, mae

delta = best_mae - baseline
np.save(f"{OUT}/rank_ens_oof.npy", rank_ens_oof)
np.save(f"{OUT}/rank_ens_test.npy", rank_ens_test)

print(f"\n>>> Ensemble best blend: w={best_w} delta={delta:+.5f} <<<")

# --- FULL MULTI-BLEND with ranking ensemble as new model ---
print(f"\n[Full multi-blend with all models]")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_ens", rank_ens_oof, rank_ens_test),
    ("rank_adj_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("cluster", np.load("results/cluster_spec/cluster_oof.npy"),
     np.load("results/cluster_spec/cluster_test.npy")),
    ("iter_r1", np.load("results/iter_pseudo/round1_oof.npy"),
     np.load("results/iter_pseudo/round1_test.npy")),
    ("iter_r2", np.load("results/iter_pseudo/round2_oof.npy"),
     np.load("results/iter_pseudo/round2_test.npy")),
    ("iter_r3", np.load("results/iter_pseudo/round3_oof.npy"),
     np.load("results/iter_pseudo/round3_test.npy")),
]

names = [s[0] for s in sources]
oofs_M = np.column_stack([s[1] for s in sources])
tests_M = np.column_stack([s[2] for s in sources])
n = len(sources)

def obj_fn(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs_M @ w)

x0 = np.zeros(n); x0[0] = 0.75
for i in range(1, n): x0[i] = 0.25 / (n - 1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta_full = res.fun - baseline

print(f"\nFinal multi-blend weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:15s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta_full:+.6f}")

# Generate submission with CORRECT ID alignment (sorted order)
blend_test = tests_M @ w_opt
pred_df = pd.DataFrame({"ID": pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values,
                         TARGET: np.clip(blend_test, 0, None)})
sub_path = f"{OUT}/submission_rank_ens_final.csv"
pred_df.to_csv(sub_path, index=False)

# verify
sorted_ids_check = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
from_fe = pd.read_pickle("results/eda_v30/v30_test_fe_cache.pkl")["ID"].values
print(f"ID alignment check: sorted==fe? {np.array_equal(sorted_ids_check, from_fe)}")

print(f"\nSubmission saved: {sub_path}")

summary = {
    "baseline": float(baseline),
    "n_variants": len(variants),
    "variant_summary": summary_rows,
    "ensemble_single_mae": float(ens_mae),
    "ensemble_corr": ens_corr,
    "ensemble_best_w": float(best_w),
    "ensemble_best_delta": float(delta),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta_full),
    "elapsed_min": round((time.time()-t0)/60, 1),
}
json.dump(summary, open(f"{OUT}/ranking_variants_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL blend delta: {delta_full:+.6f}")
print(f"Previous (single rank + iter): -0.005434")
print(f"Improvement: {delta_full - (-0.005434):+.6f}")
print(f"elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
