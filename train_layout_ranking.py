"""
Layout-level ranking: rank within LAYOUT (instead of scenario).

orig_rank: groups of 25 (within scenario) - very uniform within
layout-rank: groups of ~1000 (within layout, across scenarios) - heterogeneous

Different ranking task → potentially different error pattern from orig_rank.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/layout_ranking"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Layout-level Ranking")
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

# Layout-level group key
layout_key_train = train_fe["layout_id"].astype(str).values
layout_key_test = test_fe["layout_id"].astype(str).values

# For scenario key (needed for rank_adjust later)
sc_key_train = (train_fe["layout_id"].astype(str) + "_" + train_fe["scenario_id"].astype(str)).values
sc_key_test = (test_fe["layout_id"].astype(str) + "_" + test_fe["scenario_id"].astype(str)).values

# Rank WITHIN LAYOUT (percentile, since layouts have 400-1000 rows)
print("\n[1] Building layout-level rank target...")
tmp = pd.DataFrame({"lid": layout_key_train, "y": y})
rank_pctl = tmp.groupby("lid", sort=False)["y"].rank(pct=True).values
# binned into 10 relevance levels for lambdarank
rel = np.floor(rank_pctl * 10).clip(0, 9).astype(np.int32)
print(f"  rel dist: {np.bincount(rel)}")

del blob, train_fe, test_fe
gc.collect()


def _gs_layout(idx):
    """Group sizes for layout grouping."""
    sizes = []; cnt = 0; prev = None
    for i in idx:
        k = layout_key_train[i]
        if k != prev:
            if cnt > 0: sizes.append(cnt)
            cnt = 1; prev = k
        else: cnt += 1
    if cnt > 0: sizes.append(cnt)
    return sizes


def rank_adjust_scenario(scores, mega_preds, sc_keys):
    """Rank-adjust within scenario (same as orig_rank, using layout rank scores)."""
    out = mega_preds.copy().astype(np.float64)
    df = pd.DataFrame({"sc": sc_keys, "rs": scores, "m": mega_preds, "i": np.arange(len(mega_preds))})
    for sc, g in df.groupby("sc", sort=False):
        if len(g) < 2: continue
        sorted_m = np.sort(g["m"].values)
        order = np.argsort(g["rs"].values)
        new = np.empty(len(g))
        for pos, idx_ in enumerate(order):
            new[idx_] = sorted_m[pos]
        out[g["i"].values] = new
    return out


print("\n[2] Training layout-lambdarank 5-fold...")
# For layout-grouping to work with GroupKFold(layout_id), train fold contains all rows of
# layouts != f, val fold has layouts == f. So train groups are complete layouts.
PARAMS = dict(
    objective="lambdarank", metric="ndcg",
    n_estimators=1500, learning_rate=0.05,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
    label_gain=list(range(10)),
)

# Sort train by layout_id to make group sizes continuous
# The data is already sorted by layout_id since we sorted by (layout_id, scenario_id)
rank_raw_oof = np.zeros(len(y))
rank_raw_test = np.zeros(len(X_te))

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    tr_sizes = _gs_layout(tr)
    val_sizes = _gs_layout(val)
    print(f"  fold {f}: n_tr_layouts={len(tr_sizes)} n_val_layouts={len(val_sizes)} "
          f"avg_layout_size={np.mean(tr_sizes):.0f}")
    m = lgb.LGBMRanker(**PARAMS)
    m.fit(X_tr[tr], rel[tr], group=tr_sizes,
          eval_set=[(X_tr[val], rel[val])], eval_group=[val_sizes],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    rank_raw_oof[val] = m.predict(X_tr[val])
    rank_raw_test += m.predict(X_te) / 5
    print(f"    best_iter={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

# The layout-level rank scores indicate cross-scenario ordering within layout.
# Apply rank_adjust within SCENARIO using these scores → this is the key combination
print("\n[3] Rank-adjust within scenario using layout-level rank scores...")
lr_oof = rank_adjust_scenario(rank_raw_oof, mega33_oof, sc_key_train)
lr_test = rank_adjust_scenario(rank_raw_test, mega33_test, sc_key_test)

lr_mae = mean_absolute_error(y, lr_oof)
corr_mega = float(np.corrcoef(y - mega33_oof, y - lr_oof)[0, 1])
print(f"layout_rank single_mae: {lr_mae:.5f}")
print(f"residual_corr(mega33, layout_rank): {corr_mega:.4f}")

orig_rank = np.load("results/ranking/rank_adj_oof.npy")
corr_orig = float(np.corrcoef(y - orig_rank, y - lr_oof)[0, 1])
print(f"residual_corr(orig_rank, layout_rank): {corr_orig:.4f}")

print("\n[4] Blend with mega33:")
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1-w)*mega33_oof + w*lr_oof
    mae = float(mean_absolute_error(y, b))
    d = mae - baseline
    tag = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"  w={w:.2f}: mae={mae:.5f} delta={d:+.5f}{tag}")
    if mae < best_mae: best_w, best_mae = w, mae

np.save(f"{OUT}/layout_rank_oof.npy", lr_oof)
np.save(f"{OUT}/layout_rank_test.npy", lr_test)

# --- Full mega-blend v8 with layout_rank ---
print("\n[5] Full mega-blend v8...")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", orig_rank, np.load("results/ranking/rank_adj_test.npy")),
    ("layout_rank", lr_oof, lr_test),
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
for i in range(1, n): x0[i] = 0.25 / (n-1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol":1e-7, "maxiter":100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline

print(f"\nFinal mega-blend v8 weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:15s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2: -0.005434")
print(f"Improvement: {delta - (-0.005434):+.6f}")

blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v8.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v8.csv")

json.dump({
    "baseline": float(baseline),
    "layout_rank_oof_mae": float(lr_mae),
    "residual_corr_mega33": corr_mega,
    "residual_corr_orig_rank": corr_orig,
    "best_single_blend_w": float(best_w),
    "best_single_blend_delta": float(best_mae - baseline),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/layout_ranking_summary.json","w"), indent=2)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
