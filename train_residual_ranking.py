"""
Residual Ranking: rank (y - mega33_oof) within scenario.

Key difference from original ranking:
- Original: rank y within scenario -> model learns absolute delay ranks
- Residual: rank (y - mega33_oof) -> model learns WHERE mega33 makes ordering errors

Expected: different residual pattern than both mega33 and orig_rank.
Target: corr(mega33, res_rank) < 0.95.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/residual_ranking"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Residual Ranking")
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

# KEY: residual as ranking target
residual = y - mega33_oof
print(f"mega33 baseline: {baseline:.5f}")
print(f"residual: mean={residual.mean():+.3f} std={residual.std():.3f}")

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

scenario_key = (train_fe["layout_id"].astype(str)+"_"+train_fe["scenario_id"].astype(str)).values
scenario_key_te = (test_fe["layout_id"].astype(str)+"_"+test_fe["scenario_id"].astype(str)).values

# rank RESIDUAL within scenario
grp_key = (train_fe["layout_id"].astype(str)+"_"+train_fe["scenario_id"].astype(str))
tmp = pd.DataFrame({"sc": grp_key.values, "res": residual})
rank_within = tmp.groupby("sc", sort=False)["res"].rank(method="average", ascending=True).values
rel = (rank_within - 1).astype(np.int32)
print(f"  residual ranks: min={rel.min()} max={rel.max()}")

del blob, train_fe, test_fe
gc.collect()


def _gs(idx):
    sizes = []; cnt = 0; prev = None
    for i in idx:
        k = scenario_key[i]
        if k != prev:
            if cnt > 0: sizes.append(cnt)
            cnt = 1; prev = k
        else: cnt += 1
    if cnt > 0: sizes.append(cnt)
    return sizes


def rank_adjust(scores, mega_preds, sc_keys):
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


# Train LGB-lambdarank on residual ranks
print("\n[1] Training LGB-lambdarank on RESIDUAL ranks, 5-fold...")
PARAMS = dict(
    objective="lambdarank", metric="ndcg",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
    label_gain=list(range(25)),
)

rank_raw_oof = np.zeros(len(y))
rank_raw_test = np.zeros(len(X_te))
for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]; val = np.where(fold_ids == f)[0]
    trs = _gs(tr); vals = _gs(val)
    m = lgb.LGBMRanker(**PARAMS)
    m.fit(X_tr[tr], rel[tr], group=trs,
          eval_set=[(X_tr[val], rel[val])], eval_group=[vals],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    rank_raw_oof[val] = m.predict(X_tr[val])
    rank_raw_test += m.predict(X_te) / 5
    print(f"  fold {f}: best_iter={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

print("\n[2] Rank-adjusting mega33 by residual-rank scores...")
res_rank_oof = rank_adjust(rank_raw_oof, mega33_oof, scenario_key)
res_rank_test = rank_adjust(rank_raw_test, mega33_test, scenario_key_te)

rr_mae = mean_absolute_error(y, res_rank_oof)
corr_mega = float(np.corrcoef(y - mega33_oof, y - res_rank_oof)[0, 1])
print(f"res_rank single_mae: {rr_mae:.5f}")
print(f"residual_corr(mega33, res_rank): {corr_mega:.4f}")

# compare to orig_rank
orig_rank = np.load("results/ranking/rank_adj_oof.npy")
corr_orig = float(np.corrcoef(y - orig_rank, y - res_rank_oof)[0, 1])
print(f"residual_corr(orig_rank, res_rank): {corr_orig:.4f}")

print("\n[3] Blend with mega33:")
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1-w)*mega33_oof + w*res_rank_oof
    mae = float(mean_absolute_error(y, b))
    d = mae - baseline
    tag = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"  w={w:.2f}: mae={mae:.5f} delta={d:+.5f}{tag}")
    if mae < best_mae: best_w, best_mae = w, mae

np.save(f"{OUT}/res_rank_oof.npy", res_rank_oof)
np.save(f"{OUT}/res_rank_test.npy", res_rank_test)

# Full mega-blend v4 with res_rank
print("\n[4] Full mega-blend v4 (adding res_rank)...")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("res_rank", res_rank_oof, res_rank_test),
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
for i in range(1, n): x0[i] = 0.25 / (n-1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol":1e-7, "maxiter":100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline

print(f"\nFinal mega-blend v4 weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:12s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2: -0.005434")
print(f"Improvement from v2: {delta - (-0.005434):+.6f}")

# submission
blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v4.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v4.csv")

json.dump({
    "baseline": float(baseline),
    "res_rank_oof_mae": float(rr_mae),
    "residual_corr_mega33": corr_mega,
    "residual_corr_orig_rank": corr_orig,
    "best_single_blend_w": float(best_w),
    "best_single_blend_delta": float(best_mae - baseline),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/residual_ranking_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
