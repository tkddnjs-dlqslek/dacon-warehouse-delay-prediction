"""
Iterative Pseudo-labeling with quality filtering.

3 rounds:
  Round 1: pseudo = mega33 test predictions, full test used
  Round 2: pseudo = round1 predictions, use only STABLE test rows (|r1 - mega33| < threshold)
  Round 3: pseudo = round2 predictions, tighter stability filter

Each round saves OOF + test predictions for final blending.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/iter_pseudo"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Iterative Pseudo-labeling")
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

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
del blob, train_fe, test_fe
gc.collect()
print(f"X_tr={X_tr.shape} X_te={X_te.shape} loaded={time.time()-t0:.0f}s")

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=4,
)


def run_round(pseudo_test_preds, pseudo_mask, round_name, pseudo_weight=1.0):
    """
    Train LGB with train + pseudo_test[pseudo_mask].
    Returns (oof, test_preds, fold_metrics).
    """
    print(f"\n=== {round_name} ===")
    print(f"  pseudo rows used: {pseudo_mask.sum()} / {len(pseudo_test_preds)} "
          f"({100*pseudo_mask.mean():.1f}%)")

    X_te_p = X_te[pseudo_mask]
    pseudo_y_log = np.log1p(np.clip(pseudo_test_preds[pseudo_mask], 0, None)).astype(np.float32)

    oof = np.zeros(len(y))
    test_pred = np.zeros(len(X_te))
    rows = []
    t_round = time.time()

    for f in range(5):
        tf = time.time()
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]

        # combine train + pseudo test
        X_comb = np.vstack([X_tr[tr], X_te_p])
        y_comb = np.concatenate([y_log[tr], pseudo_y_log])
        w_comb = np.concatenate([np.ones(len(tr)), np.full(len(X_te_p), pseudo_weight)])

        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_comb, y_comb, sample_weight=w_comb,
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

        oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
        test_pred += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

        fm = mean_absolute_error(y[val], oof[val])
        rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
        print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tf:.0f}s)")
        del X_comb, y_comb, w_comb, m
        gc.collect()

    mae = mean_absolute_error(y, oof)
    r_mega = y - mega33_oof
    r_cur = y - oof
    corr = float(np.corrcoef(r_mega, r_cur)[0, 1])

    best_w, best_mae = 0, float(baseline_mae)
    for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        blend = (1-w) * mega33_oof + w * oof
        bmae = float(mean_absolute_error(y, blend))
        if bmae < best_mae: best_w, best_mae = w, bmae

    delta = best_mae - baseline_mae

    print(f"  OOF mae: {mae:.5f}")
    print(f"  residual_corr(mega33): {corr:.4f}")
    print(f"  best blend: w={best_w} mae={best_mae:.5f} delta={delta:+.5f}")
    print(f"  elapsed: {time.time()-t_round:.0f}s")

    np.save(f"{OUT}/{round_name}_oof.npy", oof)
    np.save(f"{OUT}/{round_name}_test.npy", test_pred)
    return oof, test_pred, {"oof_mae": float(mae), "corr": corr, "best_w": float(best_w),
                             "best_mae": float(best_mae), "delta": float(delta), "fold_rows": rows}


# === Round 1: full pseudo = mega33 test ===
full_mask = np.ones(len(X_te), dtype=bool)
oof_r1, test_r1, stats_r1 = run_round(mega33_test, full_mask, "round1", pseudo_weight=1.0)

# === Round 2: pseudo = round1 test preds, filter stable (agree with mega33) ===
# Quality: keep rows where |round1 - mega33| is in LOWER 80% (exclude top 20% most unstable)
diff_r1_mega = np.abs(test_r1 - mega33_test)
thresh2 = np.percentile(diff_r1_mega, 80)
stable_mask_r2 = diff_r1_mega <= thresh2
print(f"\nRound 2 stability filter threshold: |diff| <= {thresh2:.3f}")
print(f"  rows passing: {stable_mask_r2.sum()} ({100*stable_mask_r2.mean():.1f}%)")
# use average of round1 and mega33 as pseudo (more robust)
pseudo_r2 = (test_r1 + mega33_test) / 2
oof_r2, test_r2, stats_r2 = run_round(pseudo_r2, stable_mask_r2, "round2", pseudo_weight=1.0)

# === Round 3: pseudo = round2 test, tighter filter ===
# Filter: keep rows where ALL three (r1, r2, mega33) agree (low disagreement)
preds_stack = np.stack([mega33_test, test_r1, test_r2], axis=1)
std_preds = preds_stack.std(axis=1)
thresh3 = np.percentile(std_preds, 70)
stable_mask_r3 = std_preds <= thresh3
print(f"\nRound 3 stability filter threshold: std <= {thresh3:.3f}")
print(f"  rows passing: {stable_mask_r3.sum()} ({100*stable_mask_r3.mean():.1f}%)")
# use ensemble of all 3 previous as pseudo
pseudo_r3 = preds_stack.mean(axis=1)
oof_r3, test_r3, stats_r3 = run_round(pseudo_r3, stable_mask_r3, "round3", pseudo_weight=1.0)

# === Final multi-blend of all rounds + mega33 ===
print(f"\n{'='*64}")
print(f"All rounds done. Multi-blend analysis:")
print(f"{'='*64}")

from scipy.optimize import minimize
all_oofs = np.column_stack([mega33_oof, oof_r1, oof_r2, oof_r3])
all_tests = np.column_stack([mega33_test, test_r1, test_r2, test_r3])
names = ["mega33", "round1", "round2", "round3"]

def obj(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, all_oofs @ w)

x0 = [0.8, 0.1, 0.05, 0.05]
res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 50000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline_mae

print(f"\nOptimal weights:")
for n_, w_ in zip(names, w_opt):
    print(f"  {n_:10s}: {w_:.4f}")
print(f"\nBlend OOF: {res.fun:.5f}  delta: {delta:+.6f}")

# Submission if helpful
verdict = "STRONG_GO" if delta < -0.01 else ("GO" if delta < -0.005 else ("WEAK" if delta < -0.002 else "NO_GO"))

if delta < -0.002:
    blend_test = all_tests @ w_opt
    test_raw = pd.read_csv("test.csv")
    # OOFs are in layout_id+scenario_id sort order; test also from that order
    pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
        f"{OUT}/submission_iter_pseudo_blend.csv", index=False)
    print(f"\nSubmission: {OUT}/submission_iter_pseudo_blend.csv")

summary = {
    "baseline_mae": float(baseline_mae),
    "round1": stats_r1,
    "round2": stats_r2,
    "round3": stats_r3,
    "final_blend_weights": dict(zip(names, w_opt.tolist())),
    "final_blend_mae": float(res.fun),
    "final_delta": float(delta),
    "verdict": verdict,
    "total_elapsed": round(time.time()-t0, 1),
}
json.dump(summary, open(f"{OUT}/iter_pseudo_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL VERDICT: {verdict}")
print(f"  delta: {delta:+.6f}")
print(f"  total elapsed: {time.time()-t0:.0f}s")
print(f"{'='*64}")
