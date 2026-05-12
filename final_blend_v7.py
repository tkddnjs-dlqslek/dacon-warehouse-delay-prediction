"""
Final blend v7: combine Task C (3-level mega) + Task 2 (joint-9q meta) + existing rank/iter.

Candidates:
  mega33_3level (from task_c)   <- new meta_avg (weighted CB+XGB)
  meta_9q       (from task_2)   <- meta LGB over 9 quantiles
  rank_adj      (existing)
  iter_r1/r2/r3 (existing)

Goal: find optimal weights for 6-way blend, produce submission_v7_combined.csv
Compare with FIXED (original) and v6_3level (task_c only).
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
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "final_blend")
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

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
y = train[TARGET].values.astype(np.float64)

# --- Load all components ---
def load(oof_p, test_p):
    return np.load(oof_p), np.load(test_p)

# Original mega33
with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
    mega = pickle.load(f)
mega_orig_oof = mega["meta_avg_oof"]
mega_orig_test = mega["meta_avg_test"]

# Task C: 3-level weighted mega
mega_3l_oof, mega_3l_test = load(
    os.path.join(ROOT, "results", "task_c", "meta_avg_oof_3level.npy"),
    os.path.join(ROOT, "results", "task_c", "meta_avg_test_3level.npy"),
)

# Task 2: joint-9q meta
meta9q_oof, meta9q_test = load(
    os.path.join(ROOT, "results", "task_2_joint", "meta_oof.npy"),
    os.path.join(ROOT, "results", "task_2_joint", "meta_test.npy"),
)

rank_oof, rank_test = load(
    os.path.join(ROOT, "results", "ranking", "rank_adj_oof.npy"),
    os.path.join(ROOT, "results", "ranking", "rank_adj_test.npy"),
)
iter1_oof, iter1_test = load(
    os.path.join(ROOT, "results", "iter_pseudo", "round1_oof.npy"),
    os.path.join(ROOT, "results", "iter_pseudo", "round1_test.npy"),
)
iter2_oof, iter2_test = load(
    os.path.join(ROOT, "results", "iter_pseudo", "round2_oof.npy"),
    os.path.join(ROOT, "results", "iter_pseudo", "round2_test.npy"),
)
iter3_oof, iter3_test = load(
    os.path.join(ROOT, "results", "iter_pseudo", "round3_oof.npy"),
    os.path.join(ROOT, "results", "iter_pseudo", "round3_test.npy"),
)

print("=== Individual MAEs ===", flush=True)
for name, oof in [
    ("mega_orig", mega_orig_oof),
    ("mega_3level", mega_3l_oof),
    ("meta9q", meta9q_oof),
    ("rank_adj", rank_oof),
    ("iter_r1", iter1_oof),
    ("iter_r2", iter2_oof),
    ("iter_r3", iter3_oof),
]:
    m = mean_absolute_error(y, oof)
    c = float(np.corrcoef(y - mega_orig_oof, y - oof)[0, 1]) if name != "mega_orig" else 1.0
    print(f"  {name:15s}: MAE={m:.5f}  corr(mega_orig)={c:.4f}", flush=True)


def optimize_blend(names, oofs_list, tests_list, label):
    n = len(names)
    oofs_mat = np.column_stack(oofs_list)
    tests_mat = np.column_stack(tests_list)

    def obj(w):
        w = np.clip(w, 0, None)
        if w.sum() < 1e-6:
            return 99
        w = w / w.sum()
        return float(np.mean(np.abs(y - oofs_mat @ w)))

    # Initialize: equal or heavy on first (primary mega)
    x0 = np.zeros(n); x0[0] = 0.8
    for i in range(1, n): x0[i] = 0.2 / (n - 1)
    res = minimize(obj, x0, method="Nelder-Mead",
                   options={"xatol": 1e-7, "maxiter": 200000, "fatol": 1e-7})
    w = np.clip(res.x, 0, None); w = w / w.sum()
    print(f"\n=== {label} ===", flush=True)
    for n_, w_ in zip(names, w):
        if w_ > 0.001:
            print(f"  {n_:15s}: {w_:.4f}", flush=True)
    print(f"  Blend OOF: {res.fun:.5f}", flush=True)
    pred_test = tests_mat @ w
    return res.fun, w, pred_test


# 1) FIXED original (mega_orig + rank + iter_r1/r2/r3)
mae_fixed, w_fixed, test_fixed = optimize_blend(
    ["mega_orig", "rank_adj", "iter_r1", "iter_r2", "iter_r3"],
    [mega_orig_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof],
    [mega_orig_test, rank_test, iter1_test, iter2_test, iter3_test],
    "FIXED (original)",
)

# 2) v6 (3level mega + rank + iter)
mae_v6, w_v6, test_v6 = optimize_blend(
    ["mega_3level", "rank_adj", "iter_r1", "iter_r2", "iter_r3"],
    [mega_3l_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof],
    [mega_3l_test, rank_test, iter1_test, iter2_test, iter3_test],
    "v6 (3level)",
)

# 3) v7 combined: 3level mega + meta9q + rank + iter
mae_v7, w_v7, test_v7 = optimize_blend(
    ["mega_3level", "meta9q", "rank_adj", "iter_r1", "iter_r2", "iter_r3"],
    [mega_3l_oof, meta9q_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof],
    [mega_3l_test, meta9q_test, rank_test, iter1_test, iter2_test, iter3_test],
    "v7 combined (3level + meta9q + rank + iter)",
)

# 4) v7b: orig mega + meta9q (no 3level) — comparison
mae_v7b, w_v7b, test_v7b = optimize_blend(
    ["mega_orig", "meta9q", "rank_adj", "iter_r1", "iter_r2", "iter_r3"],
    [mega_orig_oof, meta9q_oof, rank_oof, iter1_oof, iter2_oof, iter3_oof],
    [mega_orig_test, meta9q_test, rank_test, iter1_test, iter2_test, iter3_test],
    "v7b (orig mega + meta9q + rank + iter)",
)

print("\n" + "=" * 60, flush=True)
print("Summary:", flush=True)
print(f"  FIXED:               {mae_fixed:.5f}", flush=True)
print(f"  v6 (3level):         {mae_v6:.5f}  delta {mae_v6-mae_fixed:+.5f}", flush=True)
print(f"  v7 combined:         {mae_v7:.5f}  delta {mae_v7-mae_fixed:+.5f}", flush=True)
print(f"  v7b orig+meta9q:     {mae_v7b:.5f}  delta {mae_v7b-mae_fixed:+.5f}", flush=True)

# Save best submission
def save_submission(test_pred, name):
    test_sorted = test.copy()
    test_sorted[TARGET] = np.clip(test_pred, 0, None)
    sub_sample = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    out = sub_sample[["ID"]].merge(test_sorted[["ID", TARGET]], on="ID", how="left")
    assert out[TARGET].isna().sum() == 0
    path = os.path.join(OUT, name)
    out.to_csv(path, index=False)
    print(f"Saved: {path}", flush=True)

# Always save v7 combined (the best expected)
save_submission(test_v7, "submission_v7_combined.csv")
save_submission(test_v7b, "submission_v7b_orig_meta9q.csv")

summary = dict(
    mae_fixed=mae_fixed, mae_v6=mae_v6, mae_v7=mae_v7, mae_v7b=mae_v7b,
    delta_v7_vs_fixed=mae_v7 - mae_fixed,
    w_v7=dict(zip(["mega_3level", "meta9q", "rank_adj", "iter_r1", "iter_r2", "iter_r3"],
                   w_v7.tolist())),
)
with open(os.path.join(OUT, "v7_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
