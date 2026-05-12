"""
Task C Part 2: Use new 3-level optimal meta in full FIXED-style multi-blend.
Replace mega33 with meta_avg_oof_3level (from task_c) and re-optimize weights.
"""
import os, sys, pickle, json, numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
TARGET = "avg_delay_minutes_next_30m"

train = (
    pd.read_csv(os.path.join(ROOT, "train.csv"))
    .sort_values(["layout_id", "scenario_id"])
    .reset_index(drop=True)
)
y = train[TARGET].values.astype(np.float64)

# Current FIXED baseline mega33 (simple avg)
with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
    mega = pickle.load(f)
mega_orig_oof = np.asarray(mega["meta_avg_oof"])
mega_orig_test = np.asarray(mega["meta_avg_test"])
print(f"Original mega33 OOF MAE: {mean_absolute_error(y, mega_orig_oof):.5f}", flush=True)

# New 3-level weighted meta (from Task C)
mega_new_oof = np.load(os.path.join(ROOT, "results", "task_c", "meta_avg_oof_3level.npy"))
mega_new_test = np.load(os.path.join(ROOT, "results", "task_c", "meta_avg_test_3level.npy"))
print(f"New 3-level mega33 OOF MAE: {mean_absolute_error(y, mega_new_oof):.5f}", flush=True)

# Other components (same as FIXED)
sources = [
    ("rank_adj", "results/ranking/rank_adj_oof.npy", "results/ranking/rank_adj_test.npy"),
    ("iter_r1", "results/iter_pseudo/round1_oof.npy", "results/iter_pseudo/round1_test.npy"),
    ("iter_r2", "results/iter_pseudo/round2_oof.npy", "results/iter_pseudo/round2_test.npy"),
    ("iter_r3", "results/iter_pseudo/round3_oof.npy", "results/iter_pseudo/round3_test.npy"),
]

print("\nLoading components...", flush=True)
models_old = {"mega33": (mega_orig_oof, mega_orig_test)}
models_new = {"mega33_3level": (mega_new_oof, mega_new_test)}

for name, oof_f, test_f in sources:
    oof = np.load(os.path.join(ROOT, oof_f))
    tst = np.load(os.path.join(ROOT, test_f))
    m = mean_absolute_error(y, oof)
    c_old = float(np.corrcoef(y - mega_orig_oof, y - oof)[0, 1])
    c_new = float(np.corrcoef(y - mega_new_oof, y - oof)[0, 1])
    print(f"  {name}: OOF={m:.5f}  corr(old_mega)={c_old:.4f}  corr(new_mega)={c_new:.4f}", flush=True)
    models_old[name] = (oof, tst)
    models_new[name] = (oof, tst)


def optimize(models, label):
    names = list(models.keys())
    oofs = np.column_stack([models[n][0] for n in names])
    tests = np.column_stack([models[n][1] for n in names])
    n = len(names)

    def obj(w):
        w = np.clip(w, 0, None)
        if w.sum() < 1e-6:
            return 99
        w = w / w.sum()
        return mean_absolute_error(y, oofs @ w)

    x0 = np.zeros(n); x0[0] = 0.85
    for i in range(1, n): x0[i] = 0.15 / (n-1)
    res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 100000})
    w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
    print(f"\n=== {label} ===", flush=True)
    for n_, w_ in zip(names, w_opt):
        if w_ > 0.001:
            print(f"  {n_:15s}: {w_:.4f}", flush=True)
    print(f"  Blend OOF: {res.fun:.5f}", flush=True)
    return res.fun, w_opt, names, oofs, tests


mae_old, w_old, n_old, _, tests_old = optimize(models_old, "FIXED (original mega33)")
mae_new, w_new, n_new, _, tests_new = optimize(models_new, "3-level mega33")

print("\n" + "=" * 60, flush=True)
print(f"FIXED original:  {mae_old:.5f}", flush=True)
print(f"3-level mega33:  {mae_new:.5f}", flush=True)
print(f"Delta:           {mae_new - mae_old:+.5f}", flush=True)

# Decision
if mae_new < mae_old - 0.0005:
    print("\nVERDICT: PROMOTE — build submission_v6_3level.csv", flush=True)
    # Build test prediction
    final_test = tests_new @ w_new
    # Save
    sub_id = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    # Align by test order
    test_raw = pd.read_csv(os.path.join(ROOT, "test.csv"))
    test_sorted = test_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
    assert len(test_sorted) == len(final_test)
    final_df = pd.DataFrame({"ID": test_sorted["ID"].values, TARGET: np.clip(final_test, 0, None)})
    # Merge back to sample_submission order
    sub_out = sub_id[["ID"]].merge(final_df, on="ID", how="left")
    assert sub_out[TARGET].isna().sum() == 0
    out_path = os.path.join(ROOT, "results", "final_blend", "submission_v6_3level.csv")
    sub_out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}", flush=True)
else:
    print("\nVERDICT: NO_GO — 3-level not meaningful in full blend", flush=True)
