"""
Mega Multi-blend: optimize weights over ALL available OOFs.

Inputs:
- mega33 meta_avg_oof (baseline)
- pseudo all_w1.0 (from earlier)
- adversarial weighted
- V_ORTH
- retrieval
- queuing
- TabNet (Colab, if downloaded)
- SSL-TabNet (Colab, if downloaded)

Output: optimized blend OOF + delta vs baseline.
If delta < -0.003, generate submission CSV.
"""
import pickle, numpy as np, pandas as pd, json, os
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore")

TARGET = "avg_delay_minutes_next_30m"

train = pd.read_csv("train.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
y = train[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

# load all available OOFs
sources = [
    ("pseudo_all_w1.0", "results/v24_cumsum/oof_pseudo_all_w1.0.npy", None),
    ("adversarial", "results/v24_cumsum/oof_adv_weighted.npy", None),
    ("retrieval", "results/retrieval/retr_oof.npy", "results/retrieval/retr_test.npy"),
    ("vorth", "results/vorth/vorth_oof.npy", "results/vorth/vorth_test.npy"),
    ("queuing", "results/queuing/queue_oof.npy", "results/queuing/queue_test.npy"),
]

models = {"mega33": (mega33_oof, mega33_test)}
for name, oof_f, test_f in sources:
    if os.path.exists(oof_f):
        oof = np.load(oof_f)
        tst = np.load(test_f) if test_f and os.path.exists(test_f) else None
        models[name] = (oof, tst)
        mae = mean_absolute_error(y, oof)
        r_mega = y - mega33_oof
        r_new = y - oof
        corr = float(np.corrcoef(r_mega, r_new)[0, 1])
        print(f"  loaded {name}: OOF={mae:.5f} corr={corr:.4f}")
    else:
        print(f"  missing {name} ({oof_f})")

print(f"\nn_models: {len(models)}")

# --- optimize weights on OOF ---
names = list(models.keys())
oofs = np.column_stack([models[n][0] for n in names])
n = len(names)

def obj(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs @ w)

# Start with mega33 dominant
x0 = np.zeros(n)
x0[0] = 0.7
for i in range(1, n): x0[i] = 0.3 / (n-1)
res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 50000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline_mae

print(f"\n=== Multi-blend optimal weights ===")
for n_, w_ in zip(names, w_opt):
    print(f"  {n_:20s}: {w_:.4f}")
print(f"\nOOF mae: {res.fun:.5f}  delta vs mega33: {delta:+.6f}")

# submission (if helpful and we have test preds)
missing_test = [n for n in names if models[n][1] is None]
if missing_test:
    print(f"\nCannot generate full submission (missing test preds): {missing_test}")
    # generate reduced blend using only models with test preds
    valid_names = [n for n in names if models[n][1] is not None]
    valid_oofs = np.column_stack([models[n][0] for n in valid_names])
    def obj2(w):
        w = np.clip(w, 0, None)
        if w.sum() < 1e-6: return 99
        w = w / w.sum()
        return mean_absolute_error(y, valid_oofs @ w)
    x0_v = np.zeros(len(valid_names))
    x0_v[0] = 0.7
    for i in range(1, len(valid_names)): x0_v[i] = 0.3 / (len(valid_names)-1)
    res2 = minimize(obj2, x0_v, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 50000})
    w_opt_v = np.clip(res2.x, 0, None); w_opt_v = w_opt_v / w_opt_v.sum()
    print(f"\n=== Multi-blend WITH test preds available ({len(valid_names)} models) ===")
    for n_, w_ in zip(valid_names, w_opt_v):
        print(f"  {n_:20s}: {w_:.4f}")
    print(f"OOF mae: {res2.fun:.5f}  delta: {res2.fun - baseline_mae:+.6f}")

    # generate submission
    if res2.fun < baseline_mae - 0.001:
        tests = np.column_stack([models[n][1] for n in valid_names])
        blend_test = tests @ w_opt_v
        test_raw = pd.read_csv("test.csv")  # raw order
        # Most oofs are sorted by (layout_id, scenario_id); test must be too
        test_sorted = test_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
        test_sorted[TARGET] = np.clip(blend_test, 0, None)
        # reorder back to raw test order
        submission = test_sorted[["ID", TARGET]].set_index("ID").loc[test_raw["ID"].values].reset_index()
        os.makedirs("results/multi_blend", exist_ok=True)
        submission.to_csv("results/multi_blend/submission_multiblend_final.csv", index=False)
        print(f"\nSubmission: results/multi_blend/submission_multiblend_final.csv")

# save summary
os.makedirs("results/multi_blend", exist_ok=True)
json.dump({
    "baseline_mae": float(baseline_mae),
    "models": names,
    "weights": w_opt.tolist(),
    "blend_mae": float(res.fun),
    "delta": float(delta),
}, open("results/multi_blend/multi_blend_summary.json", "w"), indent=2)

print(f"\nDone.")
