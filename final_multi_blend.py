"""
Final multi-blend: mega33 + ranking + cluster + others.

Use all available OOF (+ test where possible) and optimize weights.
Key new additions:
- rank_adj (corr 0.9765, delta -0.00437 single)
- cluster_spec (corr 0.9700, but high single MAE)
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
base = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {base:.5f}")

# Load all with test preds
sources = [
    ("rank_adj", "results/ranking/rank_adj_oof.npy", "results/ranking/rank_adj_test.npy"),
    ("cluster", "results/cluster_spec/cluster_oof.npy", "results/cluster_spec/cluster_test.npy"),
    ("retrieval", "results/retrieval/retr_oof.npy", "results/retrieval/retr_test.npy"),
    ("vorth", "results/vorth/vorth_oof.npy", "results/vorth/vorth_test.npy"),
    ("queuing", "results/queuing/queue_oof.npy", "results/queuing/queue_test.npy"),
    # iter_pseudo
    ("iter_r1", "results/iter_pseudo/round1_oof.npy", "results/iter_pseudo/round1_test.npy"),
    ("iter_r2", "results/iter_pseudo/round2_oof.npy", "results/iter_pseudo/round2_test.npy"),
    ("iter_r3", "results/iter_pseudo/round3_oof.npy", "results/iter_pseudo/round3_test.npy"),
]

models = {"mega33": (mega33_oof, mega33_test)}
for name, oof_f, test_f in sources:
    if os.path.exists(oof_f) and os.path.exists(test_f):
        oof = np.load(oof_f)
        tst = np.load(test_f)
        m = mean_absolute_error(y, oof)
        c = float(np.corrcoef(y - mega33_oof, y - oof)[0, 1])
        print(f"  {name}: OOF={m:.5f} corr={c:.4f}")
        models[name] = (oof, tst)
    else:
        print(f"  {name}: missing")

names = list(models.keys())
oofs = np.column_stack([models[n][0] for n in names])
tests = np.column_stack([models[n][1] for n in names])
n = len(names)
print(f"\nTotal models: {n}")

def obj(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs @ w)

x0 = np.zeros(n); x0[0] = 0.85
for i in range(1, n): x0[i] = 0.15 / (n-1)
res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - base

print(f"\n=== Optimal multi-blend weights ===")
for n_, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {n_:10s}: {w_:.4f}")
print(f"\nBlend OOF: {res.fun:.5f}  delta: {delta:+.6f}")

# Subset blends for comparison
subsets = {
    "m33+rank": ["mega33", "rank_adj"],
    "m33+rank+cluster": ["mega33", "rank_adj", "cluster"],
    "m33+rank+cluster+iter_r2": ["mega33", "rank_adj", "cluster", "iter_r2"],
    "all": names,
}
for sub_name, sub_models in subsets.items():
    valid = [n_ for n_ in sub_models if n_ in models]
    if len(valid) < 2: continue
    sub_oofs = np.column_stack([models[n_][0] for n_ in valid])
    sub_tests = np.column_stack([models[n_][1] for n_ in valid])
    def sub_obj(w):
        w = np.clip(w, 0, None)
        if w.sum() < 1e-6: return 99
        w = w / w.sum()
        return mean_absolute_error(y, sub_oofs @ w)
    x0s = np.zeros(len(valid)); x0s[0] = 0.85
    for i in range(1, len(valid)): x0s[i] = 0.15 / (len(valid)-1)
    rs = minimize(sub_obj, x0s, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 50000})
    ws = np.clip(rs.x, 0, None); ws = ws / ws.sum()
    d = rs.fun - base
    print(f"\n  [{sub_name}]")
    for n_, w_ in zip(valid, ws):
        print(f"    {n_}: {w_:.4f}")
    print(f"    mae: {rs.fun:.5f}  delta: {d:+.6f}")

# Generate final submission from full-model blend
os.makedirs("results/final_blend", exist_ok=True)
blend_test = tests @ w_opt
test_raw = pd.read_csv("test.csv")
pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    "results/final_blend/submission_final_multiblend.csv", index=False)
print(f"\nFinal submission: results/final_blend/submission_final_multiblend.csv")

# Verdict
if delta < -0.008:
    v = "GO"
elif delta < -0.005:
    v = "WEAK_GO"
elif delta < -0.003:
    v = "MARGINAL"
else:
    v = "NO_GO"
print(f"\n>>> FINAL VERDICT: {v} (delta {delta:+.6f}) <<<")

json.dump({
    "baseline": float(base),
    "blend_mae": float(res.fun),
    "delta": float(delta),
    "weights": dict(zip(names, w_opt.tolist())),
    "verdict": v,
}, open("results/final_blend/final_summary.json", "w"), indent=2, ensure_ascii=False)
