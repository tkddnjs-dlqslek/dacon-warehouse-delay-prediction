"""
Task C: 3-level stacking — optimize weights over 3 meta outputs.

Current: meta_avg = (lgb + xgb + cb) / 3
New: meta_avg = w_lgb * lgb + w_xgb * xgb + w_cb * cb, w_i >= 0, sum = 1
Optimized via CV (GroupKFold) to avoid meta-level overfit.

Also considers a slightly expanded search: allow negative small weights (no constraint),
and compare.
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
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "task_c")
os.makedirs(OUT, exist_ok=True)


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def find_weights(oofs, y, simplex=True):
    """Minimize MAE over convex combination of oofs."""
    def obj(w):
        pred = sum(wi * oi for wi, oi in zip(w, oofs))
        return np.mean(np.abs(y - pred))
    n = len(oofs)
    x0 = np.ones(n) / n
    if simplex:
        bounds = [(0, 1)] * n
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
        res = minimize(obj, x0, method="SLSQP", bounds=bounds, constraints=cons)
    else:
        res = minimize(obj, x0, method="Nelder-Mead")
    return res.x, float(res.fun)


def main():
    print("=" * 60, flush=True)
    print("Task C: 3-level stacking (meta weight optimization)", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    groups = train["layout_id"].values

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    meta_oofs = mega["meta_oofs"]
    meta_tests = mega["meta_tests"]
    avg_oof = mega["meta_avg_oof"]
    avg_test = mega["meta_avg_test"]

    baseline = mae(y, avg_oof)
    print(f"Baseline (simple avg) OOF MAE: {baseline:.5f}", flush=True)

    # Individual meta MAE
    for k in ["lgb", "xgb", "cb"]:
        m = mae(y, meta_oofs[k])
        print(f"  {k}: {m:.5f}", flush=True)

    oof_list = [meta_oofs[k] for k in ["lgb", "xgb", "cb"]]
    test_list = [meta_tests[k] for k in ["lgb", "xgb", "cb"]]

    # ─── 1. Global simplex-constrained optimization ───────────
    print("\n[1] Global simplex weights...", flush=True)
    w_global, mae_global = find_weights(oof_list, y, simplex=True)
    print(f"  weights: lgb={w_global[0]:.4f}, xgb={w_global[1]:.4f}, cb={w_global[2]:.4f}", flush=True)
    print(f"  OOF MAE: {mae_global:.5f}  delta {mae_global-baseline:+.5f}", flush=True)

    # ─── 2. Unconstrained (allow small negative / non-simplex) ─
    print("\n[2] Unconstrained weights...", flush=True)
    w_uncon, mae_uncon = find_weights(oof_list, y, simplex=False)
    print(f"  weights: lgb={w_uncon[0]:.4f}, xgb={w_uncon[1]:.4f}, cb={w_uncon[2]:.4f}  sum={w_uncon.sum():.4f}", flush=True)
    print(f"  OOF MAE: {mae_uncon:.5f}  delta {mae_uncon-baseline:+.5f}", flush=True)

    # ─── 3. CV-based weights (per-fold then averaged, honest OOF) ─
    print("\n[3] CV-based weights (per fold)...", flush=True)
    gkf = GroupKFold(n_splits=5)
    oof_cv = np.zeros_like(y)
    fold_weights = []
    for fold_i, (tr, val) in enumerate(gkf.split(avg_oof, y, groups=groups)):
        oofs_tr = [o[tr] for o in oof_list]
        w, _ = find_weights(oofs_tr, y[tr], simplex=True)
        fold_weights.append(w)
        oof_cv[val] = sum(wi * o[val] for wi, o in zip(w, oof_list))
        print(f"  fold {fold_i}: weights {w.round(3).tolist()}", flush=True)
    mae_cv = mae(y, oof_cv)
    avg_weights_cv = np.mean(fold_weights, axis=0)
    print(f"  avg CV weights: {avg_weights_cv.round(3).tolist()}", flush=True)
    print(f"  CV OOF MAE: {mae_cv:.5f}  delta {mae_cv-baseline:+.5f}", flush=True)

    # ─── Choose best (honest = CV weights; global is overfit to training OOF) ─
    if mae_cv < baseline - 0.0005:
        chosen_w = avg_weights_cv
        chosen_mae = mae_cv
        verdict = "PROMOTE (CV weights)"
    elif mae_global < baseline - 0.0005:
        # global is in-sample; still OK if stable across folds
        chosen_w = w_global
        chosen_mae = mae_global
        verdict = "PROMOTE (global weights, verify via CV)"
    else:
        chosen_w = np.array([1/3, 1/3, 1/3])
        chosen_mae = baseline
        verdict = "NO_GO (keep simple average)"

    print(f"\nVERDICT: {verdict}", flush=True)
    print(f"Chosen weights: lgb={chosen_w[0]:.4f}, xgb={chosen_w[1]:.4f}, cb={chosen_w[2]:.4f}", flush=True)
    print(f"Chosen OOF MAE: {chosen_mae:.5f}", flush=True)

    # Build test prediction with chosen weights
    new_test = sum(wi * t for wi, t in zip(chosen_w, test_list))
    new_oof = sum(wi * o for wi, o in zip(chosen_w, oof_list))

    np.save(os.path.join(OUT, "meta_avg_oof_3level.npy"), new_oof)
    np.save(os.path.join(OUT, "meta_avg_test_3level.npy"), new_test)

    summary = dict(
        baseline_simple_avg=baseline,
        global_weights=w_global.tolist(),
        global_mae=mae_global,
        uncon_weights=w_uncon.tolist(),
        uncon_mae=mae_uncon,
        cv_weights_avg=avg_weights_cv.tolist(),
        cv_mae=mae_cv,
        chosen_weights=chosen_w.tolist(),
        chosen_mae=chosen_mae,
        verdict=verdict,
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}/", flush=True)


if __name__ == "__main__":
    main()
