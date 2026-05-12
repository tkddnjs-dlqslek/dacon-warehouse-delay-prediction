"""
Post-hoc calibration: learn a monotone mapping pred -> corrected_pred on OOF.

Methods:
1. Isotonic regression (piecewise constant monotone)
2. Quantile matching (match pred CDF to y CDF)
3. Spline fit (smooth monotone)

Apply to:
- mega33 (baseline, LB 9.7759)
- fixed (our best blend, LB 9.7711)

Goal: reduce systematic bias in predictions.
"""
import pickle, numpy as np, pandas as pd, os, json
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression

OUT = "results/calibration"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Calibration Experiment (Isotonic / Quantile matching)")
print("=" * 64)

# Load
train = pd.read_csv("train.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
y = train[TARGET].values.astype(np.float64)
fold_ids = np.load("results/eda_v30/fold_idx.npy")

with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mega = mean_absolute_error(y, mega33_oof)

# Load fixed blend OOFs
w = {"mega33":0.7637, "rank_adj":0.1589, "iter_r1":0.0119, "iter_r2":0.0346, "iter_r3":0.0310}
fixed_oof = (w["mega33"]*mega33_oof
           + w["rank_adj"]*np.load("results/ranking/rank_adj_oof.npy")
           + w["iter_r1"]*np.load("results/iter_pseudo/round1_oof.npy")
           + w["iter_r2"]*np.load("results/iter_pseudo/round2_oof.npy")
           + w["iter_r3"]*np.load("results/iter_pseudo/round3_oof.npy"))
fixed_test = (w["mega33"]*mega33_test
            + w["rank_adj"]*np.load("results/ranking/rank_adj_test.npy")
            + w["iter_r1"]*np.load("results/iter_pseudo/round1_test.npy")
            + w["iter_r2"]*np.load("results/iter_pseudo/round2_test.npy")
            + w["iter_r3"]*np.load("results/iter_pseudo/round3_test.npy"))
baseline_fixed = mean_absolute_error(y, fixed_oof)

print(f"mega33 OOF: {baseline_mega:.5f}")
print(f"fixed  OOF: {baseline_fixed:.5f}")

def calibrate_isotonic(oof_pred, y_true, test_pred, fold_ids, n_folds=5):
    """Fold-safe isotonic calibration. Fit on train folds, apply to val."""
    cal_oof = np.zeros_like(oof_pred)
    for f in range(n_folds):
        tr_mask = fold_ids != f
        val_mask = fold_ids == f
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(oof_pred[tr_mask], y_true[tr_mask])
        cal_oof[val_mask] = iso.predict(oof_pred[val_mask])
    # final: fit on ALL OOF, apply to test
    iso_full = IsotonicRegression(out_of_bounds='clip')
    iso_full.fit(oof_pred, y_true)
    cal_test = iso_full.predict(test_pred)
    return cal_oof, cal_test

def calibrate_quantile_match(oof_pred, y_true, test_pred, fold_ids, n_folds=5):
    """Match oof_pred quantiles to y quantiles. Per-fold.  """
    cal_oof = np.zeros_like(oof_pred)
    for f in range(n_folds):
        tr_mask = fold_ids != f
        val_mask = fold_ids == f
        # sorted y train
        sorted_y = np.sort(y_true[tr_mask])
        # for each val pred, find its quantile in train OOF, map to y quantile
        sorted_tr_pred = np.sort(oof_pred[tr_mask])
        val_preds = oof_pred[val_mask]
        qs = np.searchsorted(sorted_tr_pred, val_preds) / len(sorted_tr_pred)
        # map to y values via interpolation
        y_quantile_positions = np.linspace(0, 1, len(sorted_y))
        cal_oof[val_mask] = np.interp(qs, y_quantile_positions, sorted_y)
    # final: full train
    sorted_y = np.sort(y_true)
    sorted_tr_pred = np.sort(oof_pred)
    qs = np.searchsorted(sorted_tr_pred, test_pred) / len(sorted_tr_pred)
    y_quantile_positions = np.linspace(0, 1, len(sorted_y))
    cal_test = np.interp(qs, y_quantile_positions, sorted_y)
    return cal_oof, cal_test

def test_calibration(oof, test, name, baseline):
    print(f"\n--- {name} ---")
    print(f"  baseline OOF: {baseline:.5f}")

    # isotonic
    c_oof, c_test = calibrate_isotonic(oof, y, test, fold_ids)
    c_mae = mean_absolute_error(y, c_oof)
    d = c_mae - baseline
    print(f"  isotonic    OOF: {c_mae:.5f}  delta: {d:+.5f}")

    # quantile match
    q_oof, q_test = calibrate_quantile_match(oof, y, test, fold_ids)
    q_mae = mean_absolute_error(y, q_oof)
    d_q = q_mae - baseline
    print(f"  quantile    OOF: {q_mae:.5f}  delta: {d_q:+.5f}")

    return {"iso_oof": c_oof, "iso_test": c_test, "iso_mae": c_mae, "iso_delta": d,
            "q_oof": q_oof, "q_test": q_test, "q_mae": q_mae, "q_delta": d_q}

r_mega = test_calibration(mega33_oof, mega33_test, "mega33 calibration", baseline_mega)
r_fixed = test_calibration(fixed_oof, fixed_test, "fixed calibration", baseline_fixed)

# Generate submissions
test_raw = pd.read_csv("test.csv")
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values

def save_sub(test_preds, name):
    pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(test_preds, 0, None)}).to_csv(
        f"{OUT}/submission_{name}.csv", index=False)
    print(f"  saved: {name}")

print("\n--- Submissions ---")
if r_mega["iso_delta"] < 0:
    save_sub(r_mega["iso_test"], "mega33_iso")
if r_mega["q_delta"] < 0:
    save_sub(r_mega["q_test"], "mega33_quantile")
if r_fixed["iso_delta"] < 0:
    save_sub(r_fixed["iso_test"], "fixed_iso")
if r_fixed["q_delta"] < 0:
    save_sub(r_fixed["q_test"], "fixed_quantile")

# Best
all_results = {
    "mega33_iso": r_mega["iso_delta"],
    "mega33_quantile": r_mega["q_delta"],
    "fixed_iso": r_fixed["iso_delta"],
    "fixed_quantile": r_fixed["q_delta"],
}
print("\n--- Summary ---")
for k, v in sorted(all_results.items(), key=lambda x: x[1]):
    tag = " ***" if v < -0.003 else (" **" if v < -0.001 else "")
    print(f"  {k:20s}: delta {v:+.5f}{tag}")

best_name = min(all_results, key=all_results.get)
print(f"\nBest: {best_name} (delta {all_results[best_name]:+.5f})")
if all_results[best_name] < 0:
    print(f"-> Submit: results/calibration/submission_{best_name}.csv")
else:
    print(f"-> No calibration helps, don't submit")

json.dump({k: float(v) for k, v in all_results.items()},
          open(f"{OUT}/summary.json","w"), indent=2)
