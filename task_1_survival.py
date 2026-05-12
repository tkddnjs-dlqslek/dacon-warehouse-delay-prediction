"""
Task 1: Survival loss — xgboost objective='survival:cox'

Treats y (delay in minutes) as "time-to-event" where all observations are "observed" (no censoring).
Cox proportional hazards assumes:
  hazard(t | X) = h0(t) * exp(X @ beta)

Prediction = exp(X @ beta) = relative hazard rate (not MAE-calibrated directly).

For MAE prediction, we need to convert hazard score back to expected time.
Simple approach: map hazard score to quantile of training y distribution.
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
import xgboost as xgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "survival")
os.makedirs(OUT, exist_ok=True)


def hazard_to_expected(haz_score, y_train_sorted):
    """Map hazard score to expected y via rank correspondence.
    Higher hazard → higher expected delay.
    Uses empirical CDF of training y to calibrate.
    """
    # Rank haz_score across all rows (higher = higher expected y)
    # Map each rank to corresponding quantile of y_train distribution
    from scipy.stats import rankdata
    n = len(haz_score)
    ranks = rankdata(haz_score, method="average")  # 1..n
    quantiles = (ranks - 0.5) / n  # in (0, 1)
    # Empirical quantile from sorted y_train
    idx = np.clip((quantiles * len(y_train_sorted)).astype(int), 0, len(y_train_sorted) - 1)
    return y_train_sorted[idx]


def main():
    print("=" * 60, flush=True)
    print("Task 1: Survival (Cox) loss base model", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values
    print(f"X shape: {X.shape}", flush=True)

    # For survival:cox, y must be non-zero (or represent time-to-event >= 0)
    # Cox requires positive time (no zero). Clip to tiny epsilon for y==0
    y_cox = np.clip(y, 1e-3, None)
    # All observations "observed" (no censoring) — label = event time
    # xgboost survival:cox: label > 0 means observed, label < 0 means censored at abs(label)
    print(f"y_cox: min={y_cox.min():.4f}, max={y_cox.max():.3f}, mean={y_cox.mean():.3f}", flush=True)

    # Sorted y for hazard-to-expected mapping
    y_sorted = np.sort(y)

    # Existing v23s42 for comparison
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"\nReference v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)

    # Mega33 for corr check
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))
    print(f"Mega33 OOF MAE: {mega_mae:.5f}", flush=True)

    # Train XGB survival:cox 5-fold
    print("\n=== Training XGB survival:cox (5-fold) ===", flush=True)
    hazard_oof = np.zeros(len(y), dtype=np.float64)
    hazard_test = np.zeros(len(X_test), dtype=np.float64)
    fold_mae_converted = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        dtrain = xgb.DMatrix(X[tr_mask], label=y_cox[tr_mask])
        dval = xgb.DMatrix(X[val_mask], label=y_cox[val_mask])
        dtest = xgb.DMatrix(X_test)

        params = dict(
            objective="survival:cox",
            eval_metric="cox-nloglik",
            eta=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            alpha=1.0,
            lambda_=1.0,
            min_child_weight=10,
            tree_method="hist",
            verbosity=0,
            seed=42,
        )
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            evals=[(dval, "val")],
            early_stopping_rounds=100,
            verbose_eval=False,
        )
        # Predict hazard score (= exp(X @ beta))
        val_haz = bst.predict(dval)
        test_haz = bst.predict(dtest)

        # Convert hazard to expected y via empirical quantile mapping
        # Use train-fold y_sorted to avoid leakage
        y_train_sorted = np.sort(y[tr_mask])
        val_pred = hazard_to_expected(val_haz, y_train_sorted)
        test_pred_fold = hazard_to_expected(test_haz, y_train_sorted)

        hazard_oof[val_mask] = val_pred
        hazard_test += test_pred_fold / 5
        fold_mae = float(np.mean(np.abs(val_pred - y[val_mask])))
        fold_mae_converted.append(fold_mae)
        print(f"  fold {f}: best_iter={bst.best_iteration} val MAE (converted)={fold_mae:.4f}", flush=True)

    cox_mae = float(mean_absolute_error(y, hazard_oof))
    corr_mega = float(np.corrcoef(y - mega_oof, y - hazard_oof)[0, 1])
    corr_v23 = float(np.corrcoef(y - old_oof, y - hazard_oof)[0, 1])

    print(f"\n=== Survival Cox results ===", flush=True)
    print(f"Single OOF MAE: {cox_mae:.5f}", flush=True)
    print(f"residual_corr vs mega33: {corr_mega:.4f}", flush=True)
    print(f"residual_corr vs v23s42_LGB_Huber: {corr_v23:.4f}", flush=True)

    # Blend test with mega33 (simple 2-way optim)
    print("\nSimple 2-way blend mega33 + survival:", flush=True)
    best_mae = mega_mae
    best_w = 0.0
    for w in np.linspace(0, 0.5, 51):
        pred = (1 - w) * mega_oof + w * hazard_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m
            best_w = w
    print(f"  best w_survival = {best_w:.3f}  blend MAE = {best_mae:.5f}  delta = {best_mae - mega_mae:+.5f}", flush=True)

    # Kill gates
    bad_quality = cox_mae > 9.5
    too_similar = corr_mega >= 0.95
    print("\nKill gates:", flush=True)
    print(f"  single MAE > 9.5: {bad_quality}", flush=True)
    print(f"  corr(mega) >= 0.95: {too_similar}", flush=True)

    np.save(os.path.join(OUT, "cox_oof.npy"), hazard_oof)
    np.save(os.path.join(OUT, "cox_test.npy"), hazard_test)
    summary = dict(
        single_mae=cox_mae,
        corr_vs_mega33=corr_mega,
        corr_vs_v23=corr_v23,
        best_blend_w=best_w,
        best_blend_mae=best_mae,
        delta_vs_mega=best_mae - mega_mae,
        fold_mae=fold_mae_converted,
        gates=dict(bad_quality=bad_quality, too_similar=too_similar),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as fp:
        json.dump(summary, fp, indent=2)

    if bad_quality or too_similar:
        print("\nVERDICT: NO_GO", flush=True)
    else:
        print("\nVERDICT: PROCEED — test inclusion in multi-blend", flush=True)


if __name__ == "__main__":
    main()
