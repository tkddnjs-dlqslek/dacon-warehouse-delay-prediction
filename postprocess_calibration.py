"""
Post-processing bucket calibration.

Compute per-bucket (layout_type x ts_bucket) mean residual on OOF,
then apply as additive correction with shrinkage.

No model retraining. Just shift mega33 predictions per bucket.
Overfitting risk: LOW (OOF is genuine out-of-sample, large bucket sizes, 1st moment only).
"""
from __future__ import annotations
import pickle
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(r"c:/Users/user/Desktop/데이콘 4월")
RESULT_DIR = ROOT / "results"
OUT = RESULT_DIR / "v24_cumsum"
OUT.mkdir(parents=True, exist_ok=True)

MEGA33_PKL = RESULT_DIR / "mega33_final.pkl"
FE_CACHE = RESULT_DIR / "eda_v30" / "v30_fe_cache.pkl"
FOLD_IDX = RESULT_DIR / "eda_v30" / "fold_idx.npy"
TARGET = "avg_delay_minutes_next_30m"
EXPECTED_BASELINE = 8.3989


def main():
    t0 = time.time()
    print("=" * 64)
    print("Post-Processing Bucket Calibration")
    print("=" * 64)

    # --- load data ---
    train = pd.read_csv(ROOT / "train.csv").sort_values(
        ["layout_id", "scenario_id"]
    ).reset_index(drop=True)
    y = train[TARGET].values.astype(np.float64)

    with open(MEGA33_PKL, "rb") as f:
        mega = pickle.load(f)
    mega33_oof = np.asarray(mega["meta_avg_oof"])
    baseline_mae = mean_absolute_error(y, mega33_oof)
    print(f"baseline MAE = {baseline_mae:.5f}")
    assert abs(baseline_mae - EXPECTED_BASELINE) < 0.005

    fold_ids = np.load(FOLD_IDX)

    # --- build bucket ids ---
    with open(FE_CACHE, "rb") as f:
        train_fe = pickle.load(f)["train_fe"]
    timeslot = train_fe["timeslot"].values
    ts_bucket = (timeslot // 5).astype(np.int8)

    layout = pd.read_csv(ROOT / "layout_info.csv")[["layout_id", "layout_type"]]
    meta = train[["layout_id"]].merge(layout, on="layout_id", how="left")
    layout_type = meta["layout_type"].values

    type_map = {"narrow": 0, "grid": 1, "hybrid": 2, "hub_spoke": 3}
    lt_code = np.array([type_map[t] for t in layout_type], dtype=np.int8)

    # combined bucket id
    bucket_id = lt_code * 5 + ts_bucket  # 0..19

    residual = y - mega33_oof
    print(f"residual mean = {residual.mean():+.5f}, std = {residual.std():.5f}")

    # --- Method 1: global bucket mean correction ---
    # compute per-bucket mean residual
    bucket_stats = {}
    for b in range(20):
        mask = bucket_id == b
        n = int(mask.sum())
        if n < 100:
            bucket_stats[b] = {"n": n, "mean_res": 0.0, "std_res": 0.0}
            continue
        mean_res = float(residual[mask].mean())
        std_res = float(residual[mask].std())
        bucket_stats[b] = {"n": n, "mean_res": mean_res, "std_res": std_res}

    print("\nbucket residual stats:")
    types_list = ["narrow", "grid", "hybrid", "hub_spoke"]
    for lt_i, lt_name in enumerate(types_list):
        for tb in range(5):
            b = lt_i * 5 + tb
            s = bucket_stats[b]
            print(
                f"  {lt_name:10s} ts{tb}: n={s['n']:5d} "
                f"mean_res={s['mean_res']:+.4f} std_res={s['std_res']:.4f}"
            )

    # --- Method 2: fold-aware bucket correction (more robust) ---
    # For each fold f:
    #   compute bucket_mean_residual on folds != f (train folds)
    #   apply to fold f (val fold)
    # This avoids any within-fold leakage in the OOF evaluation
    print("\n--- Fold-aware calibration scan ---")
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    scan_rows = []

    for alpha in alphas:
        calibrated_oof = mega33_oof.copy()
        for f in range(5):
            tr_mask = fold_ids != f
            val_mask = fold_ids == f
            # compute bucket means on train folds ONLY
            for b in range(20):
                b_tr = tr_mask & (bucket_id == b)
                b_val = val_mask & (bucket_id == b)
                if b_tr.sum() < 50 or b_val.sum() == 0:
                    continue
                mean_correction = float(residual[b_tr].mean())
                calibrated_oof[b_val] += alpha * mean_correction

        calibrated_oof = np.clip(calibrated_oof, 0, None)
        cal_mae = mean_absolute_error(y, calibrated_oof)
        delta = cal_mae - baseline_mae
        scan_rows.append({
            "method": "fold_aware_bucket",
            "alpha": alpha,
            "oof_mae": float(cal_mae),
            "delta": float(delta),
        })
        tag = " ***" if delta < -0.01 else ""
        print(f"  alpha={alpha:.1f}: mae={cal_mae:.5f} delta={delta:+.5f}{tag}")

    # --- Method 3: finer buckets (layout_type x timeslot, 100 buckets) ---
    fine_bucket = lt_code * 25 + timeslot  # 0..99
    print("\n--- Fine bucket (100 buckets) scan ---")
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        calibrated_oof = mega33_oof.copy()
        for f in range(5):
            tr_mask = fold_ids != f
            val_mask = fold_ids == f
            for b in range(100):
                b_tr = tr_mask & (fine_bucket == b)
                b_val = val_mask & (fine_bucket == b)
                if b_tr.sum() < 30 or b_val.sum() == 0:
                    continue
                mean_correction = float(residual[b_tr].mean())
                calibrated_oof[b_val] += alpha * mean_correction
        calibrated_oof = np.clip(calibrated_oof, 0, None)
        cal_mae = mean_absolute_error(y, calibrated_oof)
        delta = cal_mae - baseline_mae
        scan_rows.append({
            "method": "fold_aware_fine_100",
            "alpha": alpha,
            "oof_mae": float(cal_mae),
            "delta": float(delta),
        })
        tag = " ***" if delta < -0.01 else ""
        print(f"  alpha={alpha:.1f}: mae={cal_mae:.5f} delta={delta:+.5f}{tag}")

    # --- Method 4: scenario-level correction ---
    scenario_key = (
        train["layout_id"].astype(str) + "_" + train["scenario_id"].astype(str)
    ).values
    sc_codes, sc_uniq = pd.factorize(scenario_key)
    print(f"\n--- Scenario-level ({len(sc_uniq)} groups) scan ---")
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        calibrated_oof = mega33_oof.copy()
        for f in range(5):
            tr_mask = fold_ids != f
            val_mask = fold_ids == f
            for s in range(len(sc_uniq)):
                s_tr = tr_mask & (sc_codes == s)
                s_val = val_mask & (sc_codes == s)
                if s_tr.sum() < 5 or s_val.sum() == 0:
                    continue
                mean_correction = float(residual[s_tr].mean())
                calibrated_oof[s_val] += alpha * mean_correction
        calibrated_oof = np.clip(calibrated_oof, 0, None)
        cal_mae = mean_absolute_error(y, calibrated_oof)
        delta = cal_mae - baseline_mae
        scan_rows.append({
            "method": "fold_aware_scenario",
            "alpha": alpha,
            "oof_mae": float(cal_mae),
            "delta": float(delta),
        })
        tag = " ***" if delta < -0.01 else ""
        print(f"  alpha={alpha:.1f}: mae={cal_mae:.5f} delta={delta:+.5f}{tag}")

    # --- summary ---
    scan_df = pd.DataFrame(scan_rows).sort_values("oof_mae").reset_index(drop=True)
    scan_df.to_csv(OUT / "postprocess_scan.csv", index=False)

    best = scan_df.iloc[0]
    best_delta = float(best["delta"])
    verdict = "GO" if best_delta <= -0.02 else ("WEAK_GO" if best_delta < 0 else "NO_GO")

    summary = {
        "baseline_mae": baseline_mae,
        "best_method": str(best["method"]),
        "best_alpha": float(best["alpha"]),
        "best_mae": float(best["oof_mae"]),
        "best_delta": best_delta,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "postprocess_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*64}")
    print("TOP 5 CALIBRATIONS:")
    for _, r in scan_df.head(5).iterrows():
        print(
            f"  {r['method']:25s} alpha={r['alpha']:.1f} "
            f"mae={r['oof_mae']:.5f} delta={r['delta']:+.5f}"
        )
    print(f"\n>>> VERDICT: {verdict} <<<")
    print(f"  best = {best['method']} alpha={best['alpha']:.1f} delta={best_delta:+.5f}")
    print(f"  elapsed = {time.time()-t0:.0f}s")

    (OUT / "checkpoint_postprocess.done").write_text(
        f"verdict={verdict} delta={best_delta:+.5f}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
