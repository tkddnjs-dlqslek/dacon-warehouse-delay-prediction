"""
P3 Quantile Phase A: train Q30/Q50/Q70/Q90 LGB on v23 features.
Then blend with mega33 OOF to measure stacking diversity benefit.

Key insight: mega33's 33 bases all use MAE/Huber/AbsoluteError objective.
Quantile loss gives a genuinely different optimization landscape = diversity.
- Q30 underpredicts (30th pctile target)
- Q50 is median (robust vs MAE)
- Q70/Q90 overpredict tail

Past: v22 tried Q30/Q50/Q70 but with v22 features (no SC features).
New: v23 features + Q90 (extreme tail focus, never tried).
"""
from __future__ import annotations
import os
import time
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(r"c:/Users/user/Desktop/데이콘 4월")
OUT = ROOT / "results" / "v24_cumsum"   # reuse same output dir
OUT.mkdir(parents=True, exist_ok=True)

V30_OUT = ROOT / "results" / "eda_v30"
FE_CACHE = V30_OUT / "v30_fe_cache.pkl"
FOLD_IDX_NPY = V30_OUT / "fold_idx.npy"
MEGA33_PKL = ROOT / "results" / "mega33_final.pkl"

TARGET = "avg_delay_minutes_next_30m"
SORT_KEY = ["layout_id", "scenario_id"]
SEED = 42
N_SPLITS = 5

ALPHAS = [0.3, 0.5, 0.7, 0.9]

LGB_BASE_PARAMS = dict(
    n_estimators=5000,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    min_child_samples=50,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.0,
    reg_lambda=1.0,
    random_state=SEED,
    verbose=-1,
    n_jobs=-1,
)


def main():
    import lightgbm as lgb

    t0 = time.time()
    print("=" * 64)
    print("P3 Quantile Phase A - Q30/Q50/Q70/Q90 + mega33 blend test")
    print("=" * 64)

    # --- load data ---
    print("loading data ...")
    with open(FE_CACHE, "rb") as f:
        blob = pickle.load(f)
    train_fe = blob["train_fe"]
    feat_cols = list(blob["feat_cols"])
    fold_ids = np.load(FOLD_IDX_NPY)
    y = train_fe[TARGET].values.astype(np.float64)
    y_log = np.log1p(y)
    X = train_fe[feat_cols]
    print(f"  X shape = {X.shape}, y mean = {y.mean():.4f}")

    # --- load mega33 OOF (sorted same as v30) ---
    with open(MEGA33_PKL, "rb") as f:
        mega = pickle.load(f)
    mega33_oof = np.asarray(mega["meta_avg_oof"])
    baseline_mae = mean_absolute_error(y, mega33_oof)
    print(f"  mega33 baseline MAE = {baseline_mae:.5f}")

    # --- train quantile LGBs ---
    quantile_oofs = {}
    all_fold_rows = []
    for alpha in ALPHAS:
        name = f"Q{int(alpha*100)}"
        print(f"\n--- {name} (alpha={alpha}) ---")
        params = {**LGB_BASE_PARAMS, "objective": "quantile", "alpha": alpha}
        oof = np.zeros(len(y), dtype=np.float64)
        for f in range(N_SPLITS):
            tr_idx = np.where(fold_ids != f)[0]
            val_idx = np.where(fold_ids == f)[0]
            t_f = time.time()
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X.iloc[tr_idx], y_log[tr_idx],
                eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
            )
            pred = np.clip(np.expm1(model.predict(X.iloc[val_idx])), 0, None)
            oof[val_idx] = pred
            fold_mae = mean_absolute_error(y[val_idx], pred)
            best_it = int(model.best_iteration_ or 5000)
            all_fold_rows.append(
                {"model": name, "fold": f, "mae": float(fold_mae),
                 "best_iter": best_it, "seconds": round(time.time() - t_f, 1)}
            )
            print(f"  fold {f}: mae={fold_mae:.5f} best_it={best_it} ({time.time()-t_f:.0f}s)")

        oof_mae = mean_absolute_error(y, oof)
        quantile_oofs[name] = oof
        np.save(OUT / f"oof_{name}.npy", oof)
        print(f"  OVERALL OOF MAE = {oof_mae:.5f}")

    pd.DataFrame(all_fold_rows).to_csv(OUT / "quantile_fold_metrics.csv", index=False)

    # --- diversity analysis: residual correlation with mega33 ---
    print("\n--- Diversity Analysis ---")
    mega33_res = y - mega33_oof
    diversity = {}
    for name, oof in quantile_oofs.items():
        res = y - oof
        corr = float(np.corrcoef(mega33_res, res)[0, 1])
        oof_mae = mean_absolute_error(y, oof)
        diversity[name] = {"oof_mae": oof_mae, "residual_corr_with_mega33": corr}
        print(f"  {name}: oof_mae={oof_mae:.5f}  residual_corr_mega33={corr:.4f}")

    # --- blend scan: mega33 * (1-w) + Q * w ---
    print("\n--- Blend Scan (mega33 + quantile) ---")
    blend_rows = []
    for name, oof in quantile_oofs.items():
        for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30]:
            blend = (1 - w) * mega33_oof + w * oof
            mae = mean_absolute_error(y, blend)
            blend_rows.append(
                {"model": name, "weight": w, "oof_mae": float(mae),
                 "delta": float(mae - baseline_mae)}
            )
    # also try multi-quantile blend
    for w_total in [0.10, 0.15, 0.20, 0.25, 0.30]:
        for combo_name, combo_alphas in [
            ("Q50+Q70", [0.5, 0.7]),
            ("Q30+Q50+Q70", [0.3, 0.5, 0.7]),
            ("Q50+Q70+Q90", [0.5, 0.7, 0.9]),
            ("Q30+Q50+Q70+Q90", [0.3, 0.5, 0.7, 0.9]),
        ]:
            names = [f"Q{int(a*100)}" for a in combo_alphas]
            q_avg = np.mean([quantile_oofs[n] for n in names], axis=0)
            blend = (1 - w_total) * mega33_oof + w_total * q_avg
            mae = mean_absolute_error(y, blend)
            blend_rows.append(
                {"model": combo_name, "weight": w_total, "oof_mae": float(mae),
                 "delta": float(mae - baseline_mae)}
            )

    blend_df = pd.DataFrame(blend_rows).sort_values("oof_mae").reset_index(drop=True)
    blend_df.to_csv(OUT / "quantile_blend_scan.csv", index=False)
    print(f"\nTop 10 blends:")
    for _, r in blend_df.head(10).iterrows():
        print(
            f"  {r['model']:20s} w={r['weight']:.2f} "
            f"mae={r['oof_mae']:.5f} delta={r['delta']:+.5f}"
        )

    best = blend_df.iloc[0]
    best_delta = float(best["delta"])
    passes = best_delta <= -0.01
    verdict = "GO" if best_delta <= -0.02 else ("WEAK_GO" if passes else "NO_GO")

    summary = {
        "baseline_mega33_mae": baseline_mae,
        "best_blend_model": str(best["model"]),
        "best_blend_weight": float(best["weight"]),
        "best_blend_mae": float(best["oof_mae"]),
        "best_delta": best_delta,
        "verdict": verdict,
        "diversity": diversity,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "quantile_phase_a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*64}")
    print(f"VERDICT: {verdict}")
    print(f"  best delta = {best_delta:+.5f} (threshold: <=-0.02 GO, <=-0.01 WEAK_GO)")
    print(f"  baseline = {baseline_mae:.5f} -> best = {best['oof_mae']:.5f}")
    print(f"  elapsed = {time.time()-t0:.0f}s")
    print(f"{'='*64}")

    (OUT / "checkpoint_quantile_phase_a.done").write_text(
        f"verdict={verdict} delta={best_delta:+.5f}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
