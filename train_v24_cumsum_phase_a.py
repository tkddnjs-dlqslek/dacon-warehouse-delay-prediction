"""
v24_cumsum Phase A: isolated A/B test of 3 cumsum features.

Goal: decide Go/No-Go for full v24_cumsum (4 models x 3 seeds + mega34).

A = v23 features (149, no extreme_prob) LGB-Huber single seed
B = A + {fault_count_15m, blocked_path_15m, near_collision_15m}_cumsum LGB-Huber same seed

Baseline for comparison: v23_seed42 LGB_Huber OOF MAE = 8.61185
  (from results/v23_seed42.pkl)
Success threshold: B_oof - A_oof <= -0.06

Safety:
  - Feature selection NOT re-run (v24 failure was "core features dropped").
  - v30 FE cache reused (identical recipe to v23 minus extreme_prob).
  - Same GroupKFold(layout_id, 5) as v30 Section 0.
  - Per-fold checkpoint: model + oof slice saved immediately.
  - log1p(y) target, expm1 predict, clip>=0 (v23 pattern).
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
OUT = ROOT / "results" / "v24_cumsum"
OUT.mkdir(parents=True, exist_ok=True)

V30_OUT = ROOT / "results" / "eda_v30"
FE_CACHE = V30_OUT / "v30_fe_cache.pkl"
FOLD_IDX_NPY = V30_OUT / "fold_idx.npy"

CUMSUM_COLS = ["fault_count_15m", "blocked_path_15m", "near_collision_15m"]
SEED = 42
N_SPLITS = 5
TARGET = "avg_delay_minutes_next_30m"

# v23 LGB-Huber config (train_v23.py L293-301)
LGB_HUBER_PARAMS = dict(
    objective="huber",
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

V23_SEED42_BASELINE = 8.61185  # v23_seed42['oofs']['LGB_Huber'] MAE
TARGET_DELTA = -0.06           # success threshold


def load_inputs() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    print("loading v30 FE cache ...")
    with open(FE_CACHE, "rb") as f:
        blob = pickle.load(f)
    train_fe: pd.DataFrame = blob["train_fe"]
    feat_cols: list[str] = list(blob["feat_cols"])
    print(f"  train_fe shape = {train_fe.shape}, feat_cols = {len(feat_cols)}")

    fold_ids = np.load(FOLD_IDX_NPY)
    assert len(fold_ids) == len(train_fe), (
        f"fold_ids len mismatch: {len(fold_ids)} vs {len(train_fe)}"
    )
    print(f"  fold_ids loaded, per-fold sizes = {np.bincount(fold_ids).tolist()}")

    y = train_fe[TARGET].values.astype(np.float64)
    print(f"  y: mean={y.mean():.4f} std={y.std():.4f}")
    return train_fe, feat_cols, y, fold_ids


def build_cumsum(train_fe: pd.DataFrame) -> pd.DataFrame:
    """Add {col}_cumsum for CUMSUM_COLS. NaN-safe (col mean fill)."""
    df = train_fe[["layout_id", "scenario_id", "timeslot"]].copy()
    # engineer_features_v23 already sorted train_fe by [layout_id, scenario_id, timeslot]
    # so we can groupby without re-sort and get correct row alignment
    for col in CUMSUM_COLS:
        raw = train_fe[col].values.astype(np.float64)
        col_mean = np.nanmean(raw)
        filled = np.where(np.isnan(raw), col_mean, raw)
        tmp = pd.DataFrame(
            {
                "layout_id": train_fe["layout_id"].values,
                "scenario_id": train_fe["scenario_id"].values,
                "ts": train_fe["timeslot"].values,
                "v": filled,
            }
        )
        # already sorted — groupby cumsum respects row order
        cum = tmp.groupby(["layout_id", "scenario_id"], sort=False)["v"].cumsum().values
        df[f"{col}_cumsum"] = cum
        print(
            f"  built {col}_cumsum: min={cum.min():.3f} "
            f"max={cum.max():.3f} mean={cum.mean():.3f}"
        )
    return df[[f"{c}_cumsum" for c in CUMSUM_COLS]]


def train_variant(
    X: pd.DataFrame,
    y: np.ndarray,
    fold_ids: np.ndarray,
    variant_name: str,
) -> dict:
    """Train 5-fold LGB-Huber on log1p(y). Return oof + fold metrics."""
    import lightgbm as lgb

    print(f"\n--- variant {variant_name} ({X.shape[1]} features) ---")
    y_log = np.log1p(y)
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows = []
    t_start = time.time()

    for f in range(N_SPLITS):
        val_idx = np.where(fold_ids == f)[0]
        tr_idx = np.where(fold_ids != f)[0]
        n_tr = len(tr_idx)
        n_val = len(val_idx)
        t_fold = time.time()

        model = lgb.LGBMRegressor(**LGB_HUBER_PARAMS)
        model.fit(
            X.iloc[tr_idx],
            y_log[tr_idx],
            eval_set=[(X.iloc[val_idx], y_log[val_idx])],
            callbacks=[
                lgb.early_stopping(200, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred_log = model.predict(X.iloc[val_idx])
        pred = np.clip(np.expm1(pred_log), 0, None)
        oof[val_idx] = pred

        fold_mae = mean_absolute_error(y[val_idx], pred)
        best_it = int(model.best_iteration_ or LGB_HUBER_PARAMS["n_estimators"])
        fold_rows.append(
            {
                "variant": variant_name,
                "fold": f,
                "n_train": n_tr,
                "n_val": n_val,
                "best_iter": best_it,
                "fold_mae": float(fold_mae),
                "seconds": round(time.time() - t_fold, 1),
            }
        )
        print(
            f"  fold {f}: mae={fold_mae:.5f} "
            f"best_it={best_it} ({time.time()-t_fold:.0f}s)"
        )

        # checkpoint after each fold
        np.save(OUT / f"oof_{variant_name}.npy", oof)
        pd.DataFrame(fold_rows).to_csv(
            OUT / f"fold_metrics_{variant_name}.csv", index=False
        )

    oof_mae = mean_absolute_error(y, oof)
    elapsed = time.time() - t_start
    print(
        f"  OVERALL oof_mae = {oof_mae:.5f}  "
        f"(elapsed {elapsed:.0f}s, {X.shape[1]} feats)"
    )
    return {"oof_mae": float(oof_mae), "fold_rows": fold_rows, "oof": oof}


def main():
    t0 = time.time()
    print("=" * 64)
    print("v24_cumsum Phase A - LGB-Huber A/B test")
    print("=" * 64)

    train_fe, feat_cols, y, fold_ids = load_inputs()

    # sanity: all feat_cols are in train_fe
    missing = [c for c in feat_cols if c not in train_fe.columns]
    assert not missing, f"feat_cols missing in train_fe: {missing[:5]}"

    # --- variant A: v23 features (149, no extreme_prob) ---
    X_A = train_fe[feat_cols].copy()
    print(f"\nX_A shape = {X_A.shape}")

    # --- variant B: A + 3 cumsum ---
    cum_df = build_cumsum(train_fe)
    X_B = pd.concat([X_A, cum_df], axis=1)
    print(f"X_B shape = {X_B.shape}")
    assert X_B.shape[1] == X_A.shape[1] + 3

    # --- train both ---
    res_A = train_variant(X_A, y, fold_ids, "A_v23feats")
    res_B = train_variant(X_B, y, fold_ids, "B_v23feats_plus_cumsum")

    delta = res_B["oof_mae"] - res_A["oof_mae"]
    delta_vs_v23 = res_B["oof_mae"] - V23_SEED42_BASELINE

    # --- verdict ---
    verdict = "GO" if delta <= TARGET_DELTA else (
        "WEAK" if delta < 0 else "NO_GO"
    )

    summary = {
        "variant_A_v23feats_oof_mae": res_A["oof_mae"],
        "variant_B_cumsum_oof_mae": res_B["oof_mae"],
        "delta_B_minus_A": delta,
        "target_delta": TARGET_DELTA,
        "v23_seed42_LGB_Huber_baseline": V23_SEED42_BASELINE,
        "delta_B_minus_v23_seed42_baseline": delta_vs_v23,
        "cumsum_cols": CUMSUM_COLS,
        "n_feats_A": int(X_A.shape[1]),
        "n_feats_B": int(X_B.shape[1]),
        "seed": SEED,
        "n_splits": N_SPLITS,
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "phase_a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 64)
    print("PHASE A SUMMARY")
    print("=" * 64)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n>>> VERDICT: {verdict} <<<")

    # touch checkpoint
    (OUT / "checkpoint_phase_a.done").write_text(
        f"verdict={verdict} delta={delta:+.5f}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
