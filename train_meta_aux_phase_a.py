"""
Auxiliary-Aware Meta-Stacking Phase A.

Reload all 33 base OOFs (no retraining), add auxiliary features
(timeslot, layout_type, congestion_sc_mean, fault_sc_sum) to
the meta-learner input, and compare vs the current OOF-only meta.

A = 33 base OOFs (current mega33 structure)
B = 33 base OOFs + 4 auxiliary features

Success: B_mae <= A_mae - 0.02
"""
from __future__ import annotations
import time
import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

ROOT = Path(r"c:/Users/user/Desktop/데이콘 4월")
RESULT_DIR = ROOT / "results"
OUT = RESULT_DIR / "v24_cumsum"
OUT.mkdir(parents=True, exist_ok=True)

V30_OUT = RESULT_DIR / "eda_v30"
FE_CACHE = V30_OUT / "v30_fe_cache.pkl"
FOLD_IDX = V30_OUT / "fold_idx.npy"
MEGA33_PKL = RESULT_DIR / "mega33_final.pkl"

TARGET = "avg_delay_minutes_next_30m"
SEED = 42
N_SPLITS = 5
EXPECTED_BASELINE = 8.3989


# ==================================================================
# Step 0: Load all 33 base OOFs (same logic as train_neural_army.py L503-544)
# ==================================================================
def load_33_base_oofs() -> dict[str, np.ndarray]:
    """Load all 33 base model OOFs. Returns {name: oof_array}."""
    oofs: dict[str, np.ndarray] = {}

    # v23 x 3 seeds x 3 models = 9
    for seed in [42, 123, 2024]:
        s = pickle.load(open(RESULT_DIR / f"v23_seed{seed}.pkl", "rb"))
        for name in ["LGB_Huber", "XGB", "CatBoost"]:
            oofs[f"v23s{seed}_{name}"] = np.asarray(s["oofs"][name])

    # v24 = 4
    v24 = pickle.load(open(RESULT_DIR / "v24_final.pkl", "rb"))
    for name, oof in v24["oofs"].items():
        oofs[f"v24_{name}"] = np.asarray(oof)

    # v26 = 4
    v26 = pickle.load(open(RESULT_DIR / "v26_final.pkl", "rb"))
    for name in ["Tuned_Huber", "Tuned_sqrt", "Tuned_pow", "DART"]:
        oofs[f"v26_{name}"] = np.asarray(v26["oofs"][name])

    # mlp1, mlp2, cnn = 3
    mlp1 = pickle.load(open(RESULT_DIR / "mlp_final.pkl", "rb"))
    mlp2 = pickle.load(open(RESULT_DIR / "mlp2_final.pkl", "rb"))
    cnn = pickle.load(open(RESULT_DIR / "cnn_final.pkl", "rb"))
    oofs["mlp1"] = np.asarray(mlp1["mlp_oof"])
    oofs["mlp2"] = np.asarray(mlp2["mlp2_oof"])
    oofs["cnn"] = np.asarray(cnn["cnn_oof"])

    # domain = 3
    domain = pickle.load(open(RESULT_DIR / "domain_phase2.pkl", "rb"))
    for n in domain["oofs"]:
        oofs[f"domain_{n}"] = np.asarray(domain["oofs"][n])

    # mlp_aug = 1
    mlp_aug = pickle.load(open(RESULT_DIR / "mlp_aug_final.pkl", "rb"))
    oofs["mlp_aug"] = np.asarray(mlp_aug["mlp_aug_oof"])

    # offset = 3
    offset = pickle.load(open(RESULT_DIR / "offset_phase3.pkl", "rb"))
    for n, data in offset.items():
        oofs[f"offset_{n}"] = np.asarray(data["oof"])

    # neural_army = 6
    na = pickle.load(open(RESULT_DIR / "neural_army.pkl", "rb"))
    for name, data in na.items():
        oofs[f"na_{name}"] = np.asarray(data["oof"])

    return oofs


# ==================================================================
# Step 0b: Build stack matrix (log space, same as mega33)
# ==================================================================
def build_stack(oofs: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack(
        [np.log1p(np.clip(o, 0, None)) for o in oofs.values()]
    )


# ==================================================================
# Step 2: Build auxiliary features
# ==================================================================
def build_aux(train_fe: pd.DataFrame) -> np.ndarray:
    """4 auxiliary features aligned with train_fe row order."""
    assert "timeslot" in train_fe.columns
    assert "layout_id" in train_fe.columns
    assert "scenario_id" in train_fe.columns

    layout = pd.read_csv(ROOT / "layout_info.csv")[["layout_id", "layout_type"]]
    meta = train_fe[["layout_id"]].merge(layout, on="layout_id", how="left")

    # 1) timeslot_norm
    ts_norm = (train_fe["timeslot"].values / 24.0).astype(np.float32)

    # 2) layout_type_code
    type_map = {"narrow": 0, "grid": 1, "hybrid": 2, "hub_spoke": 3}
    lt_code = meta["layout_type"].map(type_map).values.astype(np.float32)
    assert not np.isnan(lt_code).any(), "layout_type mapping has NaN"

    # 3) congestion_sc_mean (scenario-level)
    cong = train_fe["congestion_score"].values.astype(np.float64)
    cong_filled = np.where(np.isnan(cong), np.nanmean(cong), cong)
    tmp = pd.DataFrame({
        "lid": train_fe["layout_id"].values,
        "sid": train_fe["scenario_id"].values,
        "v": cong_filled,
    })
    cong_sc_mean = (
        tmp.groupby(["lid", "sid"], sort=False)["v"]
        .transform("mean")
        .values.astype(np.float32)
    )

    # 4) fault_sc_sum (scenario-level)
    fault = train_fe["fault_count_15m"].values.astype(np.float64)
    fault_filled = np.where(np.isnan(fault), np.nanmean(fault), fault)
    tmp2 = pd.DataFrame({
        "lid": train_fe["layout_id"].values,
        "sid": train_fe["scenario_id"].values,
        "v": fault_filled,
    })
    fault_sc_sum = (
        tmp2.groupby(["lid", "sid"], sort=False)["v"]
        .transform("sum")
        .values.astype(np.float32)
    )

    aux = np.column_stack([ts_norm, lt_code, cong_sc_mean, fault_sc_sum])
    assert aux.shape == (len(train_fe), 4), f"aux shape {aux.shape}"
    assert not np.isnan(aux).any(), "aux has NaN"
    return aux


# ==================================================================
# Meta-learner training (fold-safe, with checkpointing)
# ==================================================================
def train_meta_lgb(
    stack: np.ndarray,
    y: np.ndarray,
    y_log: np.ndarray,
    fold_ids: np.ndarray,
    variant_name: str,
) -> dict:
    import lightgbm as lgb

    print(f"\n--- variant {variant_name} ({stack.shape[1]} feats) ---")
    oof = np.zeros(len(y), dtype=np.float64)
    fold_rows = []

    for f in range(N_SPLITS):
        tr_idx = np.where(fold_ids != f)[0]
        val_idx = np.where(fold_ids == f)[0]
        t_f = time.time()

        model = lgb.LGBMRegressor(
            objective="mae",
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=15,
            max_depth=4,
            min_child_samples=100,
            random_state=SEED,
            verbose=-1,
            n_jobs=-1,
        )
        model.fit(
            stack[tr_idx], y_log[tr_idx],
            eval_set=[(stack[val_idx], y_log[val_idx])],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        pred = np.clip(np.expm1(model.predict(stack[val_idx])), 0, None)
        oof[val_idx] = pred

        fold_mae = mean_absolute_error(y[val_idx], pred)
        best_it = int(model.best_iteration_ or 500)
        fold_rows.append({
            "variant": variant_name, "fold": f,
            "n_train": len(tr_idx), "n_val": len(val_idx),
            "mae": float(fold_mae), "best_iter": best_it,
            "seconds": round(time.time() - t_f, 1),
        })
        print(
            f"  fold {f}: mae={fold_mae:.5f} best_it={best_it} "
            f"({time.time()-t_f:.0f}s)"
        )
        np.save(OUT / f"meta_aux_oof_{variant_name}.npy", oof)

    oof_mae = mean_absolute_error(y, oof)
    print(f"  OVERALL = {oof_mae:.5f}")
    return {"oof_mae": float(oof_mae), "fold_rows": fold_rows, "oof": oof}


# ==================================================================
# Main
# ==================================================================
def main():
    t0 = time.time()
    print("=" * 64)
    print("Auxiliary-Aware Meta-Stacking Phase A")
    print("=" * 64)

    # --- Step 0: Load base OOFs ---
    print("\n[Step 0] Loading 33 base OOFs ...")
    all_oofs = load_33_base_oofs()
    print(f"  loaded {len(all_oofs)} base models")
    for name, oof in all_oofs.items():
        assert len(oof) == 250000, f"{name} len={len(oof)}"

    # load target
    train = pd.read_csv(ROOT / "train.csv").sort_values(
        ["layout_id", "scenario_id"]
    ).reset_index(drop=True)
    y = train[TARGET].values.astype(np.float64)
    y_log = np.log1p(y)

    stack_A = build_stack(all_oofs)
    print(f"  stack_A shape = {stack_A.shape}")

    fold_ids = np.load(FOLD_IDX)
    assert len(fold_ids) == 250000

    # --- baseline sanity check ---
    with open(MEGA33_PKL, "rb") as f:
        mega = pickle.load(f)
    mega33_oof = np.asarray(mega["meta_avg_oof"])
    baseline_mae = mean_absolute_error(y, mega33_oof)
    print(f"  mega33 baseline MAE = {baseline_mae:.5f}")
    assert abs(baseline_mae - EXPECTED_BASELINE) < 0.005, (
        f"ABORT: baseline drift! {baseline_mae:.5f} vs {EXPECTED_BASELINE}"
    )

    # --- Step 1: variant A (no aux) ---
    res_A = train_meta_lgb(stack_A, y, y_log, fold_ids, "A_no_aux")

    # --- Step 2: build auxiliary features ---
    print("\n[Step 2] Building auxiliary features ...")
    with open(FE_CACHE, "rb") as f:
        train_fe = pickle.load(f)["train_fe"]
    aux = build_aux(train_fe)
    print(f"  aux shape = {aux.shape}")
    stack_B = np.column_stack([stack_A, aux])
    print(f"  stack_B shape = {stack_B.shape}")

    # --- Step 3: variant B (with aux) ---
    res_B = train_meta_lgb(stack_B, y, y_log, fold_ids, "B_with_aux")

    # --- Step 4: metrics + fold CSV ---
    all_fold_rows = res_A["fold_rows"] + res_B["fold_rows"]
    pd.DataFrame(all_fold_rows).to_csv(
        OUT / "meta_aux_fold_metrics.csv", index=False
    )

    delta = res_B["oof_mae"] - res_A["oof_mae"]
    delta_vs_mega33 = res_B["oof_mae"] - baseline_mae

    verdict = (
        "GO" if delta <= -0.02
        else ("WEAK_GO" if delta < 0 else "NO_GO")
    )

    summary = {
        "variant_A_oof_mae": res_A["oof_mae"],
        "variant_B_oof_mae": res_B["oof_mae"],
        "delta_B_minus_A": delta,
        "baseline_mega33_avg": baseline_mae,
        "delta_B_vs_mega33": delta_vs_mega33,
        "n_base_models": len(all_oofs),
        "n_aux_features": 4,
        "stack_A_shape": list(stack_A.shape),
        "stack_B_shape": list(stack_B.shape),
        "verdict": verdict,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (OUT / "meta_aux_phase_a_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n{'='*64}")
    print("PHASE A SUMMARY")
    print(f"{'='*64}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n>>> VERDICT: {verdict} <<<")
    if verdict in ("GO", "WEAK_GO"):
        print(
            f"  A ({res_A['oof_mae']:.5f}) -> B ({res_B['oof_mae']:.5f}) "
            f"delta={delta:+.5f}"
        )

    (OUT / "checkpoint_meta_aux.done").write_text(
        f"verdict={verdict} delta={delta:+.5f}", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
