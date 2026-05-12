"""
Paradigm 1 v0: Warehouse Delay Simulator.

Hand-crafted physics-based baseline prediction:
  y_sim = base + queue_waiting + fault_impact + congestion_impact + stress

Calibrated on train in log space (OLS).
Then:
  residual = y - y_sim  (or log-space equivalent)
  LGB trains on residual with v23 features.
  final_pred = y_sim + lgb_residual

Compare:
  - simulator alone OOF MAE (sanity: should be < 15)
  - residual distribution (should have reduced variance)
  - residual LGB corr with mega33 residual
  - final blend with mega33 / FIXED
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
from sklearn.linear_model import Ridge
from scipy.optimize import minimize
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "paradigm_1_sim_v0")
os.makedirs(OUT, exist_ok=True)


def build_physics_terms(df, layout_info):
    """Build physics-meaningful terms per row from raw 94 columns + layout_info."""
    df = df.copy()
    # merge layout_info
    df = df.merge(layout_info, on="layout_id", how="left", suffixes=("", "_li"))

    out = pd.DataFrame(index=df.index)

    # Rate variables (fill nan with 0 for safety)
    order_inflow = df["order_inflow_15m"].fillna(0).values
    robot_active = df["robot_active"].fillna(0).values.clip(1, None)
    pack_station = df["pack_station_count"].fillna(1).values.clip(1, None)
    robot_util = df["robot_utilization"].fillna(0).values.clip(0, 1)
    pack_util = df["pack_utilization"].fillna(0).values.clip(0, 1)
    congestion = df["congestion_score"].fillna(0).values
    fault_count = df["fault_count_15m"].fillna(0).values
    recovery_time = df["avg_recovery_time"].fillna(0).values
    blocked = df["blocked_path_15m"].fillna(0).values
    charger = df["charger_count"].fillna(1).values.clip(1, None)
    low_batt = df["low_battery_ratio"].fillna(0).values.clip(0, 1)
    near_coll = df["near_collision_15m"].fillna(0).values
    floor_area = df["floor_area_sqm"].fillna(1).values.clip(1, None)

    # Approximate rates
    # arrivals_per_min ~ order_inflow_15m / 15
    out["p_arrivals"] = order_inflow / 15.0
    # service rate per station, approximated as (1 - pack_util) effective slack
    # Or: pack_station_count capacity scaling
    out["p_capacity"] = pack_station * (1 - pack_util).clip(0.01, 1)

    # Rho (utilization ratio)
    rho = order_inflow / np.maximum(pack_station * 20, 1)  # 20 orders/ts/station heuristic
    rho = np.clip(rho, 0.01, 0.99)
    out["p_rho"] = rho

    # M/M/1 waiting time proxy: rho / (1 - rho)
    out["p_queue_wait"] = rho / (1 - rho)

    # rho squared proxy (quadratic queue growth)
    out["p_queue_sq"] = rho**2 / (1 - rho)

    # Robot saturation
    out["p_robot_sat"] = robot_util / (1 - robot_util + 0.01)

    # Order pressure
    out["p_order_per_robot"] = order_inflow / robot_active
    out["p_order_per_station"] = order_inflow / pack_station

    # Disruption terms
    out["p_fault_impact"] = fault_count * recovery_time
    out["p_blocked_rate"] = blocked / 15.0
    out["p_collision_rate"] = near_coll / 15.0

    # Congestion stress
    out["p_congestion"] = congestion
    out["p_congestion_util"] = congestion * robot_util

    # Battery bottleneck
    out["p_batt_stress"] = low_batt * robot_active / charger  # battery demand per charger

    # Capacity stress (order rate vs area)
    out["p_density"] = order_inflow / floor_area * 1000  # orders per 1000 sqm

    # Queue buildup proxy: arrivals > capacity
    out["p_overflow"] = (order_inflow - pack_station * 10).clip(0, None)

    # Static layout pressure
    out["p_static_capacity_inv"] = 1.0 / pack_station
    out["p_static_area_inv"] = 1.0 / floor_area * 1000

    return out


def main():
    print("=" * 60, flush=True)
    print("Paradigm 1 v0: Warehouse Simulator (physics baseline)", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    test = (
        pd.read_csv(os.path.join(ROOT, "test.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    y_log = np.log1p(np.clip(y, 0, None))
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    print("Building physics terms...", flush=True)
    phys_train = build_physics_terms(train, layout_info)
    phys_test = build_physics_terms(test, layout_info)
    print(f"  physics columns: {list(phys_train.columns)}", flush=True)
    print(f"  shape: {phys_train.shape}", flush=True)

    # Check rho distribution
    print(f"\np_rho: min={phys_train['p_rho'].min():.3f}, mean={phys_train['p_rho'].mean():.3f}, "
          f"max={phys_train['p_rho'].max():.3f}, p95={phys_train['p_rho'].quantile(0.95):.3f}", flush=True)
    print(f"p_queue_wait: mean={phys_train['p_queue_wait'].mean():.3f}, p95={phys_train['p_queue_wait'].quantile(0.95):.3f}", flush=True)

    # ─── Simulator: OLS fit in log space ───
    # y_log = sum(a_i * phys_i_log) + c + error
    # But some phys are already log-like (e.g., queue_wait spans 0 to inf)
    # Use log1p transform for stability
    phys_cols = list(phys_train.columns)
    Xp_train = np.log1p(phys_train[phys_cols].clip(0, None).values)
    Xp_test = np.log1p(phys_test[phys_cols].clip(0, None).values)

    # CV fit per fold
    sim_oof = np.zeros(len(y), dtype=np.float64)
    sim_test = np.zeros(len(test), dtype=np.float64)
    print("\n=== Simulator OLS fit per fold ===", flush=True)
    for f in range(5):
        val_mask = fold_idx == f; tr_mask = ~val_mask
        m = Ridge(alpha=1.0)
        m.fit(Xp_train[tr_mask], y_log[tr_mask])
        sim_oof[val_mask] = np.expm1(m.predict(Xp_train[val_mask]))
        sim_test += np.expm1(m.predict(Xp_test)) / 5
    sim_oof = np.clip(sim_oof, 0, None)
    sim_test = np.clip(sim_test, 0, None)
    sim_mae = mean_absolute_error(y, sim_oof)
    print(f"  Simulator alone OOF MAE: {sim_mae:.5f}", flush=True)

    # Kill gate 1: simulator too crude
    if sim_mae > 20:
        print(f"  KILL GATE: sim alone MAE > 20. Too crude. Abandon.", flush=True)
        return

    # ─── Residual analysis ───
    resid = y - sim_oof
    print(f"\nResidual stats: mean={resid.mean():.3f}, std={resid.std():.3f}, max_abs={np.max(np.abs(resid)):.1f}", flush=True)
    print(f"  y std: {y.std():.3f} → residual std reduction: {(1 - resid.std()/y.std())*100:.1f}%", flush=True)

    # Kill gate 2: residual looks like y (simulator not explaining)
    if resid.std() > 0.9 * y.std():
        print(f"  KILL GATE: residual std too close to y std. Simulator doesn't explain. Abandon.", flush=True)
        return

    # ─── LGB on residual ───
    # Load v23 features
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values

    print("\n=== LGB training on residual ===", flush=True)
    resid_oof = np.zeros(len(y), dtype=np.float64)
    resid_test = np.zeros(len(test), dtype=np.float64)
    for f in range(5):
        val_mask = fold_idx == f; tr_mask = ~val_mask
        model = lgb.LGBMRegressor(
            objective="mae", n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(X[tr_mask], resid[tr_mask],
                  eval_set=[(X[val_mask], resid[val_mask])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        resid_oof[val_mask] = model.predict(X[val_mask])
        resid_test += model.predict(X_test) / 5
        fold_resid_mae = np.mean(np.abs(resid_oof[val_mask] - resid[val_mask]))
        print(f"  fold {f}: best_iter={model.best_iteration_} resid fit MAE={fold_resid_mae:.4f}", flush=True)

    # Combined prediction
    combined_oof = sim_oof + resid_oof
    combined_test = sim_test + resid_test
    combined_oof = np.clip(combined_oof, 0, None)
    combined_test = np.clip(combined_test, 0, None)
    combined_mae = mean_absolute_error(y, combined_oof)
    print(f"\nCombined (sim + lgb_residual) OOF MAE: {combined_mae:.5f}", flush=True)

    # ─── Compare with mega33 ───
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]
    mega_mae = mean_absolute_error(y, mega_oof)
    print(f"mega33 OOF: {mega_mae:.5f}", flush=True)

    corr_mega = float(np.corrcoef(y - mega_oof, y - combined_oof)[0, 1])
    print(f"corr(mega33_resid, combined_resid): {corr_mega:.4f}", flush=True)

    # Kill gate 3: too similar
    if corr_mega >= 0.95:
        print(f"  KILL GATE: corr {corr_mega:.4f} >= 0.95. Same wall. Abandon.", flush=True)
        return

    # ─── Blend test ───
    print("\n=== Blend mega33 + combined ===", flush=True)
    best_w = 0; best_m = mega_mae
    for w in np.linspace(0, 0.5, 51):
        pred = (1 - w) * mega_oof + w * combined_oof
        m = mean_absolute_error(y, pred)
        if m < best_m: best_m = m; best_w = w
    print(f"  best w: {best_w:.3f}, MAE={best_m:.5f}, delta={best_m - mega_mae:+.5f}", flush=True)

    # Save
    np.save(os.path.join(OUT, "sim_oof.npy"), sim_oof)
    np.save(os.path.join(OUT, "sim_test.npy"), sim_test)
    np.save(os.path.join(OUT, "combined_oof.npy"), combined_oof)
    np.save(os.path.join(OUT, "combined_test.npy"), combined_test)
    summary = dict(
        sim_alone_mae=float(sim_mae),
        combined_mae=float(combined_mae),
        mega_mae=float(mega_mae),
        resid_std=float(resid.std()),
        y_std=float(y.std()),
        resid_std_reduction=float(1 - resid.std()/y.std()),
        corr_mega=corr_mega,
        blend_w=float(best_w),
        blend_mae=float(best_m),
        blend_delta=float(best_m - mega_mae),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Decision
    if best_m < mega_mae - 0.003:
        print(f"\nVERDICT: PROCEED to v1 (blend improvement {mega_mae - best_m:.5f})", flush=True)
    else:
        print(f"\nVERDICT: Marginal/NO_GO. v0 simulator crude. Consider v1 refinement.", flush=True)


if __name__ == "__main__":
    main()
