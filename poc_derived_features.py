"""
POC A: 4 카테고리 파생변수 → v23 149 + new features → LGB_Huber 재학습 → corr 검증

Categories:
  A. Scenario-shape (argmax ts, peak magnitude, early/late dominance)
  D. Cross-feature ratios/products (blocked×collision, battery_std×util, etc.)
  E. Time-since-spike (ts since fault/blocked/collision spike within scenario)
  F. Within-scenario variance/skew/kurt of key columns
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "poc_derived")
os.makedirs(OUT, exist_ok=True)


def build_derived(df):
    """Build derived features from raw columns of df (must have layout_id, scenario_id + raw cols)."""
    # Ensure sorted by (layout_id, scenario_id, row order)
    df = df.reset_index(drop=True).copy()
    df["_ts_idx"] = df.groupby(["layout_id", "scenario_id"]).cumcount()

    grp = df.groupby(["layout_id", "scenario_id"])
    out = pd.DataFrame(index=df.index)

    # ─── A. Scenario-shape features ────────────────────────────
    # argmax position of key signals within scenario
    for col in ["order_inflow_15m", "congestion_score", "robot_utilization", "blocked_path_15m"]:
        if col in df.columns:
            # per-scenario argmax position (0-24)
            argmax_pos = grp[col].transform(lambda s: s.values.argmax() if s.notna().any() else np.nan)
            out[f"scA_argmax_ts_{col}"] = argmax_pos
            # peak magnitude ratio (max / mean) within scenario
            sc_max = grp[col].transform("max")
            sc_mean = grp[col].transform("mean")
            out[f"scA_peak_ratio_{col}"] = np.where(sc_mean > 0, sc_max / (sc_mean + 1e-6), 0)
    # early vs late dominance (mean first 8 vs last 8)
    for col in ["order_inflow_15m", "congestion_score"]:
        if col in df.columns:
            def first_vs_last(s):
                vals = s.values
                if len(vals) < 16:
                    return np.nan
                first8 = np.nanmean(vals[:8])
                last8 = np.nanmean(vals[-8:])
                return first8 - last8
            out[f"scA_early_late_diff_{col}"] = grp[col].transform(first_vs_last)

    # ─── D. Cross-feature ratios/products ───────────────────────
    if "blocked_path_15m" in df.columns and "near_collision_15m" in df.columns:
        out["scD_blocked_x_collision"] = df["blocked_path_15m"] * df["near_collision_15m"]
    if "battery_std" in df.columns and "robot_utilization" in df.columns:
        out["scD_battery_std_x_util"] = df["battery_std"] * df["robot_utilization"]
    if "order_inflow_15m" in df.columns and "robot_active" in df.columns:
        out["scD_order_per_robot"] = df["order_inflow_15m"] / (df["robot_active"].clip(1, None))
    if "congestion_score" in df.columns and "robot_utilization" in df.columns:
        out["scD_cong_x_util"] = df["congestion_score"] * df["robot_utilization"]
    if "fault_count_15m" in df.columns and "robot_active" in df.columns:
        out["scD_fault_per_robot"] = df["fault_count_15m"] / (df["robot_active"].clip(1, None))
    if "low_battery_ratio" in df.columns and "charge_queue_length" in df.columns:
        out["scD_batt_x_charge_q"] = df["low_battery_ratio"] * df["charge_queue_length"]

    # ─── E. Time since spike ────────────────────────────────────
    # For each scenario, for each row: how many ts since last "spike" (col > scenario_mean × 2)
    for col in ["fault_count_15m", "blocked_path_15m", "near_collision_15m"]:
        if col in df.columns:
            sc_mean = grp[col].transform("mean")
            is_spike = (df[col] > sc_mean * 2).astype(int)
            # within scenario cumulative: for each row, ts since last spike (inclusive)
            # Implement: reset counter when spike; else increment
            def ts_since_spike(s):
                out_arr = np.full(len(s), 25, dtype=np.int32)  # initialize high
                last = -1
                for i, v in enumerate(s.values):
                    if v == 1:
                        last = i
                    if last >= 0:
                        out_arr[i] = i - last
                return pd.Series(out_arr, index=s.index)
            # Apply within scenario
            out[f"scE_ts_since_{col}_spike"] = is_spike.groupby([df["layout_id"], df["scenario_id"]]).transform(ts_since_spike)

    # ─── F. Within-scenario higher moments ──────────────────────
    for col in ["order_inflow_15m", "congestion_score", "battery_mean", "robot_utilization"]:
        if col in df.columns:
            def _skew(s):
                v = s.dropna().values
                return float(skew(v)) if len(v) > 3 else 0.0
            def _kurt(s):
                v = s.dropna().values
                return float(kurtosis(v)) if len(v) > 3 else 0.0
            out[f"scF_var_{col}"] = grp[col].transform("var")
            out[f"scF_skew_{col}"] = grp[col].transform(_skew)
            out[f"scF_kurt_{col}"] = grp[col].transform(_kurt)

    # Fill NaN with 0 for new columns (conservative)
    out = out.fillna(0).astype(np.float64)
    return out


def main():
    print("=" * 60, flush=True)
    print("POC A: Derived features (4 categories)", flush=True)
    print("=" * 60, flush=True)

    # Load raw data (we need the raw columns to derive from)
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

    # Load v23 FE cache (149 base features aligned to sorted train)
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)

    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # Sanity: train_fe order aligns with sorted train
    assert (train_fe["ID"].values == train["ID"].values).all(), "train_fe order mismatch"

    # Build derived features on train and test raw (we use train/test raw for scenario-aggregations)
    print("Building derived features on train...", flush=True)
    train_derived = build_derived(train)
    print(f"  shape: {train_derived.shape}", flush=True)
    print(f"  cols: {list(train_derived.columns)[:10]} ...", flush=True)
    print("Building derived features on test...", flush=True)
    test_derived = build_derived(test)
    print(f"  shape: {test_derived.shape}", flush=True)

    # Stack: v23 + derived
    X = np.hstack([train_fe[feat_cols].values, train_derived.values])
    X_test = np.hstack([test_fe[feat_cols].values, test_derived.values])
    print(f"Combined X shape: {X.shape}", flush=True)

    # Existing v23s42_LGB_Huber OOF for comparison
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"\nOld v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)

    # Train new LGB_Huber
    print("\nTraining LGB_Huber on 149 + new features (5-fold)...", flush=True)
    new_oof = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        model = lgb.LGBMRegressor(
            objective="huber", n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X[tr_mask], y_log[tr_mask],
            eval_set=[(X[val_mask], y_log[val_mask])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        new_oof[val_mask] = np.expm1(model.predict(X[val_mask]))
        test_preds.append(np.expm1(model.predict(X_test)))
        fold_mae = np.mean(np.abs(new_oof[val_mask] - y[val_mask]))
        print(f"  fold {f}: best_iter={model.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    new_test = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))
    corr = float(np.corrcoef(y - old_oof, y - new_oof)[0, 1])

    print(f"\n=== Comparison ===", flush=True)
    print(f"Old OOF MAE:   {old_mae:.5f}", flush=True)
    print(f"New OOF MAE:   {new_mae:.5f}", flush=True)
    print(f"Delta:         {new_mae - old_mae:+.5f}", flush=True)
    print(f"residual_corr: {corr:.5f}", flush=True)

    # Kill gates
    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95
    print("\nKill gates:", flush=True)
    print(f"  MAE hurt > 0.05: {hurt}", flush=True)
    print(f"  residual_corr ≥ 0.95: {too_similar}", flush=True)

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae,
        delta=new_mae - old_mae, corr=corr,
        n_derived=int(train_derived.shape[1]),
        derived_cols=list(train_derived.columns),
        gates=dict(hurt=bool(hurt), too_similar=bool(too_similar)),
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if hurt or too_similar:
        print("\nVERDICT: NO_GO", flush=True)
    else:
        print("\nVERDICT: PROCEED — test blend contribution on mega33", flush=True)


if __name__ == "__main__":
    main()
