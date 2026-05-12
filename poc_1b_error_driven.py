"""
POC 1b: Test error-driven hypothesis features on v23 LGB_Huber.

Based on error_analysis.py findings:
  - 40.8% of MAE from 5% worst rows (all under-predicted)
  - Top shift: order_per_pack_station (AUC 0.818), pack_util, robot_util
  - 90.1% of worst rows in 7.6% "pathological" scenarios
  - Layout pathology: WH_073, WH_217, etc.

Add ~15-20 specific features, retrain single v23 LGB_Huber 5-fold, check corr.
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
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "poc_1b")
os.makedirs(OUT, exist_ok=True)


def build_error_features(df, raw_df):
    """Build features based on error analysis findings."""
    # df is train_fe (v23 features), raw_df is raw train.csv
    # Return a DataFrame of new features aligned to df.index
    out = pd.DataFrame(index=df.index)

    # ─── 1. Saturation indicators ───
    pack_u = df.get("pack_utilization", pd.Series(np.nan, index=df.index)).fillna(0)
    robot_u = df.get("robot_utilization", pd.Series(np.nan, index=df.index)).fillna(0)
    order_ps = df.get("order_per_pack_station", pd.Series(np.nan, index=df.index)).fillna(0)

    out["err_sat_hard"] = ((pack_u > 0.7) & (robot_u > 0.6) & (order_ps > 20)).astype(int)
    out["err_sat_soft"] = (pack_u.clip(0, 1) + robot_u.clip(0, 1) + (order_ps / 30).clip(0, 1)) / 3
    out["err_sat_any"] = ((pack_u > 0.7) | (robot_u > 0.6) | (order_ps > 20)).astype(int)

    # ─── 2. Scenario-level rollups (dynamic features only — safe) ───
    group_keys = ["layout_id", "scenario_id"]
    for col in ["order_inflow_15m", "pack_utilization", "robot_utilization",
                "order_per_pack_station", "congestion_score"]:
        if col not in raw_df.columns and col not in df.columns:
            continue
        src = raw_df[col] if col in raw_df.columns else df[col]
        grp = src.groupby([raw_df["layout_id"], raw_df["scenario_id"]])
        out[f"err_sc_max_{col}"] = grp.transform("max")
        out[f"err_sc_q90_{col}"] = grp.transform(lambda s: np.nanquantile(s.values, 0.9) if s.notna().any() else np.nan)

    # ─── 3. Non-linear transforms of top features ───
    out["err_order_ps_sq"] = order_ps ** 2
    out["err_order_ps_log"] = np.log1p(order_ps.clip(0, None))
    out["err_pack_util_cubed"] = pack_u ** 3
    out["err_order_x_pack"] = order_ps * pack_u
    out["err_order_x_robot"] = order_ps * robot_u

    # ─── 4. Threshold indicators ───
    out["err_high_order"] = (order_ps > 20).astype(int)
    out["err_high_pack"] = (pack_u > 0.7).astype(int)
    out["err_high_robot"] = (robot_u > 0.6).astype(int)

    # ─── 5. Capacity stress: order_inflow / pack_station_count ───
    # Need layout_info
    li = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))[["layout_id", "pack_station_count", "charger_count", "floor_area_sqm"]]
    li_map = raw_df[["layout_id"]].merge(li, on="layout_id", how="left")
    order_in = raw_df["order_inflow_15m"].fillna(0).values
    ps_count = li_map["pack_station_count"].fillna(1).values.clip(1, None)
    charger = li_map["charger_count"].fillna(1).values.clip(1, None)
    area = li_map["floor_area_sqm"].fillna(1).values.clip(1, None)

    out["err_stress_order_per_station"] = order_in / ps_count  # variant of order_per_pack_station
    out["err_stress_per_area"] = order_in / area
    out["err_stress_per_charger"] = order_in / charger

    # Fill na
    out = out.fillna(0).astype(np.float64)
    print(f"Built {out.shape[1]} error-driven features", flush=True)
    return out


def main():
    print("=" * 60, flush=True)
    print("POC 1b: Error-driven features test", flush=True)
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

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # Build error features
    train_err = build_error_features(train_fe, train)
    test_err = build_error_features(test_fe, test)

    err_cols = list(train_err.columns)
    print(f"\nError features:", flush=True)
    for c in err_cols:
        print(f"  {c}", flush=True)

    # Check correlation with y
    print("\n=== Error feature |rho| with y (train) ===", flush=True)
    for c in err_cols:
        x = train_err[c].values
        rho = float(np.corrcoef(x, y)[0, 1]) if x.std() > 0 else 0.0
        print(f"  {c:<45s} rho={rho:+.4f}", flush=True)

    # Stack v23 + error features
    X = np.hstack([train_fe[feat_cols].values, train_err.values])
    X_test = np.hstack([test_fe[feat_cols].values, test_err.values])
    print(f"\nCombined X shape: {X.shape}", flush=True)

    # Existing v23s42_LGB_Huber baseline
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"\nReference v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)

    # Train new LGB_Huber 5-fold
    print("\n=== Training LGB_Huber on v23 + error features ===", flush=True)
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

    # Also vs mega33
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))
    corr_mega = float(np.corrcoef(y - mega_oof, y - new_oof)[0, 1])
    print(f"\nvs mega33 ({mega_mae:.5f}): corr={corr_mega:.5f}", flush=True)

    # Quick blend test
    best_mae = mega_mae
    best_w = 0
    for w in np.linspace(0, 0.3, 31):
        pred = (1 - w) * mega_oof + w * new_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m
            best_w = w
    print(f"Simple blend mega33+new: best w={best_w:.3f} MAE={best_mae:.5f} delta={best_mae-mega_mae:+.5f}", flush=True)

    # Worst bucket check: does new model help worst rows?
    resid_old = np.abs(y - mega_oof)
    thresh = np.quantile(resid_old, 0.95)
    worst_mask = resid_old >= thresh
    old_worst_mae = np.mean(np.abs(y[worst_mask] - mega_oof[worst_mask]))
    new_worst_mae = np.mean(np.abs(y[worst_mask] - new_oof[worst_mask]))
    print(f"\nWorst 5% rows (|resid_old| >= {thresh:.2f}):", flush=True)
    print(f"  mega33 MAE: {old_worst_mae:.3f}", flush=True)
    print(f"  new    MAE: {new_worst_mae:.3f}", flush=True)
    print(f"  delta:      {new_worst_mae-old_worst_mae:+.3f}", flush=True)

    # Kill gates
    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95
    print("\nKill gates:", flush=True)
    print(f"  single MAE hurt > 0.05: {hurt}", flush=True)
    print(f"  residual_corr >= 0.95:  {too_similar}", flush=True)

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae, delta=new_mae - old_mae,
        corr=corr, corr_mega=corr_mega,
        blend_w=best_w, blend_mae=best_mae, blend_delta=best_mae - mega_mae,
        worst5_old=float(old_worst_mae),
        worst5_new=float(new_worst_mae),
        n_features=len(err_cols),
        features=err_cols,
        gates=dict(hurt=bool(hurt), too_similar=bool(too_similar)),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if hurt or too_similar:
        print("\nVERDICT: NO_GO", flush=True)
    else:
        print("\nVERDICT: PROCEED — consider full ensemble integration", flush=True)


if __name__ == "__main__":
    main()
