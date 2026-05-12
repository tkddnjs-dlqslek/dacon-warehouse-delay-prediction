"""
Approach 3: Static × Dynamic interaction features.

Static: layout_info 15 columns (floor_area_sqm, robot_total, charger_count, etc.)
Dynamic: top 10 by importance in error_analysis / mega33 usage

Generate all 15 × 10 × 2 (product + ratio) = 300 pairwise features.
Select top-30 by |corr(new_feat, mega33_residual)| (capturing new signal).
Add to v23, retrain LGB_Huber.
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
OUT = os.path.join(ROOT, "results", "approach_3_sd")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Approach 3: Static × Dynamic interaction", flush=True)
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

    # Mega33 residual (for feature selection target)
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    resid = y - mega_oof

    # Static features (layout_info)
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    static_cols = [c for c in layout_info.columns if c not in ["layout_id", "layout_type"]]
    train_li = train[["layout_id"]].merge(layout_info, on="layout_id", how="left")
    test_li = test[["layout_id"]].merge(layout_info, on="layout_id", how="left")
    for c in static_cols:
        train_li[c] = train_li[c].fillna(train_li[c].median())
        test_li[c] = test_li[c].fillna(train_li[c].median())

    # Dynamic features: top-10 from error analysis (or use top-10 by |rho(y)|)
    dynamic_candidates = [
        "order_inflow_15m",
        "pack_utilization",
        "robot_utilization",
        "order_per_pack_station",
        "congestion_score",
        "blocked_path_15m",
        "battery_mean",
        "fault_count_15m",
        "sku_concentration",
        "near_collision_15m",
    ]
    dynamic_cols = [c for c in dynamic_candidates if c in train.columns]
    print(f"Static: {len(static_cols)}, Dynamic: {len(dynamic_cols)}", flush=True)

    # Fill dynamic NaN with column median (train-based)
    for c in dynamic_cols:
        med = train[c].median()
        train[c] = train[c].fillna(med)
        test[c] = test[c].fillna(med)

    # ─── Build all pairwise interactions ───
    print("\nBuilding all 15 × 10 × 2 = 300 interactions...", flush=True)
    interaction_train = {}
    interaction_test = {}
    for s in static_cols:
        sv_tr = train_li[s].values.astype(np.float64)
        sv_te = test_li[s].values.astype(np.float64)
        for dcol in dynamic_cols:
            dv_tr = train[dcol].values.astype(np.float64)
            dv_te = test[dcol].values.astype(np.float64)
            # product
            pname = f"sd_{s}_x_{dcol}"
            interaction_train[pname] = sv_tr * dv_tr
            interaction_test[pname] = sv_te * dv_te
            # ratio: dynamic / static (avoid div by zero)
            rname = f"sd_{dcol}_per_{s}"
            svr = np.where(sv_tr > 0, sv_tr, 1)
            svr_te = np.where(sv_te > 0, sv_te, 1)
            interaction_train[rname] = dv_tr / svr
            interaction_test[rname] = dv_te / svr_te
    print(f"Total interactions: {len(interaction_train)}", flush=True)

    # ─── Select top-30 by |corr(new_feat, residual)| ───
    print("\nSelecting top-30 by |corr(new_feat, mega33_residual)|...", flush=True)
    scores = []
    for name, arr in interaction_train.items():
        mask = np.isfinite(arr)
        if mask.sum() < 100 or arr[mask].std() == 0:
            continue
        rho_resid = float(np.corrcoef(arr[mask], resid[mask])[0, 1])
        rho_y = float(np.corrcoef(arr[mask], y[mask])[0, 1])
        scores.append((name, rho_resid, rho_y, abs(rho_resid)))
    scores.sort(key=lambda r: -r[3])
    print("\nTop 20 by |rho(resid)|:", flush=True)
    for name, rr, ry, _ in scores[:20]:
        print(f"  {name:<55s} rho_resid={rr:+.4f}  rho_y={ry:+.4f}", flush=True)

    top_names = [s[0] for s in scores[:30]]
    print(f"\nSelected top 30 (range of rho_resid: {scores[0][1]:.3f} to {scores[29][1]:.3f})", flush=True)

    # ─── Build matrices ───
    new_train = np.column_stack([interaction_train[n] for n in top_names])
    new_test = np.column_stack([interaction_test[n] for n in top_names])
    print(f"New features matrix: {new_train.shape}", flush=True)

    # Stack
    X = np.hstack([train_fe[feat_cols].values, new_train])
    X_test = np.hstack([test_fe[feat_cols].values, new_test])
    print(f"Combined X: {X.shape}", flush=True)

    # Train LGB_Huber
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"\nReference v23 LGB_Huber OOF: {old_mae:.5f}", flush=True)

    print("\n=== Training LGB_Huber on v23 + 30 S×D interactions ===", flush=True)
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
        model.fit(X[tr_mask], y_log[tr_mask],
                  eval_set=[(X[val_mask], y_log[val_mask])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
        new_oof[val_mask] = np.expm1(model.predict(X[val_mask]))
        test_preds.append(np.expm1(model.predict(X_test)))
        fold_mae = np.mean(np.abs(new_oof[val_mask] - y[val_mask]))
        print(f"  fold {f}: best_iter={model.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    new_test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))
    corr = float(np.corrcoef(y - old_oof, y - new_oof)[0, 1])
    corr_mega = float(np.corrcoef(y - mega_oof, y - new_oof)[0, 1])
    mega_mae = float(mean_absolute_error(y, mega_oof))

    print(f"\n=== Results ===", flush=True)
    print(f"Old OOF: {old_mae:.5f}", flush=True)
    print(f"New OOF: {new_mae:.5f}  delta={new_mae-old_mae:+.5f}", flush=True)
    print(f"corr (v23 old): {corr:.5f}", flush=True)
    print(f"corr (mega33):  {corr_mega:.5f}", flush=True)

    best_mae = mega_mae; best_w = 0
    for w in np.linspace(0, 0.3, 31):
        pred = (1 - w) * mega_oof + w * new_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_mae:
            best_mae = m; best_w = w
    print(f"Blend: w={best_w:.3f}, MAE={best_mae:.5f}, delta={best_mae-mega_mae:+.5f}", flush=True)

    hurt = new_mae > old_mae + 0.05
    too_similar = corr >= 0.95

    np.save(os.path.join(OUT, "oof.npy"), new_oof)
    np.save(os.path.join(OUT, "test.npy"), new_test_pred)
    summary = dict(
        old_mae=old_mae, new_mae=new_mae, delta=new_mae-old_mae,
        corr=corr, corr_mega=corr_mega,
        blend_w=best_w, blend_mae=best_mae, blend_delta=best_mae-mega_mae,
        top_features=[dict(name=s[0], rho_resid=s[1], rho_y=s[2]) for s in scores[:30]],
        gates=dict(hurt=hurt, too_similar=too_similar),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nVERDICT: {'NO_GO' if (hurt or too_similar) else 'PROCEED'}", flush=True)


if __name__ == "__main__":
    main()
