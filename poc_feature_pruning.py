"""
POC: Importance-based feature pruning on v23 LGB_Huber (seed 42).

Steps:
  1. Fit v23 LGB_Huber on ALL 149 features one fold to extract importance
     (or reuse existing v23 importance if available).
  2. Rank features by gain importance, drop bottom-30.
  3. Retrain LGB_Huber with 119 features, 5-fold GroupKFold.
  4. Compare OOF MAE and residual_corr with existing v23s42_LGB_Huber base.

Kill gate:
  - If new base OOF MAE >= existing + 0.05: pruning hurt too much, abort
  - If residual_corr >= 0.95: new base is redundant, no blend value
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
OUT = os.path.join(ROOT, "results", "poc_prune")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("POC: v23 LGB_Huber importance-based pruning", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
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

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values
    print(f"Features: {len(feat_cols)}, train rows: {len(y)}", flush=True)

    # Existing v23s42_LGB_Huber OOF for comparison
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"Existing v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)

    # Step 1: Quick importance using single full fold
    print("\nStep 1: feature importance extraction (full train / no val)...", flush=True)
    # Use one quick fit to get gain importance
    m0 = lgb.LGBMRegressor(
        objective="huber", n_estimators=500, learning_rate=0.05,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    m0.fit(X, y_log)
    importance = m0.booster_.feature_importance(importance_type="gain")
    imp_df = pd.DataFrame({"feature": feat_cols, "gain": importance}).sort_values("gain", ascending=False).reset_index(drop=True)
    print("Top-10 features by gain:")
    print(imp_df.head(10).to_string(index=False), flush=True)
    print(f"\nBottom-10 features:")
    print(imp_df.tail(10).to_string(index=False), flush=True)

    # Drop bottom-30
    n_drop = 30
    keep_features = imp_df.iloc[:-n_drop]["feature"].tolist()
    dropped = imp_df.iloc[-n_drop:]["feature"].tolist()
    print(f"\nKeeping {len(keep_features)} features (dropping {n_drop})", flush=True)
    print(f"Dropped: {dropped[:10]} ... {dropped[-5:]}", flush=True)

    keep_idx = [feat_cols.index(c) for c in keep_features]
    X_pruned = X[:, keep_idx]
    X_test_pruned = X_test[:, keep_idx]

    # Step 2: 5-fold GroupKFold retrain on 119 features
    print("\nStep 2: 5-fold retrain on pruned features...", flush=True)
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
            X_pruned[tr_mask], y_log[tr_mask],
            eval_set=[(X_pruned[val_mask], y_log[val_mask])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        new_oof[val_mask] = np.expm1(model.predict(X_pruned[val_mask]))
        test_preds.append(np.expm1(model.predict(X_test_pruned)))
        print(f"  fold {f}: best_iter={model.best_iteration_} val MAE={np.mean(np.abs(np.expm1(model.predict(X_pruned[val_mask])) - y[val_mask])):.4f}", flush=True)
    new_test = np.mean(np.stack(test_preds, axis=0), axis=0)
    new_mae = float(mean_absolute_error(y, new_oof))

    # Compare
    r_old = y - old_oof
    r_new = y - new_oof
    corr = float(np.corrcoef(r_old, r_new)[0, 1])
    print(f"\n=== Comparison ===", flush=True)
    print(f"Old v23s42_LGB_Huber OOF MAE: {old_mae:.5f}", flush=True)
    print(f"New (119-feature) OOF MAE:    {new_mae:.5f}", flush=True)
    print(f"Delta:                         {new_mae - old_mae:+.5f}", flush=True)
    print(f"residual_corr(old, new):      {corr:.5f}", flush=True)

    # Kill gate
    hurt_too_much = (new_mae > old_mae + 0.05)
    too_similar = (corr >= 0.95)
    print("\nKill gates:", flush=True)
    print(f"  MAE hurt > 0.05:       {hurt_too_much} (delta {new_mae-old_mae:+.4f})", flush=True)
    print(f"  residual_corr ≥ 0.95:  {too_similar} (corr {corr:.4f})", flush=True)

    np.save(os.path.join(OUT, "pruned_oof.npy"), new_oof)
    np.save(os.path.join(OUT, "pruned_test.npy"), new_test)
    summary = dict(
        old_mae=old_mae,
        new_mae=new_mae,
        delta=new_mae - old_mae,
        residual_corr=corr,
        kept_features=keep_features,
        dropped_features=dropped,
        kill_gate_hurt=bool(hurt_too_much),
        kill_gate_corr=bool(too_similar),
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    imp_df.to_csv(os.path.join(OUT, "importance.csv"), index=False)
    print(f"\nSaved to {OUT}/", flush=True)

    if hurt_too_much or too_similar:
        print("\nVERDICT: NO_GO — POC base offers no blend value.", flush=True)
    else:
        print("\nVERDICT: PROCEED — test blend contribution with mega33 baseline.", flush=True)


if __name__ == "__main__":
    main()
