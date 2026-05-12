"""
Scale up: Train 7 additional seeds of v23 LGB_Huber.
Combined with existing 3 seeds → 10-seed super base.

Compare:
  - Single seed 8.61
  - 3-seed avg 8.59
  - 10-seed avg (estimated 8.58-8.58)
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
OUT = os.path.join(ROOT, "results", "scale_seeds")
os.makedirs(OUT, exist_ok=True)

NEW_SEEDS = [1000, 2001, 3003, 4004, 5005, 6006, 7007]


def train_one_seed(X_train, X_test, y_log, y, fold_idx, seed):
    oof = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        m = lgb.LGBMRegressor(
            objective="huber", n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(X_train[tr_mask], y_log[tr_mask],
              eval_set=[(X_train[val_mask], y_log[val_mask])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        oof[val_mask] = np.expm1(m.predict(X_train[val_mask]))
        test_preds.append(np.expm1(m.predict(X_test)))
        fold_mae = np.mean(np.abs(oof[val_mask] - y[val_mask]))
        print(f"  seed {seed} fold {f}: best_iter={m.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    return oof, test_pred


def main():
    print("=" * 60, flush=True)
    print(f"Scale up v23 LGB_Huber: 7 new seeds", flush=True)
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

    X_train = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values
    print(f"X_train shape: {X_train.shape}", flush=True)

    # Existing 3 seeds
    existing_oofs = []
    existing_tests = []
    for s in [42, 123, 2024]:
        v = pickle.load(open(os.path.join(ROOT, "results", f"v23_seed{s}.pkl"), "rb"))
        existing_oofs.append(v["oofs"]["LGB_Huber"])
        existing_tests.append(v["tests"]["LGB_Huber"])
    print(f"\nExisting 3 seeds loaded", flush=True)
    print(f"  Single-seed MAE: {[f'{mean_absolute_error(y, o):.5f}' for o in existing_oofs]}", flush=True)
    print(f"  3-seed avg MAE: {mean_absolute_error(y, np.mean(existing_oofs, axis=0)):.5f}", flush=True)

    # Train new seeds
    new_oofs = []
    new_tests = []
    for i, seed in enumerate(NEW_SEEDS):
        print(f"\n--- Training new seed {seed} ({i+1}/{len(NEW_SEEDS)}) ---", flush=True)
        oof, test = train_one_seed(X_train, X_test, y_log, y, fold_idx, seed)
        new_oofs.append(oof)
        new_tests.append(test)
        mae = mean_absolute_error(y, oof)
        print(f"  seed {seed}: OOF MAE = {mae:.5f}", flush=True)
        np.save(os.path.join(OUT, f"seed{seed}_oof.npy"), oof)
        np.save(os.path.join(OUT, f"seed{seed}_test.npy"), test)

    all_oofs = existing_oofs + new_oofs
    all_tests = existing_tests + new_tests
    all_avg_oof = np.mean(all_oofs, axis=0)
    all_avg_test = np.mean(all_tests, axis=0)
    mae_10seed = mean_absolute_error(y, all_avg_oof)
    print(f"\n=== 10-seed average ===", flush=True)
    print(f"  OOF MAE: {mae_10seed:.5f}", flush=True)
    print(f"  vs single seed ~8.61: gain ~{8.61 - mae_10seed:+.5f}", flush=True)
    print(f"  vs 3-seed avg ~8.59: gain ~{8.59 - mae_10seed:+.5f}", flush=True)

    # Save super base
    np.save(os.path.join(OUT, "super_lgb_huber_oof.npy"), all_avg_oof)
    np.save(os.path.join(OUT, "super_lgb_huber_test.npy"), all_avg_test)

    # Now test effect on mega33 (approximate — treat super as additional base)
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]
    mega_mae = float(mean_absolute_error(y, mega_oof))

    corr_super = float(np.corrcoef(y - mega_oof, y - all_avg_oof)[0, 1])
    print(f"\nSuper OOF corr vs mega33: {corr_super:.4f}", flush=True)

    # Simple 2-way blend
    best_w = 0; best_m = mega_mae
    for w in np.linspace(0, 0.5, 51):
        pred = (1 - w) * mega_oof + w * all_avg_oof
        m = float(np.mean(np.abs(y - pred)))
        if m < best_m: best_m = m; best_w = w
    print(f"Blend mega33 + super: w={best_w:.3f}, MAE={best_m:.5f}, delta={best_m - mega_mae:+.5f}", flush=True)

    summary = dict(
        single_seed_mae_sample=float(mean_absolute_error(y, existing_oofs[0])),
        three_seed_mae=float(mean_absolute_error(y, np.mean(existing_oofs, axis=0))),
        ten_seed_mae=float(mae_10seed),
        corr_vs_mega=corr_super,
        blend_w=float(best_w), blend_mae=float(best_m),
        blend_delta=float(best_m - mega_mae),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
