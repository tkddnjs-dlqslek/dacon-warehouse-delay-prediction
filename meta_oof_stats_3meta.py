"""
Phase A-1 extension: Test OOF-derived meta features on FULL 3-meta (LGB+XGB+Cat).

Compare:
  1. 3-meta on 33 OOFs only         → should give ~8.3989 (mega33 baseline)
  2. 3-meta on 33 OOFs + 4 stat cols → does it beat 8.3989?

If (2) < 8.3989 - 0.002: plug into FIXED as new meta_avg, re-optimize final_multi_blend weights.
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
from sklearn.model_selection import GroupKFold
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "meta_exp")
os.makedirs(OUT, exist_ok=True)


def load_33_bases():
    R = os.path.join(ROOT, "results")
    bases = {}
    bases_test = {}
    for seed in [42, 123, 2024]:
        s = pickle.load(open(f"{R}/v23_seed{seed}.pkl", "rb"))
        for name in ["LGB_Huber", "XGB", "CatBoost"]:
            bases[f"v23s{seed}_{name}"] = s["oofs"][name]
            bases_test[f"v23s{seed}_{name}"] = s["tests"][name]
    v24 = pickle.load(open(f"{R}/v24_final.pkl", "rb"))
    for name in v24["oofs"]:
        bases[f"v24_{name}"] = v24["oofs"][name]
        bases_test[f"v24_{name}"] = v24["tests"][name]
    v26 = pickle.load(open(f"{R}/v26_final.pkl", "rb"))
    for name in ["Tuned_Huber", "Tuned_sqrt", "Tuned_pow", "DART"]:
        bases[f"v26_{name}"] = v26["oofs"][name]
        bases_test[f"v26_{name}"] = v26["tests"][name]
    mlp1 = pickle.load(open(f"{R}/mlp_final.pkl", "rb"))
    mlp2 = pickle.load(open(f"{R}/mlp2_final.pkl", "rb"))
    cnn = pickle.load(open(f"{R}/cnn_final.pkl", "rb"))
    bases["mlp1"] = mlp1["mlp_oof"]; bases_test["mlp1"] = mlp1["mlp_test"]
    bases["mlp2"] = mlp2["mlp2_oof"]; bases_test["mlp2"] = mlp2["mlp2_test"]
    bases["cnn"] = cnn["cnn_oof"]; bases_test["cnn"] = cnn["cnn_test"]
    domain = pickle.load(open(f"{R}/domain_phase2.pkl", "rb"))
    for n in domain["oofs"]:
        bases[f"domain_{n}"] = domain["oofs"][n]
        bases_test[f"domain_{n}"] = domain["tests"][n]
    mlp_aug = pickle.load(open(f"{R}/mlp_aug_final.pkl", "rb"))
    bases["mlp_aug"] = mlp_aug["mlp_aug_oof"]
    bases_test["mlp_aug"] = mlp_aug["mlp_aug_test"]
    offset_p3 = pickle.load(open(f"{R}/offset_phase3.pkl", "rb"))
    for n, d in offset_p3.items():
        bases[f"offset_{n}"] = d["oof"]
        bases_test[f"offset_{n}"] = d["test"]
    na = pickle.load(open(f"{R}/neural_army.pkl", "rb"))
    for n in na:
        bases[f"na_{n}"] = na[n]["oof"]
        bases_test[f"na_{n}"] = na[n]["test"]
    return bases, bases_test


def fit_3meta(stack_train, stack_test, y, y_log, folds, tag=""):
    """3-meta = LGB+XGB+Cat averaged. Matches mega33 structure."""
    N_SPLITS = len(folds)
    oofs = {}
    tests = {}

    # LGB
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = lgb.LGBMRegressor(
            objective="mae", n_estimators=500, learning_rate=0.05,
            num_leaves=15, max_depth=4, min_child_samples=100,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oofs["lgb"] = np.clip(oof, 0, None)
    tests["lgb"] = np.clip(tpred, 0, None)
    print(f"  [{tag}] LGB: {mean_absolute_error(y, oofs['lgb']):.5f}", flush=True)

    # XGB
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = xgb.XGBRegressor(
            objective="reg:absoluteerror", n_estimators=500, learning_rate=0.05,
            max_depth=4, min_child_weight=100, random_state=42, n_jobs=-1,
            early_stopping_rounds=50, verbosity=0,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])], verbose=False)
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oofs["xgb"] = np.clip(oof, 0, None)
    tests["xgb"] = np.clip(tpred, 0, None)
    print(f"  [{tag}] XGB: {mean_absolute_error(y, oofs['xgb']):.5f}", flush=True)

    # CatBoost
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = CatBoostRegressor(
            loss_function="MAE", iterations=500, learning_rate=0.05,
            depth=4, random_seed=42, verbose=0,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=(stack_train[val_idx], y_log[val_idx]),
              early_stopping_rounds=50, verbose=0)
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oofs["cb"] = np.clip(oof, 0, None)
    tests["cb"] = np.clip(tpred, 0, None)
    print(f"  [{tag}] CB : {mean_absolute_error(y, oofs['cb']):.5f}", flush=True)

    # Average
    avg_oof = (oofs["lgb"] + oofs["xgb"] + oofs["cb"]) / 3
    avg_test = (tests["lgb"] + tests["xgb"] + tests["cb"]) / 3
    mae_avg = mean_absolute_error(y, avg_oof)
    print(f"  [{tag}] avg: {mae_avg:.5f}", flush=True)
    return avg_oof, avg_test, oofs, tests


def main():
    print("=" * 60, flush=True)
    print("3-meta with/without OOF stats features", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values
    y_log = np.log1p(y)
    groups = train["layout_id"].values

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    baseline_mae = mean_absolute_error(y, mega["meta_avg_oof"])
    print(f"Baseline mega33 3-meta: {baseline_mae:.5f}", flush=True)

    bases, bases_test = load_33_bases()
    names = list(bases.keys())
    print(f"Loaded {len(names)} bases", flush=True)

    stack_train = np.column_stack([np.log1p(np.clip(bases[n], 0, None)) for n in names])
    stack_test = np.column_stack([np.log1p(np.clip(bases_test[n], 0, None)) for n in names])

    # OOF stats features
    oof_matrix = np.column_stack([np.clip(bases[n], 0, None) for n in names])
    stats_train = np.log1p(np.column_stack([
        oof_matrix.std(axis=1),
        oof_matrix.mean(axis=1),
        oof_matrix.max(axis=1) - oof_matrix.min(axis=1),
        np.median(oof_matrix, axis=1),
    ]))
    test_matrix = np.column_stack([np.clip(bases_test[n], 0, None) for n in names])
    stats_test = np.log1p(np.column_stack([
        test_matrix.std(axis=1),
        test_matrix.mean(axis=1),
        test_matrix.max(axis=1) - test_matrix.min(axis=1),
        np.median(test_matrix, axis=1),
    ]))

    stack_train_ext = np.hstack([stack_train, stats_train])
    stack_test_ext = np.hstack([stack_test, stats_test])

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(stack_train, y, groups=groups))

    # Run 1: 33 bases only (reproduce baseline)
    print("\n" + "=" * 60, flush=True)
    print("Run 1: 3-meta on 33 bases only (baseline reproduction)", flush=True)
    print("=" * 60, flush=True)
    avg_oof_1, avg_test_1, _, _ = fit_3meta(stack_train, stack_test, y, y_log, folds, tag="33only")
    mae1 = mean_absolute_error(y, avg_oof_1)
    print(f">>> Run 1 3-meta avg: {mae1:.5f}  (baseline mega33: {baseline_mae:.5f}, delta {mae1-baseline_mae:+.5f})", flush=True)

    # Run 2: 33 + 4 OOF stats
    print("\n" + "=" * 60, flush=True)
    print("Run 2: 3-meta on 33 + 4 OOF stats", flush=True)
    print("=" * 60, flush=True)
    avg_oof_2, avg_test_2, _, _ = fit_3meta(stack_train_ext, stack_test_ext, y, y_log, folds, tag="33+stats")
    mae2 = mean_absolute_error(y, avg_oof_2)
    print(f">>> Run 2 3-meta avg: {mae2:.5f}  (vs Run 1 {mae1:.5f}, delta {mae2-mae1:+.5f})", flush=True)
    print(f">>> Run 2 vs baseline mega33 {baseline_mae:.5f}, delta {mae2-baseline_mae:+.5f}", flush=True)

    # Save
    np.save(os.path.join(OUT, "avg_oof_33only.npy"), avg_oof_1)
    np.save(os.path.join(OUT, "avg_test_33only.npy"), avg_test_1)
    np.save(os.path.join(OUT, "avg_oof_33plus_stats.npy"), avg_oof_2)
    np.save(os.path.join(OUT, "avg_test_33plus_stats.npy"), avg_test_2)

    summary = dict(
        baseline_mega33=float(baseline_mae),
        run1_33only=float(mae1),
        run2_33plus_stats=float(mae2),
        delta_stats_improvement=float(mae2 - mae1),
        delta_vs_baseline=float(mae2 - baseline_mae),
    )
    with open(os.path.join(OUT, "summary_3meta.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT}/summary_3meta.json", flush=True)

    # Verdict
    print("\n" + "=" * 60, flush=True)
    if mae2 < baseline_mae - 0.002:
        print(f"VERDICT: PROMOTE. Run 2 ({mae2:.5f}) improves baseline ({baseline_mae:.5f}) by {baseline_mae-mae2:+.5f}.", flush=True)
        print("Next step: re-optimize final_multi_blend weights with this new meta_avg.", flush=True)
    else:
        print(f"VERDICT: NOT PROMOTE. Run 2 ({mae2:.5f}) does not improve baseline ({baseline_mae:.5f}) meaningfully.", flush=True)
        print("OOF stats features improved single-LGB but not 3-meta aggregate.", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
