"""
Meta-side experiments on mega33 (no base retraining).

Phase A-0: Load 33 base OOFs, verify baseline meta = 8.3989
Phase A-1: OOF-derived meta features (std/mean/range/max-min)
Phase A-2: Ridge/Lasso/NNLS meta as alternatives
Phase A-3: Greedy backward base selection
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
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from scipy.optimize import nnls
import lightgbm as lgb
import xgboost as xgb

# Force unbuffered stdout
sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "meta_exp")
os.makedirs(OUT, exist_ok=True)

BASELINE_MAE = 8.3989


def load_33_bases():
    R = os.path.join(ROOT, "results")
    bases = {}
    bases_test = {}

    # 9 x v23 GBDT
    for seed in [42, 123, 2024]:
        s = pickle.load(open(f"{R}/v23_seed{seed}.pkl", "rb"))
        for name in ["LGB_Huber", "XGB", "CatBoost"]:
            bases[f"v23s{seed}_{name}"] = s["oofs"][name]
            bases_test[f"v23s{seed}_{name}"] = s["tests"][name]

    # 4 x v24
    v24 = pickle.load(open(f"{R}/v24_final.pkl", "rb"))
    for name in v24["oofs"]:
        bases[f"v24_{name}"] = v24["oofs"][name]
        bases_test[f"v24_{name}"] = v24["tests"][name]

    # 4 x v26
    v26 = pickle.load(open(f"{R}/v26_final.pkl", "rb"))
    for name in ["Tuned_Huber", "Tuned_sqrt", "Tuned_pow", "DART"]:
        bases[f"v26_{name}"] = v26["oofs"][name]
        bases_test[f"v26_{name}"] = v26["tests"][name]

    # 3 x MLP/CNN
    mlp1 = pickle.load(open(f"{R}/mlp_final.pkl", "rb"))
    mlp2 = pickle.load(open(f"{R}/mlp2_final.pkl", "rb"))
    cnn = pickle.load(open(f"{R}/cnn_final.pkl", "rb"))
    bases["mlp1"] = mlp1["mlp_oof"]; bases_test["mlp1"] = mlp1["mlp_test"]
    bases["mlp2"] = mlp2["mlp2_oof"]; bases_test["mlp2"] = mlp2["mlp2_test"]
    bases["cnn"] = cnn["cnn_oof"]; bases_test["cnn"] = cnn["cnn_test"]

    # 3 x domain
    domain = pickle.load(open(f"{R}/domain_phase2.pkl", "rb"))
    for n in domain["oofs"]:
        bases[f"domain_{n}"] = domain["oofs"][n]
        bases_test[f"domain_{n}"] = domain["tests"][n]

    # 1 x mlp_aug
    mlp_aug = pickle.load(open(f"{R}/mlp_aug_final.pkl", "rb"))
    bases["mlp_aug"] = mlp_aug["mlp_aug_oof"]
    bases_test["mlp_aug"] = mlp_aug["mlp_aug_test"]

    # 3 x offset
    offset_p3 = pickle.load(open(f"{R}/offset_phase3.pkl", "rb"))
    for n, d in offset_p3.items():
        bases[f"offset_{n}"] = d["oof"]
        bases_test[f"offset_{n}"] = d["test"]

    # 6 x neural army
    na = pickle.load(open(f"{R}/neural_army.pkl", "rb"))
    for n in na:
        bases[f"na_{n}"] = na[n]["oof"]
        bases_test[f"na_{n}"] = na[n]["test"]

    return bases, bases_test


def train_3meta(stack_train, stack_test, y_log, y, folds, seed=42):
    """Replicate mega33 3-meta (LGB+XGB+Cat averaged) meta-learner."""
    from catboost import CatBoostRegressor
    preds_oof = {}
    preds_test = {}

    # LGB
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = lgb.LGBMRegressor(
            objective="mae", n_estimators=500, learning_rate=0.05,
            num_leaves=15, max_depth=4, min_child_samples=100,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / len(folds)
    preds_oof["lgb"] = np.clip(oof, 0, None)
    preds_test["lgb"] = np.clip(tpred, 0, None)

    # XGB
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = xgb.XGBRegressor(
            objective="reg:absoluteerror", n_estimators=500, learning_rate=0.05,
            max_depth=4, min_child_weight=100, random_state=seed, n_jobs=-1,
            early_stopping_rounds=50, verbosity=0,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])], verbose=False)
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / len(folds)
    preds_oof["xgb"] = np.clip(oof, 0, None)
    preds_test["xgb"] = np.clip(tpred, 0, None)

    # CatBoost
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for tr_idx, val_idx in folds:
        m = CatBoostRegressor(
            loss_function="MAE", iterations=500, learning_rate=0.05,
            depth=4, random_seed=seed, verbose=0,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=(stack_train[val_idx], y_log[val_idx]),
              early_stopping_rounds=50, verbose=0)
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / len(folds)
    preds_oof["cb"] = np.clip(oof, 0, None)
    preds_test["cb"] = np.clip(tpred, 0, None)

    # Average
    avg_oof = (preds_oof["lgb"] + preds_oof["xgb"] + preds_oof["cb"]) / 3
    avg_test = (preds_test["lgb"] + preds_test["xgb"] + preds_test["cb"]) / 3
    return avg_oof, avg_test, preds_oof, preds_test


def train_single_meta_lgb(stack_train, y, y_log, folds, seed=42):
    """Quick single-meta LGB (for fast iteration in greedy backward)."""
    oof = np.zeros(len(y))
    for tr_idx, val_idx in folds:
        m = lgb.LGBMRegressor(
            objective="mae", n_estimators=500, learning_rate=0.05,
            num_leaves=15, max_depth=4, min_child_samples=100,
            random_state=seed, verbose=-1, n_jobs=-1,
        )
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    return np.clip(oof, 0, None)


def main():
    print("=" * 60)
    print("Meta experiments — Phase A")
    print("=" * 60)

    # Load train
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values
    y_log = np.log1p(y)
    groups = train["layout_id"].values

    # Load mega33 baseline for comparison
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega_final = pickle.load(f)
    baseline_avg_oof = mega_final["meta_avg_oof"]
    baseline_mae = mean_absolute_error(y, baseline_avg_oof)
    print(f"Baseline mega33 meta_avg OOF MAE: {baseline_mae:.5f}  (expect {BASELINE_MAE})")

    # Load 33 base OOFs
    print("\nLoading 33 base OOFs...")
    bases, bases_test = load_33_bases()
    print(f"  Loaded {len(bases)} bases")
    names = list(bases.keys())
    # Individual MAEs
    for n in names[:5]:
        print(f"    {n}: {mean_absolute_error(y, bases[n]):.4f}")
    print(f"    ... (total {len(names)})")

    # Build stack arrays (log1p)
    stack_train = np.column_stack([np.log1p(np.clip(bases[n], 0, None)) for n in names])
    stack_test = np.column_stack([np.log1p(np.clip(bases_test[n], 0, None)) for n in names])
    print(f"stack shape: train {stack_train.shape}  test {stack_test.shape}")

    # Folds
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(stack_train, y, groups=groups))

    summary = dict(baseline_mae=baseline_mae)

    # ─── Phase A-1: OOF-derived meta features ───────────────────
    print("\n" + "=" * 60)
    print("Phase A-1: OOF-derived meta features")
    print("=" * 60)
    # Derive: std, mean, max-min range, median from 33 OOFs (in original y space)
    oof_matrix = np.column_stack([np.clip(bases[n], 0, None) for n in names])
    oof_std = oof_matrix.std(axis=1, keepdims=True)
    oof_mean = oof_matrix.mean(axis=1, keepdims=True)
    oof_range = (oof_matrix.max(axis=1) - oof_matrix.min(axis=1))[:, None]
    oof_median = np.median(oof_matrix, axis=1, keepdims=True)

    test_matrix = np.column_stack([np.clip(bases_test[n], 0, None) for n in names])
    test_std = test_matrix.std(axis=1, keepdims=True)
    test_mean = test_matrix.mean(axis=1, keepdims=True)
    test_range = (test_matrix.max(axis=1) - test_matrix.min(axis=1))[:, None]
    test_median = np.median(test_matrix, axis=1, keepdims=True)

    extra_train = np.log1p(np.hstack([oof_std, oof_mean, oof_range, oof_median]))
    extra_test = np.log1p(np.hstack([test_std, test_mean, test_range, test_median]))

    stack_train_ext = np.hstack([stack_train, extra_train])
    stack_test_ext = np.hstack([stack_test, extra_test])

    print("Running LGB single-meta (quick check)...")
    oof_lgb_plain = train_single_meta_lgb(stack_train, y, y_log, folds)
    oof_lgb_ext = train_single_meta_lgb(stack_train_ext, y, y_log, folds)
    mae_plain = mean_absolute_error(y, oof_lgb_plain)
    mae_ext = mean_absolute_error(y, oof_lgb_ext)
    print(f"  LGB meta plain (33 only):     {mae_plain:.5f}")
    print(f"  LGB meta + OOF stats (33+4):  {mae_ext:.5f}  delta {mae_ext-mae_plain:+.5f}")
    summary["phase_a1"] = dict(
        lgb_plain=mae_plain,
        lgb_extended=mae_ext,
        delta=mae_ext - mae_plain,
    )

    # ─── Phase A-2: Alternative meta learners ───────────────────
    print("\n" + "=" * 60)
    print("Phase A-2: Alternative metas (Ridge/Lasso/NNLS)")
    print("=" * 60)

    # OOF-based Ridge/Lasso: simple linear combinations
    # For each fold: fit on train fold, predict on val
    def train_linear_meta(Xtr, ytr, folds, model_fn, name):
        oof = np.zeros(len(ytr))
        for tr_idx, val_idx in folds:
            m = model_fn()
            m.fit(Xtr[tr_idx], ytr[tr_idx])
            oof[val_idx] = m.predict(Xtr[val_idx])
        oof = np.clip(np.expm1(oof), 0, None)
        return oof

    # use stack_train (log1p) vs y_log target
    print("Ridge meta (alpha=1.0)...")
    oof_ridge = train_linear_meta(stack_train, y_log, folds, lambda: Ridge(alpha=1.0, positive=True), "ridge_pos")
    mae_ridge = mean_absolute_error(y, oof_ridge)
    print(f"  Ridge(alpha=1, pos): {mae_ridge:.5f}  delta {mae_ridge-baseline_mae:+.5f}")

    print("Lasso meta...")
    oof_lasso = train_linear_meta(stack_train, y_log, folds, lambda: Lasso(alpha=0.001, positive=True, max_iter=5000), "lasso")
    mae_lasso = mean_absolute_error(y, oof_lasso)
    print(f"  Lasso(alpha=0.001, pos): {mae_lasso:.5f}  delta {mae_lasso-baseline_mae:+.5f}")

    # NNLS: non-negative weights no intercept
    print("NNLS meta...")
    oof_nnls = np.zeros(len(y))
    nnls_weights_folds = []
    for tr_idx, val_idx in folds:
        w, _ = nnls(stack_train[tr_idx], y_log[tr_idx])
        oof_nnls[val_idx] = np.expm1(stack_train[val_idx] @ w)
        nnls_weights_folds.append(w)
    oof_nnls = np.clip(oof_nnls, 0, None)
    mae_nnls = mean_absolute_error(y, oof_nnls)
    print(f"  NNLS: {mae_nnls:.5f}  delta {mae_nnls-baseline_mae:+.5f}")
    # NNLS average weights for insight
    avg_w = np.mean(nnls_weights_folds, axis=0)
    n_nonzero = int((avg_w > 1e-4).sum())
    print(f"  NNLS avg weights: {n_nonzero}/{len(avg_w)} non-zero bases")

    summary["phase_a2"] = dict(
        ridge=mae_ridge, lasso=mae_lasso, nnls=mae_nnls,
        nnls_nonzero_bases=n_nonzero,
    )

    # ─── Phase A-3: Greedy backward base selection (Ridge-accelerated) ──
    print("\n" + "=" * 60, flush=True)
    print("Phase A-3: Greedy backward base selection (Ridge-fast)", flush=True)
    print("=" * 60, flush=True)
    # Use Ridge meta for fast iteration; verify with LGB meta at the end only.

    def ridge_meta_mae(sub_train, ytr_log, folds_local):
        oof = np.zeros(len(ytr_log))
        for tr_idx, val_idx in folds_local:
            m = Ridge(alpha=1.0, positive=True)
            m.fit(sub_train[tr_idx], ytr_log[tr_idx])
            oof[val_idx] = m.predict(sub_train[val_idx])
        return float(mean_absolute_error(y, np.clip(np.expm1(oof), 0, None)))

    current = list(range(len(names)))
    iteration = 0
    history = []
    current_mae_ridge = ridge_meta_mae(stack_train, y_log, folds)
    print(f"Starting Ridge meta MAE on 33 bases: {current_mae_ridge:.5f}", flush=True)

    while len(current) > 8 and iteration < 20:
        iteration += 1
        best_removal = None
        best_new_mae = current_mae_ridge + 1e9
        per_removal_mae = []
        for idx in current:
            cand = [j for j in current if j != idx]
            sub = stack_train[:, cand]
            m = ridge_meta_mae(sub, y_log, folds)
            per_removal_mae.append((idx, names[idx], m))
            if m < best_new_mae:
                best_new_mae = m
                best_removal = idx
        per_removal_mae.sort(key=lambda r: r[2])
        # Accept if new_mae <= current + 0.0003 (allow small noise)
        threshold = current_mae_ridge + 0.0003
        if best_new_mae <= threshold:
            current.remove(best_removal)
            history.append(dict(
                iter=iteration,
                removed_base=names[best_removal],
                new_n=len(current),
                new_mae=best_new_mae,
            ))
            prev = current_mae_ridge
            current_mae_ridge = best_new_mae
            print(f"  iter {iteration:>2}: drop {names[best_removal]:<30s} |curr|={len(current)} MAE={current_mae_ridge:.5f} (prev {prev:.5f})", flush=True)
        else:
            print(f"  iter {iteration}: no safe removal (best {best_new_mae:.5f} > thresh {threshold:.5f}). Stop.", flush=True)
            break

    print(f"\nRidge greedy done: {len(names)} -> {len(current)} bases. MAE {current_mae_ridge:.5f}", flush=True)
    # Only accept the reduction; use single LGB meta for fair baseline check

    # Single LGB meta on selected subset (instead of full 3-meta to save time)
    print(f"Verifying LGB meta on selected subset...", flush=True)
    sub_train_final = stack_train[:, current]
    oof_lgb_sub = train_single_meta_lgb(sub_train_final, y, y_log, folds)
    mae_lgb_sub = mean_absolute_error(y, oof_lgb_sub)
    print(f"  LGB meta on {len(current)} bases: {mae_lgb_sub:.5f}  delta vs 33-base LGB: {mae_lgb_sub - mae_plain:+.5f}", flush=True)

    best_new_mae = mae_lgb_sub

    final_bases = [names[i] for i in current]
    print(f"\nFinal subset: {len(current)} bases")
    print(f"Final LGB meta MAE: {current_mae:.5f}")
    print(f"vs starting: {mae_plain:.5f}  delta {current_mae-mae_plain:+.5f}")
    summary["phase_a3"] = dict(
        initial_n=len(names),
        final_n=len(current),
        final_mae=current_mae,
        delta_vs_start=current_mae - mae_plain,
        history=history,
        final_bases=final_bases,
    )

    # Final: 3-meta on selected subset
    if len(current) < len(names):
        print(f"\nTesting 3-meta (LGB+XGB+CB) on selected {len(current)} bases...")
        sub_train = stack_train[:, current]
        sub_test = stack_test[:, current]
        avg_oof, avg_test, _, _ = train_3meta(sub_train, sub_test, y_log, y, folds)
        mae_3meta = mean_absolute_error(y, avg_oof)
        print(f"  3-meta on {len(current)} bases: {mae_3meta:.5f}  delta vs baseline 8.39895 = {mae_3meta-baseline_mae:+.5f}")
        summary["phase_a3_3meta"] = dict(
            mae=mae_3meta,
            delta_vs_baseline=mae_3meta - baseline_mae,
        )
        np.save(os.path.join(OUT, "selected_subset_3meta_oof.npy"), avg_oof)
        np.save(os.path.join(OUT, "selected_subset_3meta_test.npy"), avg_test)

    summary["final_subset"] = final_bases

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved to {OUT}/summary.json")


if __name__ == "__main__":
    main()
