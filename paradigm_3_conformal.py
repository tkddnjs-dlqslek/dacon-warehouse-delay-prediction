"""
Paradigm 3: Conformal + strategic shrinkage (post-process only).

Bayesian shrinkage interpretation:
  pred_new = (1 - alpha(uncertainty)) * mega33 + alpha(uncertainty) * shrinkage_target

Uncertainty proxies (3 options):
  U1: std(lgb_oof, xgb_oof, cb_oof) per row
  U2: |residual| per (layout_type x ts_bucket) bucket (quantile-based)
  U3: adversarial prob per row (if precomputed)

Shrinkage targets:
  T1: global median of y (9.03)
  T2: bucket median (per layout_type x ts_bucket)
  T3: mega33 OOF bucket mean

Strict CV: for each fold, fit shrinkage weights on tr-fold only, apply to val-fold.
Kill gate: CV-honest OOF improvement < 0.001 → discard.
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
from scipy.optimize import minimize_scalar

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "paradigm_3")
os.makedirs(OUT, exist_ok=True)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    print("=" * 60, flush=True)
    print("Paradigm 3: Conformal + Strategic Shrinkage", flush=True)
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
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # mega33 baseline
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]
    mega_mae = mae(y, mega_oof)
    print(f"mega33 baseline: {mega_mae:.5f}", flush=True)

    # Individual meta OOFs for uncertainty
    lgb_oof = mega["meta_oofs"]["lgb"]
    xgb_oof = mega["meta_oofs"]["xgb"]
    cb_oof = mega["meta_oofs"]["cb"]
    lgb_test = mega["meta_tests"]["lgb"]
    xgb_test = mega["meta_tests"]["xgb"]
    cb_test = mega["meta_tests"]["cb"]

    # ─── U1: per-row std across 3 meta ───
    u1_train = np.std(np.column_stack([lgb_oof, xgb_oof, cb_oof]), axis=1)
    u1_test = np.std(np.column_stack([lgb_test, xgb_test, cb_test]), axis=1)
    print(f"\nU1 (meta-std): train mean={u1_train.mean():.3f}, max={u1_train.max():.3f}", flush=True)

    # ─── Shrinkage targets ───
    y_median_global = float(np.median(y))
    print(f"\nT1 global y median: {y_median_global:.3f}", flush=True)

    # layout_type × ts_bucket median
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))[["layout_id", "layout_type"]]
    train = train.merge(layout_info, on="layout_id", how="left")
    test = test.merge(layout_info, on="layout_id", how="left")
    train["ts_idx"] = np.tile(np.arange(25), len(train) // 25)
    test["ts_idx"] = np.tile(np.arange(25), len(test) // 25)
    train["ts_bucket"] = pd.cut(train["ts_idx"], bins=[-1, 7, 15, 24], labels=["early", "mid", "late"]).astype(str)
    test["ts_bucket"] = pd.cut(test["ts_idx"], bins=[-1, 7, 15, 24], labels=["early", "mid", "late"]).astype(str)

    # bucket median (from train only; applied to test by bucket key)
    # For CV: compute bucket median per fold (train fold only)
    print("\nComputing bucket medians...", flush=True)
    bucket_median_train = np.zeros(len(y))
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        tmp = train[tr_mask].copy()
        tmp["_y"] = y[tr_mask]
        g = tmp.groupby(["layout_type", "ts_bucket"])["_y"].median()
        # map to val
        val_keys = list(zip(train.loc[val_mask, "layout_type"], train.loc[val_mask, "ts_bucket"]))
        vals = []
        for k in val_keys:
            vals.append(g.loc[k] if k in g.index else y_median_global)
        bucket_median_train[val_mask] = np.array(vals)
    # test bucket median uses full train
    train["_y"] = y
    g_full = train.groupby(["layout_type", "ts_bucket"])["_y"].median()
    test_keys = list(zip(test["layout_type"], test["ts_bucket"]))
    bucket_median_test = np.array([g_full.loc[k] if k in g_full.index else y_median_global for k in test_keys])

    # ─── Strategy 1: Global shrinkage with uncertainty-scaled alpha ───
    # pred_new = (1 - alpha * norm_u) * mega + alpha * norm_u * T
    # norm_u = u1 / percentile95(u1) so max ~1
    u_p95 = np.percentile(u1_train, 95)
    norm_u_train = np.clip(u1_train / u_p95, 0, 1)
    norm_u_test = np.clip(u1_test / u_p95, 0, 1)

    def score_shrink(alpha, target_train, return_pred=False):
        pred = (1 - alpha * norm_u_train) * mega_oof + alpha * norm_u_train * target_train
        if return_pred:
            return pred
        return mae(y, pred)

    print("\n=== Strategy A: Global shrinkage toward T1 (global median) ===", flush=True)
    # CV-honest search: fit alpha on train-fold only, apply to val-fold
    # Implement simple grid per fold
    cv_oof_A = np.zeros(len(y))
    per_fold_alpha_A = []
    for f in range(5):
        val_mask = fold_idx == f; tr_mask = ~val_mask
        best_a = 0; best_m = np.inf
        for a in np.linspace(0, 1.5, 31):
            pred = (1 - a * norm_u_train[tr_mask]) * mega_oof[tr_mask] + a * norm_u_train[tr_mask] * y_median_global
            m = mae(y[tr_mask], pred)
            if m < best_m:
                best_m = m; best_a = a
        cv_oof_A[val_mask] = (1 - best_a * norm_u_train[val_mask]) * mega_oof[val_mask] + best_a * norm_u_train[val_mask] * y_median_global
        per_fold_alpha_A.append(best_a)
    mae_A = mae(y, cv_oof_A)
    print(f"  per-fold alpha: {[f'{a:.2f}' for a in per_fold_alpha_A]}", flush=True)
    print(f"  CV OOF MAE: {mae_A:.5f}  delta: {mae_A - mega_mae:+.5f}", flush=True)

    # Global (in-sample) for reference
    best_a_global = 0; best_m_global = np.inf
    for a in np.linspace(0, 1.5, 31):
        m = score_shrink(a, y_median_global)
        if m < best_m_global: best_m_global = m; best_a_global = a
    print(f"  in-sample best alpha: {best_a_global:.2f}, MAE: {best_m_global:.5f}  delta: {best_m_global - mega_mae:+.5f}", flush=True)

    # ─── Strategy B: Shrinkage toward T2 (bucket median) ───
    print("\n=== Strategy B: Shrinkage toward T2 (bucket median) ===", flush=True)
    cv_oof_B = np.zeros(len(y))
    per_fold_alpha_B = []
    for f in range(5):
        val_mask = fold_idx == f; tr_mask = ~val_mask
        best_a = 0; best_m = np.inf
        for a in np.linspace(0, 1.5, 31):
            pred = (1 - a * norm_u_train[tr_mask]) * mega_oof[tr_mask] + a * norm_u_train[tr_mask] * bucket_median_train[tr_mask]
            m = mae(y[tr_mask], pred)
            if m < best_m:
                best_m = m; best_a = a
        cv_oof_B[val_mask] = (1 - best_a * norm_u_train[val_mask]) * mega_oof[val_mask] + best_a * norm_u_train[val_mask] * bucket_median_train[val_mask]
        per_fold_alpha_B.append(best_a)
    mae_B = mae(y, cv_oof_B)
    print(f"  per-fold alpha: {[f'{a:.2f}' for a in per_fold_alpha_B]}", flush=True)
    print(f"  CV OOF MAE: {mae_B:.5f}  delta: {mae_B - mega_mae:+.5f}", flush=True)

    # ─── Strategy C: Blend both targets ───
    print("\n=== Strategy C: Blend T1 + T2 ===", flush=True)
    # pred = mega + alpha*u * (beta*T1 + (1-beta)*T2 - mega)
    # For simplicity, fixed beta=0.5
    beta = 0.5
    target_blend_train = beta * y_median_global + (1 - beta) * bucket_median_train
    target_blend_test = beta * y_median_global + (1 - beta) * bucket_median_test
    cv_oof_C = np.zeros(len(y))
    per_fold_alpha_C = []
    for f in range(5):
        val_mask = fold_idx == f; tr_mask = ~val_mask
        best_a = 0; best_m = np.inf
        for a in np.linspace(0, 1.5, 31):
            pred = (1 - a * norm_u_train[tr_mask]) * mega_oof[tr_mask] + a * norm_u_train[tr_mask] * target_blend_train[tr_mask]
            m = mae(y[tr_mask], pred)
            if m < best_m:
                best_m = m; best_a = a
        cv_oof_C[val_mask] = (1 - best_a * norm_u_train[val_mask]) * mega_oof[val_mask] + best_a * norm_u_train[val_mask] * target_blend_train[val_mask]
        per_fold_alpha_C.append(best_a)
    mae_C = mae(y, cv_oof_C)
    print(f"  per-fold alpha: {[f'{a:.2f}' for a in per_fold_alpha_C]}", flush=True)
    print(f"  CV OOF MAE: {mae_C:.5f}  delta: {mae_C - mega_mae:+.5f}", flush=True)

    # ─── Select best ───
    strategies = {"A_globalT1": mae_A, "B_bucketT2": mae_B, "C_blend": mae_C}
    best_name = min(strategies, key=strategies.get)
    best_mae = strategies[best_name]
    print(f"\nBest strategy: {best_name}, CV OOF MAE {best_mae:.5f}, delta {best_mae - mega_mae:+.5f}", flush=True)

    # ─── Kill gate: CV-honest improvement >= 0.001 ───
    if best_mae > mega_mae - 0.001:
        print(f"\nVERDICT: NO_GO (CV improvement {mega_mae - best_mae:.5f} below threshold 0.001)", flush=True)
    else:
        print(f"\nVERDICT: PROCEED", flush=True)
        # Build test prediction with mean alpha across folds
        if best_name == "A_globalT1":
            a_mean = np.mean(per_fold_alpha_A)
            pred_test = (1 - a_mean * norm_u_test) * mega_test + a_mean * norm_u_test * y_median_global
        elif best_name == "B_bucketT2":
            a_mean = np.mean(per_fold_alpha_B)
            pred_test = (1 - a_mean * norm_u_test) * mega_test + a_mean * norm_u_test * bucket_median_test
        else:
            a_mean = np.mean(per_fold_alpha_C)
            pred_test = (1 - a_mean * norm_u_test) * mega_test + a_mean * norm_u_test * target_blend_test
        print(f"  mean alpha: {a_mean:.3f}", flush=True)

        # Save mega33-only version with shrinkage (base replacement)
        np.save(os.path.join(OUT, "shrink_oof.npy"), {"A": cv_oof_A, "B": cv_oof_B, "C": cv_oof_C}[best_name.split("_")[0]])
        np.save(os.path.join(OUT, "shrink_test.npy"), pred_test)

        # Now combine with full FIXED blend
        print("\nCombining with FIXED blend (rank + iter)...", flush=True)
        rank_oof = np.load(os.path.join(ROOT, "results", "ranking", "rank_adj_oof.npy"))
        rank_test = np.load(os.path.join(ROOT, "results", "ranking", "rank_adj_test.npy"))
        iter_r2_oof = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round2_oof.npy"))
        iter_r2_test = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round2_test.npy"))
        iter_r3_oof = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round3_oof.npy"))
        iter_r3_test = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round3_test.npy"))

        # Use chosen shrink result
        shrink_oof = {"A_globalT1": cv_oof_A, "B_bucketT2": cv_oof_B, "C_blend": cv_oof_C}[best_name]

        from scipy.optimize import minimize
        oofs_mat = np.column_stack([shrink_oof, rank_oof, iter_r2_oof, iter_r3_oof])
        tests_mat = np.column_stack([pred_test, rank_test, iter_r2_test, iter_r3_test])
        def obj(w):
            w = np.clip(w, 0, None)
            if w.sum() < 1e-6: return 99
            w = w / w.sum()
            return mae(y, oofs_mat @ w)
        x0 = np.array([0.77, 0.16, 0.04, 0.03])
        res = minimize(obj, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 100000})
        w_final = np.clip(res.x, 0, None); w_final /= w_final.sum()
        print(f"  blend w: shrink={w_final[0]:.4f}, rank={w_final[1]:.4f}, iter_r2={w_final[2]:.4f}, iter_r3={w_final[3]:.4f}", flush=True)
        print(f"  final OOF MAE: {res.fun:.5f}", flush=True)

        # Save submission
        final_test = tests_mat @ w_final
        test_raw = pd.read_csv(os.path.join(ROOT, "test.csv"))
        test_sorted = test_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
        out_df = pd.DataFrame({"ID": test_sorted["ID"].values, "avg_delay_minutes_next_30m": np.clip(final_test, 0, None)})
        sub_sample = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
        sub_out = sub_sample[["ID"]].merge(out_df, on="ID", how="left")
        assert sub_out["avg_delay_minutes_next_30m"].isna().sum() == 0
        sub_path = os.path.join(ROOT, "results", "final_blend", "submission_v8_conformal.csv")
        sub_out.to_csv(sub_path, index=False)
        print(f"  Saved: {sub_path}", flush=True)

    summary = dict(
        mega_mae=mega_mae,
        strat_A=mae_A, strat_B=mae_B, strat_C=mae_C,
        alpha_A=per_fold_alpha_A, alpha_B=per_fold_alpha_B, alpha_C=per_fold_alpha_C,
        best=best_name, best_mae=best_mae,
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
