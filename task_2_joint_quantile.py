"""
Task 2: Joint 9-quantile distributional base.

Train 9 LGB quantile models (alpha = 0.1, 0.2, ..., 0.9).
Each gives per-row Q_alpha(y | X) prediction.

Strategy 1: Use (Q10, ..., Q90) directly as 9 features for LGB meta that predicts y.
  - This captures distributional shape per row
  - Different from any single Q_alpha base

Strategy 2: Derive summary features: mean, dispersion, skewness-like from 9 quantiles
  - Add to v23 149 and retrain LGB

We use Strategy 1 (joint distribution as input to meta LGB) as primary.
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
OUT = os.path.join(ROOT, "results", "task_2_joint")
os.makedirs(OUT, exist_ok=True)

ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def main():
    print("=" * 60, flush=True)
    print("Task 2: Joint 9-quantile distributional base", flush=True)
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
    print(f"X shape: {X.shape}", flush=True)

    # Mega33 baseline
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))
    print(f"Mega33 OOF: {mega_mae:.5f}", flush=True)

    # Check existing q_sweep results to reuse
    q_path = os.path.join(ROOT, "results", "q_sweep")
    reusable = {}
    for a in [0.55, 0.60, 0.65, 0.70, 0.80]:
        p = os.path.join(q_path, f"q{int(a*100):02d}_oof.npy")
        if os.path.exists(p):
            reusable[a] = p
    print(f"Reusable q_sweep OOFs: {list(reusable.keys())}", flush=True)
    # Also q90 from earlier
    if os.path.exists(os.path.join(ROOT, "results", "q90", "q90_oof.npy")):
        reusable[0.90] = os.path.join(ROOT, "results", "q90", "q90_oof.npy")

    # Train 9 quantile models: 0.1, 0.2, ..., 0.9 (skipping ones we have)
    # To save time, train only: 0.1, 0.2, 0.3, 0.4, 0.5, 0.9 (reuse 0.55/0.60/0.65/0.70/0.80)
    # Quantile models at 0.5 already exist in v23 (MAE objective)
    # But quantile 0.5 != MAE (both are median-optimal but different parametrization)
    # Need to train strict quantiles for joint distribution

    # For simplicity: train all 9 quantiles fresh in log space
    # With reduced LGB (n_estimators smaller) to fit in time budget
    print("\n=== Training 9 quantile models ===", flush=True)
    q_oofs = {}  # alpha -> OOF (y space)
    q_tests = {}  # alpha -> test

    for alpha in ALPHAS:
        print(f"\n--- alpha={alpha} ---", flush=True)
        oof = np.zeros(len(y), dtype=np.float64)
        test_preds = []
        for f in range(5):
            val_mask = fold_idx == f
            tr_mask = ~val_mask
            params = dict(
                objective="quantile", alpha=alpha,
                learning_rate=0.05, n_estimators=1000,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            m = lgb.LGBMRegressor(**params)
            m.fit(
                X[tr_mask], y_log[tr_mask],
                eval_set=[(X[val_mask], y_log[val_mask])],
                callbacks=[lgb.early_stopping(80, verbose=False)],
            )
            oof[val_mask] = np.expm1(m.predict(X[val_mask]))
            test_preds.append(np.expm1(m.predict(X_test)))
        q_oofs[alpha] = np.clip(oof, 0, None)
        q_tests[alpha] = np.clip(np.mean(np.stack(test_preds, axis=0), axis=0), 0, None)
        single_mae = mean_absolute_error(y, q_oofs[alpha])
        print(f"  Q{int(alpha*100)}: single MAE = {single_mae:.5f}", flush=True)

    # Build joint quantile matrix (train: 250k x 9, test: 50k x 9)
    q_train_mat = np.column_stack([q_oofs[a] for a in ALPHAS])
    q_test_mat = np.column_stack([q_tests[a] for a in ALPHAS])

    # Summary features
    def summary(mat):
        mean = mat.mean(axis=1)
        median = mat[:, 4]  # Q50
        spread = mat[:, 8] - mat[:, 0]  # Q90 - Q10
        iqr = mat[:, 6] - mat[:, 2]  # Q70 - Q30 (approx)
        skew_like = mat[:, 8] + mat[:, 0] - 2 * mat[:, 4]  # Q90+Q10-2Q50
        return np.column_stack([mean, median, spread, iqr, skew_like])

    q_train_sum = summary(q_train_mat)
    q_test_sum = summary(q_test_mat)

    print("\n=== Strategy 1: meta LGB over 9 quantiles only ===", flush=True)
    # Meta LGB: input = 9 quantiles (log), output = y (MAE)
    meta_oof = np.zeros(len(y), dtype=np.float64)
    meta_test_preds = []
    q_train_log = np.log1p(q_train_mat)
    q_test_log = np.log1p(q_test_mat)
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        m = lgb.LGBMRegressor(
            objective="mae", n_estimators=500, learning_rate=0.05,
            num_leaves=15, max_depth=4, min_child_samples=100,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(q_train_log[tr_mask], y_log[tr_mask],
              eval_set=[(q_train_log[val_mask], y_log[val_mask])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        meta_oof[val_mask] = np.expm1(m.predict(q_train_log[val_mask]))
        meta_test_preds.append(np.expm1(m.predict(q_test_log)))
    meta_oof = np.clip(meta_oof, 0, None)
    meta_test = np.clip(np.mean(np.stack(meta_test_preds, axis=0), axis=0), 0, None)
    meta_mae = float(mean_absolute_error(y, meta_oof))
    corr_meta = float(np.corrcoef(y - mega_oof, y - meta_oof)[0, 1])
    print(f"  meta LGB over 9 quantiles: MAE = {meta_mae:.5f}, corr(mega) = {corr_meta:.4f}", flush=True)

    print("\n=== Strategy 2: LGB over v23 149 + 9 quantiles + 5 summary (163 features) ===", flush=True)
    X_ext = np.hstack([X, q_train_mat, q_train_sum])
    X_test_ext = np.hstack([X_test, q_test_mat, q_test_sum])
    ext_oof = np.zeros(len(y), dtype=np.float64)
    ext_test_preds = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        m = lgb.LGBMRegressor(
            objective="huber", n_estimators=3000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(X_ext[tr_mask], y_log[tr_mask],
              eval_set=[(X_ext[val_mask], y_log[val_mask])],
              callbacks=[lgb.early_stopping(200, verbose=False)])
        ext_oof[val_mask] = np.expm1(m.predict(X_ext[val_mask]))
        ext_test_preds.append(np.expm1(m.predict(X_test_ext)))
    ext_oof = np.clip(ext_oof, 0, None)
    ext_test = np.clip(np.mean(np.stack(ext_test_preds, axis=0), axis=0), 0, None)
    ext_mae = float(mean_absolute_error(y, ext_oof))
    corr_ext = float(np.corrcoef(y - mega_oof, y - ext_oof)[0, 1])
    print(f"  ext LGB (v23+9q+5sum): MAE = {ext_mae:.5f}, corr(mega) = {corr_ext:.4f}", flush=True)

    # Blend test
    def test_blend(candidate_oof, candidate_test, name):
        best_mae = mega_mae
        best_w = 0
        for w in np.linspace(0, 0.5, 51):
            pred = (1 - w) * mega_oof + w * candidate_oof
            m = float(np.mean(np.abs(y - pred)))
            if m < best_mae:
                best_mae = m
                best_w = w
        print(f"  [{name}] best w={best_w:.3f}  blend MAE={best_mae:.5f}  delta={best_mae-mega_mae:+.5f}", flush=True)
        return best_mae, best_w

    print("\n=== Blend tests ===", flush=True)
    mae_meta_blend, w_meta = test_blend(meta_oof, meta_test, "meta-over-9q")
    mae_ext_blend, w_ext = test_blend(ext_oof, ext_test, "ext-v23+9q+sum")

    # Save
    np.save(os.path.join(OUT, "q_train_mat.npy"), q_train_mat)
    np.save(os.path.join(OUT, "q_test_mat.npy"), q_test_mat)
    np.save(os.path.join(OUT, "meta_oof.npy"), meta_oof)
    np.save(os.path.join(OUT, "meta_test.npy"), meta_test)
    np.save(os.path.join(OUT, "ext_oof.npy"), ext_oof)
    np.save(os.path.join(OUT, "ext_test.npy"), ext_test)
    summary_dict = dict(
        meta_mae=meta_mae, meta_corr=corr_meta,
        ext_mae=ext_mae, ext_corr=corr_ext,
        meta_blend_w=w_meta, meta_blend_mae=mae_meta_blend,
        ext_blend_w=w_ext, ext_blend_mae=mae_ext_blend,
        mega_mae=mega_mae,
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary_dict, f, indent=2)

    # Verdict
    best_blend = min(mae_meta_blend, mae_ext_blend)
    if best_blend < mega_mae - 0.0005:
        print(f"\nVERDICT: PROCEED (best blend {best_blend:.5f}, improvement {mega_mae-best_blend:+.5f})", flush=True)
    else:
        print(f"\nVERDICT: NO_GO (best blend {best_blend:.5f} = baseline)", flush=True)


if __name__ == "__main__":
    main()
