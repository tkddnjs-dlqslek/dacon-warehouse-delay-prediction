"""
Zero classifier + hard override.

Train LGB binary for P(y==0 | v23). Use high-precision threshold to override
mega33 predictions on confident zero predictions.

Kill gates:
  AUC < 0.85 -> abort
  OOF delta > -0.05 -> abort (no submission)

Output:
  results/zero_cls/p_zero_oof.npy
  results/zero_cls/p_zero_test.npy
  results/zero_cls/summary.json
  results/final_blend/submission_v5_zero.csv (only if pass)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "zero_cls")
FINAL_DIR = os.path.join(ROOT, "results", "final_blend")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60)
    print("Zero Classifier + Hard Override")
    print("=" * 60)

    # Load
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    y_bin = (y == 0).astype(np.int32)
    pos_rate = float(y_bin.mean())
    print(f"y==0 rows: {int(y_bin.sum())} ({pos_rate*100:.2f}%)")

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values

    # Train LGB binary
    oof = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    spw = (1 - pos_rate) / pos_rate
    params = dict(
        objective="binary",
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=80,
        lambda_l2=1.0,
        scale_pos_weight=spw,
        seed=42,
        verbose=-1,
    )
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        dtrain = lgb.Dataset(X[tr_mask], label=y_bin[tr_mask])
        dval = lgb.Dataset(X[val_mask], label=y_bin[val_mask], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof[val_mask] = model.predict(X[val_mask])
        test_preds.append(model.predict(X_test))
        fold_auc = roc_auc_score(y_bin[val_mask], oof[val_mask])
        print(f"  fold {f}: best_iter={model.best_iteration} AUC={fold_auc:.4f}")
    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)

    auc = float(roc_auc_score(y_bin, oof))
    ap = float(average_precision_score(y_bin, oof))
    print(f"\nOverall AUC: {auc:.4f}")
    print(f"Overall AP:  {ap:.4f}")

    # Kill gate 1: AUC
    pass_auc = auc >= 0.85
    print(f"Kill gate (AUC >= 0.85): {'PASS' if pass_auc else 'ABORT'}")

    # Threshold sweep for precision analysis
    print("\n=== Threshold sweep (precision-focused) ===")
    mega_mae = float(np.mean(np.abs(y - mega_oof)))
    print(f"mega33 baseline OOF MAE: {mega_mae:.4f}")
    print(f"{'thresh':>8} {'flagged':>8} {'flag_rate':>10} {'recall':>8} {'precision':>10} {'override_MAE':>14} {'delta':>10}")
    thresh_results = []
    best_thresh = None
    best_mae = mega_mae
    for t in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.98]:
        mask = oof > t
        flag_n = int(mask.sum())
        if flag_n == 0:
            continue
        tp = int((mask & (y_bin == 1)).sum())
        fp = flag_n - tp
        fn = int(((~mask) & (y_bin == 1)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / flag_n if flag_n > 0 else 0
        oof_override = mega_oof.copy()
        oof_override[mask] = 0.0
        m_override = float(np.mean(np.abs(y - oof_override)))
        delta = m_override - mega_mae
        print(f"{t:>8.2f} {flag_n:>8} {flag_n/len(y)*100:>9.3f}% {recall:>8.3f} {precision:>10.3f} {m_override:>14.4f} {delta:>+10.4f}")
        thresh_results.append(dict(
            thresh=t, flagged=flag_n, recall=recall, precision=precision,
            mae=m_override, delta=delta,
        ))
        if m_override < best_mae:
            best_mae = m_override
            best_thresh = t

    best_delta = best_mae - mega_mae
    print(f"\nBest threshold: {best_thresh}  MAE: {best_mae:.4f}  delta: {best_delta:+.4f}")

    # Kill gate 2: delta <= -0.05
    pass_delta = best_delta <= -0.05
    print(f"Kill gate (delta <= -0.05): {'PASS' if pass_delta else 'ABORT'}")

    # Save artifacts
    np.save(os.path.join(OUT, "p_zero_oof.npy"), oof)
    np.save(os.path.join(OUT, "p_zero_test.npy"), test_pred)

    # Build hybrid prediction on train (for reporting) and test (if pass)
    hybrid_oof = mega_oof.copy()
    hybrid_test = mega_test.copy()
    if best_thresh is not None:
        mask_tr = oof > best_thresh
        hybrid_oof[mask_tr] = 0.0
        mask_te = test_pred > best_thresh
        hybrid_test[mask_te] = 0.0
        print(f"\nTrain override: {int(mask_tr.sum())} rows → 0")
        print(f"Test override:  {int(mask_te.sum())} rows → 0")

    summary = dict(
        pos_rate=pos_rate,
        auc=auc,
        ap=ap,
        kill_gate_auc_pass=bool(pass_auc),
        kill_gate_delta_pass=bool(pass_delta),
        best_thresh=best_thresh,
        best_mae=best_mae,
        best_delta=best_delta,
        mega_mae=mega_mae,
        threshold_sweep=thresh_results,
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    if not (pass_auc and pass_delta):
        print("\n[ABORT] One or more kill gates failed. No submission generated.")
        return

    # Build submission using existing best blend + hard zero override on top
    # Current best: mega33 0.7637 + rank_orig 0.1589 + iter_r1 0.0119 + iter_r2 0.0346 + iter_r3 0.0310
    # Load the existing FIXED prediction and apply the zero override to it.
    print("\n=== Building submission_v5_zero.csv (FIXED + zero override) ===")
    fixed_csv = os.path.join(FINAL_DIR, "submission_final_multiblend_FIXED.csv")
    sub_fixed = pd.read_csv(fixed_csv)
    print(f"Loaded FIXED submission: {sub_fixed.shape}")
    assert len(sub_fixed) == len(test_pred), "length mismatch between FIXED submission and test pred"
    # test_pred ordering must match sub_fixed (both aligned to test.csv?). Verify via IDs.
    # test_fe is sorted by (layout_id, scenario_id); FIXED is in original test.csv order.
    # We need to reindex test_pred to match FIXED order.
    # Approach: build a DataFrame with test_fe ID and test_pred, merge to FIXED by ID.
    test_pred_df = pd.DataFrame({
        "ID": test_fe["ID"].values,
        "p_zero": test_pred,
    })
    sub = sub_fixed.merge(test_pred_df, on="ID", how="left")
    assert sub["p_zero"].isna().sum() == 0, "ID mismatch on merge"
    mask_sub = sub["p_zero"].values > best_thresh
    delay_col = "avg_delay_minutes_next_30m"
    before_mean = float(sub[delay_col].mean())
    sub.loc[mask_sub, delay_col] = 0.0
    after_mean = float(sub[delay_col].mean())
    print(f"Override count: {int(mask_sub.sum())}")
    print(f"Pred mean before/after: {before_mean:.4f} -> {after_mean:.4f}")

    out_sub = sub[["ID", delay_col]].copy()
    out_path = os.path.join(FINAL_DIR, "submission_v5_zero.csv")
    out_sub.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    print("(FIXED submission preserved.)")


if __name__ == "__main__":
    main()
