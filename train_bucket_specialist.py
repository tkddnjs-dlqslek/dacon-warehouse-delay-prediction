"""
Experiment B - Phase 2-a: LGB specialist for worst bucket (hub_spoke x late).

Data rule:
- k = bucket row count
- k >= 500: B*-only rows
- k < 500: same layout_type rows + sample_weight = (total / k) on B*-rows
(B* selected from Phase 1 diagnosis = hub_spoke x late, k=15615 >> 500 -> B*-only)

Model: LGB (MAE objective), log1p(y) target, v23 149 features.
CV: pre-computed GroupKFold(layout_id, n=5) from fold_idx.npy (filtered to B*).

Kill gate (Phase 2-a):
  specialist bucket-internal OOF MAE vs mega33 bucket-internal OOF MAE
  improvement < 0.3 -> ABORT.

Outputs:
  results/bucket_specialist/lgb_oof_b.npy (bucket-aligned OOF, shape = k)
  results/bucket_specialist/lgb_test_b.npy (bucket-aligned test pred, shape = test bucket rows)
  results/bucket_specialist/lgb_row_index_train.npy (train row idx into sorted train for B*)
  results/bucket_specialist/lgb_row_index_test.npy (test row idx for B*)
  results/bucket_specialist/lgb_summary.json
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT_DIR = os.path.join(ROOT, "results", "bucket_specialist")
os.makedirs(OUT_DIR, exist_ok=True)

BUCKET_LAYOUT = "hub_spoke"
BUCKET_TS = "late"  # ts_idx 16..24


def load_data():
    # Train (sorted for mega33 alignment)
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values

    # Mega33 OOF for comparison (same sort order)
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]

    # v23 FE cache (train) — already built on sorted train
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    # test fe
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f)
    test_fe = test_fe.reset_index(drop=True)

    # Pre-computed fold assignments aligned to sorted train
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))
    assert len(fold_idx) == len(train), "fold_idx length mismatch"

    # layout_type
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    train = train.merge(
        layout_info[["layout_id", "layout_type"]], on="layout_id", how="left"
    )

    # test layout_type via merge on layout_id (no row-order assumption)
    test_fe = test_fe.merge(
        layout_info[["layout_id", "layout_type"]], on="layout_id", how="left"
    )
    assert test_fe["layout_type"].isna().sum() == 0, "layout_type has NaN on test"

    # Build ts_idx (row within scenario; 25 rows per scenario after sort)
    train["_ts_idx"] = np.tile(np.arange(25), len(train) // 25)
    # Test: same structure (25 rows per scenario, sorted by layout_id, scenario_id inside cache? check)
    # Check if test_fe is sorted by (layout_id, scenario_id)
    test_sorted_check = (
        test_fe[["layout_id", "scenario_id"]].values
        == test_fe.sort_values(["layout_id", "scenario_id"])[["layout_id", "scenario_id"]].values
    ).all()
    if not test_sorted_check:
        # Re-sort
        print("[WARN] test_fe not sorted — sorting now")
        test_fe = test_fe.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
    test_fe["_ts_idx"] = np.tile(np.arange(25), len(test_fe) // 25)

    # ts_bucket
    def _ts_bucket(ix):
        return np.where(ix <= 7, "early", np.where(ix <= 15, "mid", "late"))

    train["_ts_bucket"] = _ts_bucket(train["_ts_idx"].values)
    test_fe["_ts_bucket"] = _ts_bucket(test_fe["_ts_idx"].values)

    return train, y, mega_oof, mega_test, train_fe, feat_cols, test_fe, fold_idx


def main():
    print("=" * 60)
    print("Experiment B Phase 2-a: LGB Specialist (hub_spoke x late)")
    print("=" * 60)

    train, y, mega_oof, mega_test, train_fe, feat_cols, test_fe, fold_idx = load_data()

    # Select B* mask on train
    train_mask = (train["layout_type"] == BUCKET_LAYOUT) & (
        train["_ts_bucket"] == BUCKET_TS
    )
    test_mask = (test_fe["layout_type"] == BUCKET_LAYOUT) & (
        test_fe["_ts_bucket"] == BUCKET_TS
    )
    train_b_idx = np.where(train_mask)[0]
    test_b_idx = np.where(test_mask)[0]
    k_train = len(train_b_idx)
    k_test = len(test_b_idx)
    print(f"B* train rows: {k_train} (~{k_train/len(train)*100:.2f}%)")
    print(f"B* test rows:  {k_test} (~{k_test/len(test_fe)*100:.2f}%)")
    assert k_train >= 500, "Data rule: expected k>=500 for B*-only training"

    # Bucket-internal mega33 baseline MAE
    mega_bucket_mae = np.mean(np.abs(y[train_b_idx] - mega_oof[train_b_idx]))
    print(f"mega33 bucket-internal OOF MAE: {mega_bucket_mae:.4f}")

    # Build matrices on B* subset
    X_all = train_fe[feat_cols].values
    X_b = X_all[train_b_idx]
    y_b = y[train_b_idx]
    fold_b = fold_idx[train_b_idx]
    # log1p target
    y_b_log = np.log1p(np.clip(y_b, 0, None))
    X_test_b = test_fe[feat_cols].values[test_b_idx]

    # Fold coverage check
    fold_counts = pd.Series(fold_b).value_counts().sort_index()
    print(f"fold counts in B*: {fold_counts.to_dict()}")
    # NOTE: fold_idx is GroupKFold(layout_id). Within hub_spoke, layouts split into folds.
    # Some folds might have 0 rows if hub_spoke layouts aren't in every fold.
    folds_present = sorted(fold_counts.index.tolist())
    print(f"folds present: {folds_present}")

    # Train LGB per fold (only folds that have val rows)
    oof_b = np.full(k_train, np.nan, dtype=np.float64)
    test_preds = []
    params = dict(
        objective="regression_l1",
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=40,
        lambda_l2=1.0,
        seed=42,
        verbose=-1,
    )

    for f in folds_present:
        val_mask = fold_b == f
        tr_mask = ~val_mask
        if val_mask.sum() == 0 or tr_mask.sum() < 100:
            print(f"[fold {f}] skipped (val={val_mask.sum()}, tr={tr_mask.sum()})")
            continue
        dtrain = lgb.Dataset(X_b[tr_mask], label=y_b_log[tr_mask])
        dval = lgb.Dataset(X_b[val_mask], label=y_b_log[val_mask], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        pred_val_log = model.predict(X_b[val_mask])
        pred_val = np.expm1(pred_val_log)
        oof_b[val_mask] = pred_val
        pred_test_log = model.predict(X_test_b)
        test_preds.append(np.expm1(pred_test_log))
        mae_f = np.mean(np.abs(pred_val - y_b[val_mask]))
        print(f"[fold {f}] val={val_mask.sum()} tr={tr_mask.sum()} best_iter={model.best_iteration} bucket-MAE={mae_f:.4f}")

    # If some rows not covered (shouldn't happen with 5 folds but safety)
    uncovered = np.isnan(oof_b).sum()
    if uncovered > 0:
        print(f"[WARN] {uncovered} rows not covered by any fold. Filling with mega33 OOF.")
        oof_b[np.isnan(oof_b)] = mega_oof[train_b_idx][np.isnan(oof_b)]

    # Aggregate test prediction (mean across folds)
    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)

    # Specialist bucket-internal MAE
    spec_bucket_mae = float(np.mean(np.abs(y_b - oof_b)))
    improvement = mega_bucket_mae - spec_bucket_mae
    print(f"\n=== Phase 2-a results ===")
    print(f"mega33 bucket-internal MAE:  {mega_bucket_mae:.4f}")
    print(f"specialist bucket-internal:  {spec_bucket_mae:.4f}")
    print(f"improvement:                 {improvement:+.4f}")
    print(f"kill gate: >= 0.30 to pass")

    pass_gate = bool(improvement >= 0.30)
    print(f"kill gate verdict: {'PASS' if pass_gate else 'ABORT'}")

    # Whole-train OOF impact (hard-gated replacement)
    hybrid_oof = mega_oof.copy()
    hybrid_oof[train_b_idx] = oof_b
    global_mae_mega = float(np.mean(np.abs(y - mega_oof)))
    global_mae_hybrid = float(np.mean(np.abs(y - hybrid_oof)))
    global_delta = global_mae_hybrid - global_mae_mega
    print(f"\nWhole-train OOF impact (mega33 baseline only, no ranking blend):")
    print(f"  mega33 OOF MAE:          {global_mae_mega:.4f}")
    print(f"  mega33+specialist hybrid:{global_mae_hybrid:.4f}")
    print(f"  delta vs mega33:         {global_delta:+.4f}")

    # Save artifacts
    np.save(os.path.join(OUT_DIR, "lgb_oof_b.npy"), oof_b)
    np.save(os.path.join(OUT_DIR, "lgb_test_b.npy"), test_pred)
    np.save(os.path.join(OUT_DIR, "lgb_row_index_train.npy"), train_b_idx)
    np.save(os.path.join(OUT_DIR, "lgb_row_index_test.npy"), test_b_idx)
    summary = dict(
        bucket_layout=BUCKET_LAYOUT,
        bucket_ts=BUCKET_TS,
        k_train=int(k_train),
        k_test=int(k_test),
        mega_bucket_mae=float(mega_bucket_mae),
        spec_bucket_mae=spec_bucket_mae,
        bucket_improvement=float(improvement),
        kill_gate_pass=pass_gate,
        global_mae_mega=global_mae_mega,
        global_mae_hybrid=global_mae_hybrid,
        global_delta_vs_mega=global_delta,
        folds_present=[int(f) for f in folds_present],
    )
    with open(os.path.join(OUT_DIR, "lgb_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_DIR}")
    if not pass_gate:
        print("[ABORT] Phase 2-a failed. Stop experiment B.")
    else:
        print("[PASS] Proceed to Phase 2-b or Phase 3.")


if __name__ == "__main__":
    main()
