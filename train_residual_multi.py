"""
Experiment A (reduced): Residual 3-class classification.

Setup:
  r = y - mega33_oof
  classes via quantile thresholds (lo, hi) = np.quantile(r, [0.33, 0.67])
    under: r < lo
    neutral
    over : r > hi
  LGB multiclass, v23 149 features, GroupKFold(layout_id, n=5) via fold_idx.npy
  OOF class probabilities -> (p_under, p_over)

Correction:
  scale = np.median(np.abs(r))
  corrected(mega33) = mega33 + alpha * (p_over - p_under) * scale
  alpha grid = {0, 0.1, 0.3, 0.5, 1.0}
  pick alpha by OOF MAE minimization.

Kill gates:
  alpha_best < 0.1  -> discard
  residual_corr(mega33, corrected) > 0.97 -> discard (no new signal)
  OOF delta > -0.001 (vs mega33 only) -> discard
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT_DIR = os.path.join(ROOT, "results", "residual_multi")
os.makedirs(OUT_DIR, exist_ok=True)

ALPHA_GRID = [0.0, 0.1, 0.3, 0.5, 1.0]


def main():
    print("=" * 60)
    print("Experiment A: Residual 3-class classification")
    print("=" * 60)

    # Load sorted train + mega33
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_test = mega["meta_avg_test"]
    assert len(mega_oof) == len(train)

    # v23 FE cache
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)

    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    # Residual & class labels
    r = y - mega_oof
    lo, hi = np.quantile(r, [0.33, 0.67])
    print(f"residual mean={r.mean():.3f}, median={np.median(r):.3f}, std={r.std():.3f}")
    print(f"class thresholds: lo={lo:.4f}, hi={hi:.4f}")
    cls = np.where(r < lo, 0, np.where(r > hi, 2, 1)).astype(np.int32)
    class_counts = pd.Series(cls).value_counts().sort_index()
    print(f"class counts: {class_counts.to_dict()}")

    scale = float(np.median(np.abs(r)))
    print(f"scale = median|r| = {scale:.4f}")

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values

    oof_proba = np.zeros((len(train), 3), dtype=np.float64)
    test_proba_folds = []

    params = dict(
        objective="multiclass",
        num_class=3,
        learning_rate=0.03,
        num_leaves=63,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=80,
        lambda_l2=1.0,
        seed=42,
        verbose=-1,
    )

    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        print(f"[fold {f}] val={val_mask.sum()} tr={tr_mask.sum()}")
        dtrain = lgb.Dataset(X[tr_mask], label=cls[tr_mask])
        dval = lgb.Dataset(X[val_mask], label=cls[val_mask], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
        )
        p_val = model.predict(X[val_mask])
        oof_proba[val_mask] = p_val
        p_test = model.predict(X_test)
        test_proba_folds.append(p_test)
        # fold-level accuracy (sanity)
        acc = float((p_val.argmax(1) == cls[val_mask]).mean())
        print(f"[fold {f}] best_iter={model.best_iteration} val_acc={acc:.4f}")

    test_proba = np.mean(np.stack(test_proba_folds, axis=0), axis=0)

    # Correction search
    p_under = oof_proba[:, 0]
    p_over = oof_proba[:, 2]
    delta_signal = (p_over - p_under) * scale
    mega_mae = float(np.mean(np.abs(y - mega_oof)))
    print(f"\nmega33 OOF MAE: {mega_mae:.4f}")
    print(f"alpha search:")
    results = []
    best_alpha = 0.0
    best_mae = mega_mae
    for alpha in ALPHA_GRID:
        corrected = mega_oof + alpha * delta_signal
        mae_c = float(np.mean(np.abs(y - corrected)))
        results.append((alpha, mae_c))
        print(f"  alpha={alpha:>4}: OOF MAE={mae_c:.4f}  delta={mae_c-mega_mae:+.4f}")
        if mae_c < best_mae:
            best_mae = mae_c
            best_alpha = alpha
    oof_delta = best_mae - mega_mae
    print(f"\nbest alpha={best_alpha}, best OOF MAE={best_mae:.4f}, delta={oof_delta:+.4f}")

    # Residual correlation (mega33 vs corrected)
    corrected_best = mega_oof + best_alpha * delta_signal
    resid_mega = y - mega_oof
    resid_corr_c = y - corrected_best
    if np.std(resid_corr_c) < 1e-9:
        corr = 1.0
    else:
        corr = float(np.corrcoef(resid_mega, resid_corr_c)[0, 1])
    print(f"residual_corr(mega33, corrected): {corr:.4f}")

    # Kill gates
    gate_alpha_fail = best_alpha < 0.1
    gate_corr_fail = corr > 0.97
    gate_oof_fail = oof_delta > -0.001
    verdict = "DISCARD" if (gate_alpha_fail or gate_corr_fail or gate_oof_fail) else "PASS"
    print("\nKill gate check:")
    print(f"  alpha_best < 0.1   ? {gate_alpha_fail}  (alpha={best_alpha})")
    print(f"  resid_corr > 0.97  ? {gate_corr_fail}  (corr={corr:.4f})")
    print(f"  oof_delta > -0.001 ? {gate_oof_fail}  (delta={oof_delta:+.4f})")
    print(f"verdict: {verdict}")

    # Save artifacts
    np.save(os.path.join(OUT_DIR, "oof_proba.npy"), oof_proba)
    np.save(os.path.join(OUT_DIR, "test_proba.npy"), test_proba)
    summary = dict(
        lo=float(lo),
        hi=float(hi),
        scale=scale,
        class_counts={int(k): int(v) for k, v in class_counts.items()},
        alpha_grid=ALPHA_GRID,
        alpha_mae={str(a): mae for a, mae in results},
        best_alpha=best_alpha,
        best_oof_mae=best_mae,
        mega_oof_mae=mega_mae,
        oof_delta=oof_delta,
        residual_corr_corrected=corr,
        kill_gate_alpha_fail=gate_alpha_fail,
        kill_gate_corr_fail=gate_corr_fail,
        kill_gate_oof_fail=gate_oof_fail,
        verdict=verdict,
    )
    with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {OUT_DIR}")


if __name__ == "__main__":
    main()
