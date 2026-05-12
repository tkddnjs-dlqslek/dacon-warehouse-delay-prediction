"""
Phase 2: Q90 quantile regression.

LGB objective='quantile', alpha=0.9, log1p(y) target.
v23 149 features, GroupKFold 5-fold.

Kill gate: Q90 OOF on y>50 subset doesn't beat mega33 bucket-MAE -> discard.
(Expected: Q90 predicts higher, so |y - Q90| on tail should be smaller than |y - mega33|.)
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "q90")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60)
    print("Phase 2: Q90 quantile regression")
    print("=" * 60)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values
    y_log = np.log1p(np.clip(y, 0, None))

    oof_log = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    params = dict(
        objective="quantile",
        alpha=0.9,
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
        dtrain = lgb.Dataset(X[tr_mask], label=y_log[tr_mask])
        dval = lgb.Dataset(X[val_mask], label=y_log[val_mask], reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=3000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)],
        )
        oof_log[val_mask] = model.predict(X[val_mask])
        test_preds.append(model.predict(X_test))
        mae_log = float(np.mean(np.abs(oof_log[val_mask] - y_log[val_mask])))
        print(f"  fold {f}: best_iter={model.best_iteration} val_pinball_log={mae_log:.4f}")
    q90_oof = np.expm1(oof_log)
    q90_test = np.expm1(np.mean(np.stack(test_preds, axis=0), axis=0))

    # Overall / tail comparison
    tail = y > 50
    q90_mae_all = float(np.mean(np.abs(y - q90_oof)))
    mega_mae_all = float(np.mean(np.abs(y - mega_oof)))
    q90_mae_tail = float(np.mean(np.abs(y[tail] - q90_oof[tail])))
    mega_mae_tail = float(np.mean(np.abs(y[tail] - mega_oof[tail])))
    q90_mae_body = float(np.mean(np.abs(y[~tail] - q90_oof[~tail])))
    mega_mae_body = float(np.mean(np.abs(y[~tail] - mega_oof[~tail])))
    print(f"\n=== Q90 OOF metrics ===")
    print(f"Q90 overall MAE: {q90_mae_all:.4f}   (mega33: {mega_mae_all:.4f})")
    print(f"Q90 tail MAE:    {q90_mae_tail:.4f}   (mega33: {mega_mae_tail:.4f})  improvement: {mega_mae_tail - q90_mae_tail:+.4f}")
    print(f"Q90 body MAE:    {q90_mae_body:.4f}   (mega33: {mega_mae_body:.4f})  delta: {q90_mae_body - mega_mae_body:+.4f}")

    # Residual correlation (Q90 as candidate in blend)
    resid_mega = y - mega_oof
    resid_q90 = y - q90_oof
    corr = float(np.corrcoef(resid_mega, resid_q90)[0, 1])
    print(f"residual_corr(mega33, Q90): {corr:.4f}")

    # Kill gate: Q90 must beat mega on tail
    pass_gate = q90_mae_tail < mega_mae_tail
    print(f"\nKill gate (Q90 tail MAE < mega tail MAE): {'PASS' if pass_gate else 'ABORT'}")

    np.save(os.path.join(OUT, "q90_oof.npy"), q90_oof)
    np.save(os.path.join(OUT, "q90_test.npy"), q90_test)
    summary = dict(
        q90_mae_all=q90_mae_all,
        mega_mae_all=mega_mae_all,
        q90_mae_tail=q90_mae_tail,
        mega_mae_tail=mega_mae_tail,
        q90_mae_body=q90_mae_body,
        mega_mae_body=mega_mae_body,
        residual_corr_vs_mega=corr,
        kill_gate_pass=bool(pass_gate),
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not pass_gate:
        print("[ABORT] Q90 does not improve tail. Reconsider (maybe Q95 / Q99).")


if __name__ == "__main__":
    main()
