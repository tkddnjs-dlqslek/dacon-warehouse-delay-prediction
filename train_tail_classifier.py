"""
Phase 1: Extreme tail binary classifier.

Targets: y > 30, y > 50, y > 100 (three variants).
Model: LGB binary, scale_pos_weight = (neg/pos), v23 149 features.
CV: 5-fold GroupKFold via fold_idx.npy.

Kill gate: AUC < 0.75 for the primary threshold (y>50) -> discard.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "tail_cls")
os.makedirs(OUT, exist_ok=True)


def train_binary(X, y_bin, fold_idx, X_test, name):
    n = len(y_bin)
    pos = int(y_bin.sum())
    neg = n - pos
    spw = max(1.0, neg / max(pos, 1))
    oof = np.zeros(n, dtype=np.float64)
    test_preds = []
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
        auc_f = roc_auc_score(y_bin[val_mask], oof[val_mask])
        print(f"  [{name}] fold {f}: best_iter={model.best_iteration} AUC={auc_f:.4f}")
    test_pred = np.mean(np.stack(test_preds, axis=0), axis=0)
    auc = float(roc_auc_score(y_bin, oof))
    ap = float(average_precision_score(y_bin, oof))
    print(f"  [{name}] overall AUC={auc:.4f}  AP={ap:.4f}  pos_rate={pos/n*100:.2f}%")
    return oof, test_pred, auc, ap, pos / n


def main():
    print("=" * 60)
    print("Phase 1: Extreme tail classifier")
    print("=" * 60)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    X = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values

    summary = {}
    for thresh in [30, 50, 100]:
        y_bin = (y > thresh).astype(np.int32)
        name = f"gt{thresh}"
        print(f"\n--- training y>{thresh} (pos_rate={y_bin.mean()*100:.2f}%) ---")
        oof, test_pred, auc, ap, rate = train_binary(X, y_bin, fold_idx, X_test, name)
        np.save(os.path.join(OUT, f"p_{name}_oof.npy"), oof)
        np.save(os.path.join(OUT, f"p_{name}_test.npy"), test_pred)
        summary[name] = dict(thresh=thresh, auc=auc, ap=ap, pos_rate=rate)

    # Kill gate on primary threshold (y>50)
    auc50 = summary["gt50"]["auc"]
    pass_gate = bool(auc50 >= 0.75)
    summary["kill_gate_pass"] = pass_gate
    summary["primary_auc"] = auc50
    print("\n" + "=" * 60)
    print(f"Primary AUC (y>50): {auc50:.4f}")
    print(f"Kill gate (>= 0.75): {'PASS' if pass_gate else 'ABORT'}")
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if not pass_gate:
        print("[ABORT] Phase 1 failed. Do not proceed.")
    else:
        print("[PASS] Proceed to Phase 2.")


if __name__ == "__main__":
    main()
