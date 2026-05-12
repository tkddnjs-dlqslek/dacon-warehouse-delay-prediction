"""
Q-sweep: Q60, Q65, Q70, Q80 quantile regression.

Cheap check to verify whether a subtle upward shift (Q60/Q65) beats Q50/mega33
on MAE, or whether the gap is wider between median and tail.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "q_sweep")
os.makedirs(OUT, exist_ok=True)

ALPHAS = [0.55, 0.60, 0.65, 0.70, 0.80]


def main():
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
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    X = train_fe[feat_cols].values
    y_log = np.log1p(np.clip(y, 0, None))

    mega_mae = float(np.mean(np.abs(y - mega_oof)))
    print(f"mega33 OOF MAE: {mega_mae:.4f}")

    summary = {}
    for alpha in ALPHAS:
        oof_log = np.zeros(len(y), dtype=np.float64)
        params = dict(
            objective="quantile",
            alpha=alpha,
            learning_rate=0.05,
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
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[lgb.early_stopping(80), lgb.log_evaluation(0)],
            )
            oof_log[val_mask] = model.predict(X[val_mask])
        q_oof = np.expm1(oof_log)
        mae_all = float(np.mean(np.abs(y - q_oof)))
        corr = float(np.corrcoef(y - mega_oof, y - q_oof)[0, 1])

        # Blend with mega33
        best_blend_w = 0.0
        best_blend_mae = mega_mae
        for w in np.linspace(0, 1, 21):
            pred = (1 - w) * mega_oof + w * q_oof
            m = float(np.mean(np.abs(y - pred)))
            if m < best_blend_mae:
                best_blend_mae = m
                best_blend_w = float(w)

        tail = y > 50
        tail_mae = float(np.mean(np.abs(y[tail] - q_oof[tail])))
        body_mae = float(np.mean(np.abs(y[~tail] - q_oof[~tail])))
        print(
            f"Q{int(alpha*100):02d}: MAE={mae_all:.4f}  "
            f"tail={tail_mae:.4f}  body={body_mae:.4f}  "
            f"resid_corr(mega)={corr:.4f}  "
            f"best_blend w={best_blend_w:.2f} MAE={best_blend_mae:.4f} delta={best_blend_mae-mega_mae:+.4f}"
        )
        summary[f"Q{int(alpha*100):02d}"] = dict(
            mae_all=mae_all,
            tail_mae=tail_mae,
            body_mae=body_mae,
            resid_corr=corr,
            best_blend_w=best_blend_w,
            best_blend_mae=best_blend_mae,
            delta_vs_mega=best_blend_mae - mega_mae,
        )
        np.save(os.path.join(OUT, f"q{int(alpha*100):02d}_oof.npy"), q_oof)

    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
