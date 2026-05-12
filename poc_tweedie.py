"""
POC: Tweedie loss (compound Poisson-Gamma) on v23 features.

y == 0 spike 2.73% + continuous positive → Tweedie natural fit.
Try multiple variance_power values.

Objective: objective='tweedie', tweedie_variance_power in (1.1, 1.3, 1.5, 1.7, 1.9)
LGB uses log-link internally. Target = raw y (non-negative).
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
OUT = os.path.join(ROOT, "results", "tweedie")
os.makedirs(OUT, exist_ok=True)

VARIANCE_POWERS = [1.1, 1.3, 1.5, 1.7, 1.9]


def train_tweedie(X, y, X_test, fold_idx, variance_power, name=""):
    oof = np.zeros(len(y), dtype=np.float64)
    test_preds = []
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        model = lgb.LGBMRegressor(
            objective="tweedie",
            tweedie_variance_power=variance_power,
            learning_rate=0.03,
            n_estimators=5000,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        model.fit(
            X[tr_mask], y[tr_mask],  # raw y, LGB tweedie uses log-link internally
            eval_set=[(X[val_mask], y[val_mask])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_mask] = np.clip(model.predict(X[val_mask]), 0, None)
        test_preds.append(np.clip(model.predict(X_test), 0, None))
        fold_mae = np.mean(np.abs(oof[val_mask] - y[val_mask]))
        print(f"  [{name}] fold {f}: best_iter={model.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
    test = np.mean(np.stack(test_preds, axis=0), axis=0)
    return oof, test


def main():
    print("=" * 60, flush=True)
    print("Tweedie POC: compound Poisson-Gamma on v23", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    print(f"y stats: mean={y.mean():.3f}, zeros={int((y==0).sum())} ({(y==0).mean()*100:.2f}%)", flush=True)

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

    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"Reference v23 LGB_Huber OOF: {old_mae:.5f}", flush=True)

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))
    print(f"mega33 OOF: {mega_mae:.5f}", flush=True)

    best_result = None
    results = []

    for vp in VARIANCE_POWERS:
        print(f"\n=== Training Tweedie variance_power={vp} ===", flush=True)
        oof, test = train_tweedie(X, y, X_test, fold_idx, vp, name=f"tw{vp}")
        tw_mae = float(mean_absolute_error(y, oof))
        corr_v23 = float(np.corrcoef(y - old_oof, y - oof)[0, 1])
        corr_mega = float(np.corrcoef(y - mega_oof, y - oof)[0, 1])
        # zero prediction check
        zero_mae_tw = np.mean(np.abs(oof[y == 0]))
        zero_mae_mega = np.mean(np.abs(mega_oof[y == 0]))
        print(f"  [tw{vp}] overall MAE: {tw_mae:.5f}", flush=True)
        print(f"  [tw{vp}] corr(v23): {corr_v23:.4f}, corr(mega33): {corr_mega:.4f}", flush=True)
        print(f"  [tw{vp}] y==0 pred mean: tweedie={zero_mae_tw:.3f}, mega33={zero_mae_mega:.3f}", flush=True)

        # Blend with mega33
        best_w = 0; best_bmae = mega_mae
        for w in np.linspace(0, 0.4, 41):
            pred = (1 - w) * mega_oof + w * oof
            m = float(np.mean(np.abs(y - pred)))
            if m < best_bmae:
                best_bmae = m; best_w = w
        print(f"  [tw{vp}] blend: w={best_w:.3f}, MAE={best_bmae:.5f}, delta={best_bmae - mega_mae:+.5f}", flush=True)

        results.append(dict(
            vp=vp, single_mae=tw_mae, corr_v23=corr_v23, corr_mega=corr_mega,
            zero_mae_tw=float(zero_mae_tw), zero_mae_mega=float(zero_mae_mega),
            blend_w=best_w, blend_mae=best_bmae, blend_delta=best_bmae - mega_mae,
        ))
        np.save(os.path.join(OUT, f"tw{int(vp*10)}_oof.npy"), oof)
        np.save(os.path.join(OUT, f"tw{int(vp*10)}_test.npy"), test)

        if best_result is None or best_bmae < best_result["blend_mae"]:
            best_result = results[-1].copy()

    # Summary
    print("\n" + "=" * 60, flush=True)
    print("Summary:", flush=True)
    for r in results:
        print(f"  vp={r['vp']}: single={r['single_mae']:.5f}, corr_mega={r['corr_mega']:.4f}, blend_w={r['blend_w']:.3f}, blend_delta={r['blend_delta']:+.5f}", flush=True)
    print(f"\nBest: vp={best_result['vp']}, blend_delta={best_result['blend_delta']:+.5f}", flush=True)

    summary = dict(results=results, best=best_result, mega_mae=mega_mae, v23_mae=old_mae)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
