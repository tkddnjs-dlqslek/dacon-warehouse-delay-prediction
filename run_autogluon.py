"""
AutoGluon Tabular experiment.

Run AutoGluon on v23 features with time_limit 3600s (1 hour) per fold.
Compare OOF + residual_corr with mega33.

Note: on CPU this can be slow. We use only 3 folds instead of 5 to save time,
and use best_quality preset for max signal.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from autogluon.tabular import TabularPredictor

OUT = "results/autogluon"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("AutoGluon Tabular")
print("=" * 64)
t0 = time.time()

# Load
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

# Include target for AG format
data_tr = train_fe[feat_cols + [TARGET]].copy()
data_te = test_fe[feat_cols].copy()
print(f"data_tr: {data_tr.shape}, data_te: {data_te.shape}")

# We'll run 5-fold manually so we can compute residual_corr
# Time budget: total ~2.5 hours on CPU. 5 folds × 30 min each.
TIME_LIMIT = 1800  # 30 min per fold

ag_oof = np.zeros(len(y))
ag_test = np.zeros(len(data_te))
fold_rows = []

for f in range(5):
    tf = time.time()
    tr_idx = np.where(fold_ids != f)[0]
    val_idx = np.where(fold_ids == f)[0]

    fold_dir = f"{OUT}/fold_{f}_models"
    predictor = TabularPredictor(
        label=TARGET,
        eval_metric="mean_absolute_error",
        path=fold_dir,
        verbosity=2,
    ).fit(
        train_data=data_tr.iloc[tr_idx],
        time_limit=TIME_LIMIT,
        presets="medium_quality",  # faster than best_quality; good balance
        num_bag_folds=0,            # no inner bagging (we handle outer folds)
        num_stack_levels=0,         # no stacking (we do our own)
        excluded_model_types=["KNN"],  # KNN is slow/useless here
        hyperparameters={"GBM": {}, "XGB": {}, "CAT": {}, "NN_TORCH": {}, "FASTAI": {}},
    )

    pred_val = predictor.predict(data_tr.iloc[val_idx][feat_cols])
    pred_test = predictor.predict(data_te)
    ag_oof[val_idx] = np.clip(pred_val.values, 0, None)
    ag_test += np.clip(pred_test.values, 0, None) / 5

    fm = mean_absolute_error(y[val_idx], ag_oof[val_idx])
    fold_rows.append({"fold": f, "mae": float(fm), "time_min": round((time.time()-tf)/60, 1)})
    print(f"\nfold {f}: mae={fm:.5f} ({(time.time()-tf)/60:.1f}min)")

    # save intermediate
    np.save(f"{OUT}/ag_oof.npy", ag_oof)
    np.save(f"{OUT}/ag_test.npy", ag_test)
    pd.DataFrame(fold_rows).to_csv(f"{OUT}/ag_fold_metrics.csv", index=False)
    del predictor
    gc.collect()

ag_mae = mean_absolute_error(y, ag_oof)
print(f"\nAutoGluon OOF: {ag_mae:.5f}")

# Diversity check
r_mega = y - mega33_oof
r_ag = y - ag_oof
corr = float(np.corrcoef(r_mega, r_ag)[0, 1])
print(f"residual_corr(mega33, AG): {corr:.4f}")

best_w, best_mae = 0, float(baseline_mae)
for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    blend = (1 - w) * mega33_oof + w * ag_oof
    mae = float(mean_absolute_error(y, blend))
    if mae < best_mae: best_w, best_mae = w, mae
    print(f"  blend w={w:.2f}: mae={mae:.5f} delta={mae-baseline_mae:+.5f}")

delta = best_mae - baseline_mae
verdict = "STRONG_GO" if delta < -0.02 else ("GO" if delta < -0.008 else ("WEAK" if delta < -0.003 else "NO_GO"))

if delta < -0.003:
    blend_test = (1 - best_w) * mega33_test + best_w * ag_test
    test_raw = pd.read_csv("test.csv")
    pd.DataFrame({"ID": test_raw["ID"].values, TARGET: np.clip(blend_test, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_ag_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission: {OUT}/submission_mega33_ag_w{int(best_w*100)}.csv")

json.dump({
    "ag_oof_mae": float(ag_mae),
    "residual_corr": corr,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(delta),
    "verdict": verdict,
    "fold_rows": fold_rows,
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/ag_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}  corr={corr:.4f}  delta={delta:+.5f}")
print(f"elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
