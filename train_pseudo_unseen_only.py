"""
Pseudo-labeling v2 (unseen-only):
Add mega33 pseudo-labels ONLY for unseen-layout test rows (20K) to training.
Memory-efficient: incremental checkpoint save, gc after each variant.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

OUT = "results/v24_cumsum"
TARGET = "avg_delay_minutes_next_30m"
CKPT = f"{OUT}/pseudo_unseen_progress.json"

print("="*60, flush=True)
print("Pseudo-labeling v2 (unseen-only)", flush=True)
print("="*60, flush=True)
t0 = time.time()

# Load
print("loading data...", flush=True)
with open("results/eda_v30/v30_fe_cache.pkl","rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl","rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)
with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
pseudo_labels = np.asarray(mega["meta_avg_test"])

train_layouts = set(train_fe["layout_id"].unique())
unseen_mask = ~np.isin(test_fe["layout_id"].values, list(train_layouts))

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te_unseen = test_fe[feat_cols].values[unseen_mask].astype(np.float32)
y_log = np.log1p(y)
pseudo_log_unseen = np.log1p(np.clip(pseudo_labels[unseen_mask], 0, None)).astype(np.float32)

del blob, test_fe, mega, pseudo_labels, train_fe
gc.collect()

print(f"X_tr={X_tr.shape} X_te_unseen={X_te_unseen.shape} loaded={time.time()-t0:.0f}s", flush=True)

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

results = []

# A baseline (reuse if exists)
oof_A_path = f"{OUT}/oof_raw_y_log1p_huber.npy"
try:
    oof_A = np.load(oof_A_path)
    mae_A = mean_absolute_error(y, oof_A)
    print(f"reused A baseline oof from {oof_A_path}: {mae_A:.5f}", flush=True)
except Exception:
    print("training A (no pseudo baseline)...", flush=True)
    oof_A = np.zeros(len(y))
    for f in range(5):
        t1 = time.time()
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_tr[tr], y_log[tr],
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_A[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
        print(f"  A fold {f}: it={m.best_iteration_} ({time.time()-t1:.0f}s)", flush=True)
        del m; gc.collect()
    mae_A = mean_absolute_error(y, oof_A)
    np.save(oof_A_path, oof_A)
    print(f"A baseline: {mae_A:.5f} ({time.time()-t0:.0f}s)", flush=True)

# Pseudo variants
for pw in [0.1, 0.3, 0.5, 1.0]:
    name = f"unseen_w{pw}"
    print(f"\n--- {name} ---", flush=True)
    oof_P = np.zeros(len(y))
    t_var = time.time()
    for f in range(5):
        t1 = time.time()
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]

        # build combined train in place
        n_tr_real = len(tr)
        n_pseudo = len(X_te_unseen)
        X_comb = np.empty((n_tr_real + n_pseudo, X_tr.shape[1]), dtype=np.float32)
        X_comb[:n_tr_real] = X_tr[tr]
        X_comb[n_tr_real:] = X_te_unseen
        y_comb = np.empty(n_tr_real + n_pseudo, dtype=np.float32)
        y_comb[:n_tr_real] = y_log[tr]
        y_comb[n_tr_real:] = pseudo_log_unseen
        w_comb = np.empty(n_tr_real + n_pseudo, dtype=np.float32)
        w_comb[:n_tr_real] = 1.0
        w_comb[n_tr_real:] = pw

        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_comb, y_comb, sample_weight=w_comb,
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof_P[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)

        # checkpoint per fold
        fold_mae = mean_absolute_error(y[val], oof_P[val])
        print(f"  {name} fold {f}: mae={fold_mae:.5f} it={m.best_iteration_} ({time.time()-t1:.0f}s)", flush=True)
        np.save(f"{OUT}/oof_pseudo_{name}.npy", oof_P)
        del X_comb, y_comb, w_comb, m; gc.collect()

    mae_P = mean_absolute_error(y, oof_P)
    delta = mae_P - mae_A
    results.append({"method": "unseen_pseudo", "weight": pw, "mae": float(mae_P), "delta": float(delta)})
    tag = " ***" if delta < -0.01 else ""
    print(f"{name} OVERALL: mae={mae_P:.5f} delta={delta:+.5f} ({time.time()-t_var:.0f}s){tag}", flush=True)

    # persistent summary after each variant
    df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
    df.to_csv(f"{OUT}/pseudo_unseen_scan.csv", index=False)
    json.dump({"results": results, "baseline_A": float(mae_A), "elapsed": round(time.time()-t0,1)},
              open(CKPT, "w"), indent=2)
    del oof_P; gc.collect()

best = pd.DataFrame(results).sort_values("mae").iloc[0]
v = "GO" if best["delta"] <= -0.06 else ("WEAK" if best["delta"] < -0.01 else "NO_GO")
print(f"\n>>> VERDICT: {v} best_delta={best['delta']:+.5f} elapsed={time.time()-t0:.0f}s <<<", flush=True)

json.dump({"verdict": v, "best_method": str(best["method"]), "best_weight": float(best["weight"]),
           "best_mae": float(best["mae"]), "best_delta": float(best["delta"]),
           "baseline_A": float(mae_A), "elapsed": round(time.time()-t0,1)},
          open(f"{OUT}/pseudo_v2_summary.json", "w"), indent=2)
