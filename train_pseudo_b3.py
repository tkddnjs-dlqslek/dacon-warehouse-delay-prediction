"""
B3: Extended pseudo-label scan.
- unseen_pseudo w=1.5, 2.0, 2.5 (continue the monotonic trend check)
- all_pseudo w=0.3, 0.5, 1.0 (memory-efficient retry)

Reuse A baseline from saved OOF.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

OUT = "results/v24_cumsum"
TARGET = "avg_delay_minutes_next_30m"

print("=" * 60, flush=True)
print("B3: extended pseudo-label scan", flush=True)
print("=" * 60, flush=True)
t0 = time.time()

# Load
print("loading...", flush=True)
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
pseudo_labels = np.asarray(mega["meta_avg_test"])

train_layouts = set(train_fe["layout_id"].unique())
unseen_mask = ~np.isin(test_fe["layout_id"].values, list(train_layouts))

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te_all = test_fe[feat_cols].values.astype(np.float32)
X_te_u = X_te_all[unseen_mask]
pseudo_log_all = np.log1p(np.clip(pseudo_labels, 0, None)).astype(np.float32)
pseudo_log_u = pseudo_log_all[unseen_mask]
y_log = np.log1p(y)

del blob, test_fe, mega, pseudo_labels, train_fe
gc.collect()

# Reuse A baseline
oof_A = np.load(f"{OUT}/oof_raw_y_log1p_huber.npy")
mae_A = mean_absolute_error(y, oof_A)
print(f"A baseline reused: mae={mae_A:.5f}", flush=True)
print(f"X_tr={X_tr.shape} X_te_u={X_te_u.shape} X_te_all={X_te_all.shape} ({time.time()-t0:.0f}s)", flush=True)

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

# Reuse prior unseen results + add new weights
results = []
for pw, mae_prior in [(0.1, 8.595941), (0.3, 8.585284), (0.5, 8.571553), (1.0, 8.556892)]:
    results.append({"method": "unseen_pseudo", "weight": pw, "mae": mae_prior, "delta": mae_prior - mae_A})


def run_variant(X_pseudo, pl_pseudo, pw, tag):
    oof = np.zeros(len(y))
    n_pseudo = len(X_pseudo)
    for f in range(5):
        t1 = time.time()
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]
        n_real = len(tr)
        X_c = np.empty((n_real + n_pseudo, X_tr.shape[1]), dtype=np.float32)
        X_c[:n_real] = X_tr[tr]
        X_c[n_real:] = X_pseudo
        y_c = np.empty(n_real + n_pseudo, dtype=np.float32)
        y_c[:n_real] = y_log[tr]
        y_c[n_real:] = pl_pseudo
        w_c = np.empty(n_real + n_pseudo, dtype=np.float32)
        w_c[:n_real] = 1.0
        w_c[n_real:] = pw
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_c, y_c, sample_weight=w_c,
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
        fm = mean_absolute_error(y[val], oof[val])
        print(f"  {tag} fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-t1:.0f}s)", flush=True)
        del X_c, y_c, w_c, m
        gc.collect()
    mae = mean_absolute_error(y, oof)
    delta = mae - mae_A
    np.save(f"{OUT}/oof_pseudo_{tag}.npy", oof)
    return mae, delta


# unseen w=1.5, 2.0, 2.5
for pw in [1.5, 2.0, 2.5]:
    tag = f"unseen_w{pw}"
    print(f"\n--- {tag} ---", flush=True)
    tv = time.time()
    mae, delta = run_variant(X_te_u, pseudo_log_u, pw, tag)
    results.append({"method": "unseen_pseudo", "weight": pw, "mae": float(mae), "delta": float(delta)})
    print(f"{tag} OVERALL mae={mae:.5f} delta={delta:+.5f} ({time.time()-tv:.0f}s)", flush=True)
    pd.DataFrame(results).sort_values("mae").to_csv(f"{OUT}/pseudo_unseen_scan.csv", index=False)

# all_pseudo w=0.3, 0.5, 1.0
for pw in [0.3, 0.5, 1.0]:
    tag = f"all_w{pw}"
    print(f"\n--- {tag} ---", flush=True)
    tv = time.time()
    mae, delta = run_variant(X_te_all, pseudo_log_all, pw, tag)
    results.append({"method": "all_pseudo", "weight": pw, "mae": float(mae), "delta": float(delta)})
    print(f"{tag} OVERALL mae={mae:.5f} delta={delta:+.5f} ({time.time()-tv:.0f}s)", flush=True)
    pd.DataFrame(results).sort_values("mae").to_csv(f"{OUT}/pseudo_b3_scan.csv", index=False)

# final
df = pd.DataFrame(results).sort_values("mae").reset_index(drop=True)
df.to_csv(f"{OUT}/pseudo_b3_scan.csv", index=False)
print(f"\n=== ALL RESULTS (sorted by mae) ===", flush=True)
for _, r in df.iterrows():
    tag_ = " ***" if r["delta"] < -0.04 else (" **" if r["delta"] < -0.02 else "")
    print(f"  {r['method']:20s} w={r['weight']:.1f} mae={r['mae']:.5f} delta={r['delta']:+.5f}{tag_}", flush=True)

best = df.iloc[0]
v = "GO" if best["delta"] <= -0.06 else ("WEAK_GO" if best["delta"] < -0.02 else "NO_GO")
print(f"\n>>> VERDICT: {v} best={best['method']} w={best['weight']} delta={best['delta']:+.5f} ({time.time()-t0:.0f}s) <<<", flush=True)

json.dump({
    "verdict": v,
    "best_method": str(best["method"]),
    "best_weight": float(best["weight"]),
    "best_mae": float(best["mae"]),
    "best_delta": float(best["delta"]),
    "baseline_A": float(mae_A),
    "elapsed": round(time.time() - t0, 1),
}, open(f"{OUT}/pseudo_b3_summary.json", "w"), indent=2)
