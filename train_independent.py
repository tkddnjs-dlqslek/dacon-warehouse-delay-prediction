"""
Independent track: build predictor WITHOUT referencing v23/mega33 at all.

Features source:
- Raw 90 columns (ONLY)
- Time-series frequency domain: FFT amplitudes/phases on 25-step sequences
- Time-series statistical: skew, kurtosis, autocorr, trend slope
- Cyclical timeslot encoding
- Layout static info (raw, no ratios)

NO v23 lag/rolling/SC/forward-looking.
NO residual correction.
NO mega33 as input.

At the very end (only), measure corr with mega33 as diagnostic.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from scipy.stats import skew, kurtosis
import lightgbm as lgb

OUT = "results/independent"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Independent Track - NO v23, NO mega33 reference")
print("=" * 64)
t0 = time.time()

# --- Load RAW data ---
train = pd.read_csv("train.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
test = pd.read_csv("test.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
layout = pd.read_csv("layout_info.csv")
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
print(f"train: {train.shape}, test: {test.shape}")

train["timeslot"] = train.groupby(["layout_id", "scenario_id"]).cumcount()
test["timeslot"] = test.groupby(["layout_id", "scenario_id"]).cumcount()

# Merge layout (but DON'T drop layout_type — keep as static)
train = train.merge(layout, on="layout_id", how="left")
test = test.merge(layout, on="layout_id", how="left")

# Identify numeric raw cols (exclude ID/target/categorical)
exclude = {"ID", "layout_id", "scenario_id", TARGET, "layout_type"}
raw_cols = [c for c in train.columns if c not in exclude and pd.api.types.is_numeric_dtype(train[c])]
print(f"raw numeric cols: {len(raw_cols)}")

# Key cols for heavy time-series feature extraction (to limit compute)
key_cols_for_ts = [
    "order_inflow_15m", "congestion_score", "robot_utilization",
    "battery_mean", "fault_count_15m", "blocked_path_15m",
    "pack_utilization", "charge_queue_length",
    "robot_active", "robot_idle", "robot_charging",
    "low_battery_ratio", "near_collision_15m",
    "avg_trip_distance", "staging_area_util", "loading_dock_util",
    "unique_sku_15m", "avg_items_per_order"
]
key_cols_for_ts = [c for c in key_cols_for_ts if c in train.columns]
print(f"time-series feature cols: {len(key_cols_for_ts)}")


def extract_ts_features(df, cols, name_prefix):
    """For each (layout_id, scenario_id) group, compute time-series features per col.
    Returns scenario-level features DataFrame indexed by (layout_id, scenario_id)."""
    print(f"  extracting TS features for {len(cols)} cols...")
    t_start = time.time()
    # Group and compute
    result = []
    grp = df.groupby(["layout_id", "scenario_id"], sort=False)
    for (lid, sid), g in grp:
        g = g.sort_values("timeslot")
        row = {"layout_id": lid, "scenario_id": sid}
        for c in cols:
            s = g[c].fillna(g[c].mean() if not g[c].isna().all() else 0).values.astype(np.float64)
            if len(s) < 5 or s.std() < 1e-6:
                # not enough data or constant
                row[f"{c}_fft1"] = 0
                row[f"{c}_fft2"] = 0
                row[f"{c}_fft3"] = 0
                row[f"{c}_trend"] = 0
                row[f"{c}_ac1"] = 0
                row[f"{c}_peak_pos"] = 0
                row[f"{c}_peak_mag"] = 0
                row[f"{c}_skew"] = 0
                row[f"{c}_range"] = 0
                continue
            # FFT amplitudes
            fft = np.abs(np.fft.rfft(s - s.mean()))
            row[f"{c}_fft1"] = float(fft[1]) if len(fft) > 1 else 0
            row[f"{c}_fft2"] = float(fft[2]) if len(fft) > 2 else 0
            row[f"{c}_fft3"] = float(fft[3]) if len(fft) > 3 else 0
            # Trend (linear regression slope on time)
            t_idx = np.arange(len(s))
            slope = np.polyfit(t_idx, s, 1)[0]
            row[f"{c}_trend"] = float(slope)
            # Autocorrelation lag 1
            s_c = s - s.mean()
            ac1 = np.sum(s_c[:-1] * s_c[1:]) / (np.sum(s_c ** 2) + 1e-6)
            row[f"{c}_ac1"] = float(ac1)
            # Peak position & magnitude
            peak_pos = np.argmax(s) / (len(s) - 1)
            row[f"{c}_peak_pos"] = float(peak_pos)
            row[f"{c}_peak_mag"] = float(s.max() - s.mean())
            # Skewness & kurtosis
            row[f"{c}_skew"] = float(skew(s)) if len(s) > 3 else 0
            row[f"{c}_range"] = float(s.max() - s.min())
        result.append(row)
    print(f"  done in {time.time()-t_start:.0f}s")
    return pd.DataFrame(result)


print("\n[1] Extracting TS features (train)...")
ts_train = extract_ts_features(train, key_cols_for_ts, "ts")
print(f"  ts_train shape: {ts_train.shape}")

print("\n[2] Extracting TS features (test)...")
ts_test = extract_ts_features(test, key_cols_for_ts, "ts")
print(f"  ts_test shape: {ts_test.shape}")

# Merge TS features back to row level
ts_feat_cols = [c for c in ts_train.columns if c not in ("layout_id", "scenario_id")]
print(f"  TS feature count: {len(ts_feat_cols)}")
train = train.merge(ts_train, on=["layout_id", "scenario_id"], how="left")
test = test.merge(ts_test, on=["layout_id", "scenario_id"], how="left")

# Add cyclical timeslot
for df in [train, test]:
    df["ts_sin"] = np.sin(2 * np.pi * df["timeslot"] / 25.0)
    df["ts_cos"] = np.cos(2 * np.pi * df["timeslot"] / 25.0)
    df["ts_third"] = (df["timeslot"] // 9).astype(np.int8)  # 3 thirds

# Per-row deviation from scenario mean/std (for key cols)
print("\n[3] Adding per-row deviation features...")
for c in key_cols_for_ts:
    # scenario mean and std
    grp_tr = train.groupby(["layout_id", "scenario_id"], sort=False)
    m_tr = grp_tr[c].transform("mean")
    s_tr = grp_tr[c].transform("std")
    train[f"{c}_zscore"] = (train[c] - m_tr) / (s_tr + 1e-6)
    grp_te = test.groupby(["layout_id", "scenario_id"], sort=False)
    m_te = grp_te[c].transform("mean")
    s_te = grp_te[c].transform("std")
    test[f"{c}_zscore"] = (test[c] - m_te) / (s_te + 1e-6)

# Layout type one-hot
for t in ["narrow", "grid", "hybrid", "hub_spoke"]:
    train[f"lt_{t}"] = (train["layout_type"] == t).astype(np.int8)
    test[f"lt_{t}"] = (test["layout_type"] == t).astype(np.int8)

# Build final feature list (independent of v23)
# Includes: raw cols + TS features + zscore + cyclical + layout static
indep_feats = raw_cols + ts_feat_cols + [f"{c}_zscore" for c in key_cols_for_ts] + \
              ["ts_sin", "ts_cos", "ts_third"] + [f"lt_{t}" for t in ["narrow","grid","hybrid","hub_spoke"]]
indep_feats = [c for c in indep_feats if c in train.columns and c in test.columns]
print(f"\n[4] Total independent features: {len(indep_feats)}")

X_tr = train[indep_feats].values.astype(np.float32)
X_te = test[indep_feats].values.astype(np.float32)
print(f"  X_tr: {X_tr.shape}, X_te: {X_te.shape}")

# NOW train
print("\n[5] Training LGB-Huber 5-fold on INDEPENDENT features...")
PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

indep_oof = np.zeros(len(y))
indep_test = np.zeros(len(X_te))
fold_rows = []

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr], y_log[tr],
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    indep_oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    indep_test += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
    fm = mean_absolute_error(y[val], indep_oof[val])
    fold_rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

indep_mae = mean_absolute_error(y, indep_oof)
print(f"\n  Independent OOF: {indep_mae:.5f}")

np.save(f"{OUT}/indep_oof.npy", indep_oof)
np.save(f"{OUT}/indep_test.npy", indep_test)

# NOW for the first time we check mega33 corr
with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline = mean_absolute_error(y, mega33_oof)

r_mega = y - mega33_oof
r_indep = y - indep_oof
corr = float(np.corrcoef(r_mega, r_indep)[0, 1])
print(f"\n[6] residual_corr(mega33, INDEPENDENT): {corr:.4f}")
print(f"  mega33 baseline: {baseline:.5f}")

# blend scan
print("\n  Blend with mega33:")
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1-w)*mega33_oof + w*indep_oof
    m = float(mean_absolute_error(y, b))
    d = m - baseline
    tag = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"    w={w:.2f}: mae={m:.5f} delta={d:+.5f}{tag}")
    if m < best_mae: best_w, best_mae = w, m

# Full multi-blend v7 (with indep)
print("\n[7] Full mega-blend v7 (adding independent)...")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("indep", indep_oof, indep_test),
    ("iter_r2", np.load("results/iter_pseudo/round2_oof.npy"),
     np.load("results/iter_pseudo/round2_test.npy")),
    ("iter_r3", np.load("results/iter_pseudo/round3_oof.npy"),
     np.load("results/iter_pseudo/round3_test.npy")),
]
names = [s[0] for s in sources]
oofs_M = np.column_stack([s[1] for s in sources])
tests_M = np.column_stack([s[2] for s in sources])
n = len(sources)

def obj_fn(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs_M @ w)

x0 = np.zeros(n); x0[0] = 0.75
for i in range(1,n): x0[i] = 0.25/(n-1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol":1e-7,"maxiter":100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt/w_opt.sum()
delta = res.fun - baseline

print(f"\nFinal mega-blend v7 weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:10s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2: -0.005434")
print(f"Improvement: {delta - (-0.005434):+.6f}")

# submission
blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v7.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v7.csv")

json.dump({
    "baseline": float(baseline),
    "indep_oof_mae": float(indep_mae),
    "indep_corr_with_mega33": corr,
    "best_single_blend_delta": float(best_mae - baseline),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "n_indep_features": len(indep_feats),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/indep_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
