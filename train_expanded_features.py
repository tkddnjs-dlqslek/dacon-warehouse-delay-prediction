"""
Expanded features: TS/SC/interaction on NEGLECTED columns.

v23 only heavily engineered 13 cols. There are ~77 untouched cols across:
- Environmental (9)
- System reliability (7)
- Order complexity (5)
- Facility bottlenecks (9)
- Personnel / time (4)

Strategy:
1. Define 5 groups of neglected cols
2. Build per-group: SC_mean, SC_std, cumsum, deviation, group_aggregate_scores
3. Cross-group interactions (environment × system, order × facility, etc.)
4. Train LGB + compare with mega33
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/expanded_features"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Expanded Features: TS/SC on NEGLECTED columns")
print("=" * 64)
t0 = time.time()

# Load raw
train = pd.read_csv("train.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
test = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
layout = pd.read_csv("layout_info.csv")
train["timeslot"] = train.groupby(["layout_id","scenario_id"]).cumcount()
test["timeslot"] = test.groupby(["layout_id","scenario_id"]).cumcount()
train = train.merge(layout, on="layout_id", how="left")
test = test.merge(layout, on="layout_id", how="left")

fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train[TARGET].values.astype(np.float64)
y_log = np.log1p(y)

with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline:.5f}")

# --- Define neglected column groups ---
ENV_COLS = ["warehouse_temp_c", "humidity_pct", "external_temp_c", "wind_speed_kmh",
            "precipitation_mm", "lighting_level_lux", "noise_level_db",
            "vibration_level", "air_quality_index"]
SYS_COLS = ["wms_response_time_ms", "scanner_error_rate", "wifi_signal_db",
            "network_latency_ms", "label_print_queue", "barcode_success_rate", "ups_battery_pct"]
ORDER_COLS = ["unique_sku_15m", "avg_items_per_order", "urgent_order_ratio",
              "heavy_item_ratio", "cold_chain_ratio", "bulk_order_ratio"]
FACIL_COLS = ["staging_area_util", "cold_storage_util", "pallet_wrap_util",
              "conveyor_speed", "manual_override_ratio", "loading_dock_util",
              "pack_utilization", "inventory_turnover", "outbound_truck_wait_min"]
PEOPLE_COLS = ["staff_on_floor", "safety_score", "tenure_avg_months", "shift_hour"]

# Filter to columns that actually exist
def filter_exist(cols, df):
    return [c for c in cols if c in df.columns]
ENV_COLS = filter_exist(ENV_COLS, train)
SYS_COLS = filter_exist(SYS_COLS, train)
ORDER_COLS = filter_exist(ORDER_COLS, train)
FACIL_COLS = filter_exist(FACIL_COLS, train)
PEOPLE_COLS = filter_exist(PEOPLE_COLS, train)
print(f"ENV: {len(ENV_COLS)}, SYS: {len(SYS_COLS)}, ORDER: {len(ORDER_COLS)}, "
      f"FACIL: {len(FACIL_COLS)}, PEOPLE: {len(PEOPLE_COLS)}")

# --- Build expanded features ---
def build_expanded(df, out_cols):
    """Add expanded features to df. Modifies in place, returns list of new cols added."""
    new_cols = []
    grp = df.groupby(["layout_id","scenario_id"], sort=False)

    # Per-col TS features on each group
    for group_name, cols in [("env", ENV_COLS), ("sys", SYS_COLS),
                              ("order", ORDER_COLS), ("facil", FACIL_COLS)]:
        for c in cols:
            if c not in df.columns: continue
            g = grp[c]
            # SC features
            sc_mean = g.transform("mean")
            sc_std = g.transform("std").fillna(0)
            df[f"{c}_sc_mean"] = sc_mean
            df[f"{c}_sc_std"] = sc_std
            df[f"{c}_dev"] = df[c] - sc_mean
            df[f"{c}_zscore"] = (df[c] - sc_mean) / (sc_std + 1e-6)
            new_cols.extend([f"{c}_sc_mean", f"{c}_sc_std", f"{c}_dev", f"{c}_zscore"])

    # Group aggregate scores (product/mean of standardized cols in each group)
    for group_name, cols in [("env", ENV_COLS), ("sys", SYS_COLS),
                              ("order", ORDER_COLS), ("facil", FACIL_COLS)]:
        valid_cols = [c for c in cols if c in df.columns]
        if len(valid_cols) >= 2:
            # standardize using group's own stats (per-row)
            vals = df[valid_cols].values.astype(np.float64)
            vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
            means = vals.mean(axis=0)
            stds = vals.std(axis=0) + 1e-6
            vals_std = (vals - means) / stds
            df[f"{group_name}_agg_mean"] = vals_std.mean(axis=1)
            df[f"{group_name}_agg_max"] = vals_std.max(axis=1)
            df[f"{group_name}_agg_std"] = vals_std.std(axis=1)
            new_cols.extend([f"{group_name}_agg_mean", f"{group_name}_agg_max", f"{group_name}_agg_std"])

    # Cross-group interactions (a × b for aggregate scores)
    for a in ["env_agg_mean", "sys_agg_mean", "order_agg_mean", "facil_agg_mean"]:
        for b in ["env_agg_mean", "sys_agg_mean", "order_agg_mean", "facil_agg_mean"]:
            if a < b and a in df.columns and b in df.columns:
                nm = f"{a}_x_{b}"
                df[nm] = df[a] * df[b]
                new_cols.append(nm)

    # Personnel: shift_hour cyclical + flags
    if "shift_hour" in df.columns:
        df["shift_sin"] = np.sin(2*np.pi*df["shift_hour"]/24)
        df["shift_cos"] = np.cos(2*np.pi*df["shift_hour"]/24)
        df["is_night"] = ((df["shift_hour"] < 6) | (df["shift_hour"] > 22)).astype(np.int8)
        df["is_peak"] = ((df["shift_hour"] >= 9) & (df["shift_hour"] <= 11)).astype(np.int8)
        new_cols.extend(["shift_sin","shift_cos","is_night","is_peak"])

    # Personnel × load interactions
    if "staff_on_floor" in df.columns and "order_inflow_15m" in df.columns:
        df["orders_per_staff"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)
        new_cols.append("orders_per_staff")

    # Order complexity × utilization
    if "avg_items_per_order" in df.columns and "robot_utilization" in df.columns:
        df["complexity_x_util"] = df["avg_items_per_order"] * df["robot_utilization"]
        new_cols.append("complexity_x_util")

    # Environmental × system reliability
    if "noise_level_db" in df.columns and "scanner_error_rate" in df.columns:
        df["noise_x_scanner"] = df["noise_level_db"] * df["scanner_error_rate"]
        new_cols.append("noise_x_scanner")
    if "vibration_level" in df.columns and "scanner_error_rate" in df.columns:
        df["vib_x_scanner"] = df["vibration_level"] * df["scanner_error_rate"]
        new_cols.append("vib_x_scanner")

    # Cumulative features on order cols
    for c in ORDER_COLS:
        if c not in df.columns: continue
        vals = df[c].values.astype(np.float64)
        vals = np.where(np.isnan(vals), np.nanmean(vals), vals)
        cum = grp[c].apply(lambda s: s.fillna(s.mean()).cumsum()).values
        df[f"{c}_cumsum"] = cum
        new_cols.append(f"{c}_cumsum")

    # Facility bottleneck interactions (max of utilizations)
    util_cols = [c for c in ["robot_utilization","pack_utilization","staging_area_util",
                              "cold_storage_util","loading_dock_util"] if c in df.columns]
    if len(util_cols) >= 2:
        df["facil_util_max"] = df[util_cols].max(axis=1)
        df["facil_util_range"] = df[util_cols].max(axis=1) - df[util_cols].min(axis=1)
        new_cols.extend(["facil_util_max","facil_util_range"])

    return new_cols

print("\n[1] Building expanded features for train...")
new_cols_tr = build_expanded(train, None)
print(f"  added {len(new_cols_tr)} new columns to train")

print("\n[2] Building expanded features for test...")
new_cols_te = build_expanded(test, None)
print(f"  added {len(new_cols_te)} new columns to test")

# intersection
new_cols = [c for c in new_cols_tr if c in test.columns]
print(f"  aligned new cols: {len(new_cols)}")

# --- Load v23 cached features ---
print("\n[3] Loading v23 cache + merging with expanded...")
with open("results/eda_v30/v30_fe_cache.pkl","rb") as f:
    blob_tr = pickle.load(f)
train_v23 = blob_tr["train_fe"]
v23_feats = list(blob_tr["feat_cols"])

with open("results/eda_v30/v30_test_fe_cache.pkl","rb") as f:
    test_v23 = pickle.load(f)

# sort v23 to align
assert (train_v23["ID"].values == train["ID"].values).all(), "order mismatch"

# Build combined feature matrix
combined_feats = v23_feats + new_cols
X_tr_combined = np.column_stack([
    train_v23[v23_feats].values,
    train[new_cols].fillna(0).values
]).astype(np.float32)
X_te_combined = np.column_stack([
    test_v23[v23_feats].values,
    test[new_cols].fillna(0).values
]).astype(np.float32)
print(f"  X_tr: {X_tr_combined.shape}, X_te: {X_te_combined.shape}")

# --- Train LGB-Huber 5-fold ---
print("\n[4] Training LGB-Huber 5-fold (v23 + expanded)...")
PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

exp_oof = np.zeros(len(y))
exp_test = np.zeros(len(X_te_combined))
fold_rows = []
imp_accum = np.zeros(X_tr_combined.shape[1])

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr_combined[tr], y_log[tr],
          eval_set=[(X_tr_combined[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    exp_oof[val] = np.clip(np.expm1(m.predict(X_tr_combined[val])), 0, None)
    exp_test += np.clip(np.expm1(m.predict(X_te_combined)), 0, None) / 5
    imp_accum += m.feature_importances_
    fm = mean_absolute_error(y[val], exp_oof[val])
    fold_rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

exp_mae = mean_absolute_error(y, exp_oof)
exp_corr = float(np.corrcoef(y - mega33_oof, y - exp_oof)[0, 1])
print(f"\nExpanded OOF: {exp_mae:.5f}")
print(f"residual_corr(mega33): {exp_corr:.4f}")

np.save(f"{OUT}/exp_oof.npy", exp_oof)
np.save(f"{OUT}/exp_test.npy", exp_test)

# --- Top 20 features (check if expanded features get used) ---
print("\n[5] Top 20 features by importance:")
imp = pd.Series(imp_accum / 5, index=combined_feats).sort_values(ascending=False)
for i, (nm, v) in enumerate(imp.head(20).items()):
    is_new = "★ NEW" if nm in new_cols else ""
    print(f"  {i+1:2d}. {nm:40s} imp={v:.0f}  {is_new}")

# How many new features in top 50?
top50 = imp.head(50).index.tolist()
new_in_top50 = [c for c in top50 if c in new_cols]
print(f"\n  new features in top 50: {len(new_in_top50)}")

# --- Blend with mega33 ---
print("\n[6] Blend with mega33:")
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25]:
    b = (1-w)*mega33_oof + w*exp_oof
    m = float(mean_absolute_error(y, b))
    d = m - baseline
    tag = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"  w={w:.2f}: mae={m:.5f} delta={d:+.5f}{tag}")
    if m < best_mae: best_w, best_mae = w, m

# --- Full multi-blend v9 ---
print("\n[7] Full mega-blend v9 (adding expanded)...")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("expanded", exp_oof, exp_test),
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
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol":1e-7, "maxiter":100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline

print("\nFinal mega-blend v9 weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:12s}: {w_:.4f}")
print(f"OOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2: -0.005434")
print(f"Improvement: {delta - (-0.005434):+.6f}")

# submission
blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v9.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v9.csv")

json.dump({
    "baseline": float(baseline),
    "exp_oof_mae": float(exp_mae),
    "exp_corr": exp_corr,
    "n_new_features": len(new_cols),
    "new_features_in_top50": len(new_in_top50),
    "best_single_blend_delta": float(best_mae - baseline),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/expanded_summary.json","w"), indent=2)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
