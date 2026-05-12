"""
Queuing Theory + Domain Knowledge Features.

Apply warehouse operations research concepts that v23 doesn't explicitly have:
- Little's Law: L = lambda * W
- M/M/1 / M/M/c expected delay formulas
- Bottleneck analysis
- Capacity margin / saturation
- Reliability × utilization interactions

Goal: genuinely new feature family not derived from same statistical patterns.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/queuing"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Queuing Theory / Domain Knowledge Features")
print("=" * 64)
t0 = time.time()

# --- Load ---
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

# also need raw columns for some computations
train_raw = pd.read_csv("train.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
test_raw = pd.read_csv("test.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
train_raw["timeslot"] = train_raw.groupby(["layout_id", "scenario_id"]).cumcount()
test_raw["timeslot"] = test_raw.groupby(["layout_id", "scenario_id"]).cumcount()
layout = pd.read_csv("layout_info.csv")
train_raw = train_raw.merge(layout, on="layout_id", how="left")
test_raw = test_raw.merge(layout, on="layout_id", how="left")

# sanity check alignment
assert (train_raw["ID"].values == train_fe["ID"].values).all(), "order mismatch"

print("\n[1] Building queuing/domain features...")

def build_queue_features(df):
    out = pd.DataFrame(index=df.index)

    # === Arrival / service rate concepts ===
    lam = df["order_inflow_15m"].fillna(df["order_inflow_15m"].mean()).astype(np.float64)
    robots = df["robot_active"].fillna(0).astype(np.float64) + 1  # avoid div-by-zero
    util = df["robot_utilization"].fillna(0).clip(0, 0.99).astype(np.float64)
    item_per_order = df["avg_items_per_order"].fillna(1).astype(np.float64)

    # Effective load (jobs per robot)
    out["q_load_per_robot"] = lam / robots
    # Considering item complexity
    out["q_work_per_robot"] = lam * item_per_order / robots
    # Utilization-squared (near-capacity nonlinear)
    out["q_util_sq"] = util ** 2

    # === M/M/1 / M/M/c delay approximations ===
    # M/M/1 expected delay formula: W = 1/(mu - lambda) = 1/(mu(1-rho))
    # Proxy: if util is rho, then delay ∝ 1/(1-rho)
    rho = util
    out["q_mm1_delay_proxy"] = 1.0 / (1.0 - rho.clip(0, 0.99))
    # M/M/c approximation: expected wait ~ rho^c / (c*(1-rho)*c!)
    # Use robot_total as c
    rt = df["robot_total"].fillna(1).astype(np.float64).clip(1, None)
    # simplified: delay proportional to rho / (c*(1-rho))
    out["q_mmc_delay_proxy"] = rho / (rt * (1.0 - rho.clip(0, 0.99)))

    # === Bottleneck analysis ===
    # utilization across multiple resources
    util_cols = ["robot_utilization", "pack_utilization", "staging_area_util", "loading_dock_util"]
    util_cols = [c for c in util_cols if c in df.columns]
    util_mat = df[util_cols].fillna(0).values
    out["q_max_util"] = util_mat.max(axis=1)
    out["q_min_util"] = util_mat.min(axis=1)
    out["q_bottleneck_ratio"] = (util_mat.max(axis=1) + 0.01) / (util_mat.min(axis=1) + 0.01)
    out["q_util_range"] = util_mat.max(axis=1) - util_mat.min(axis=1)
    out["q_util_std"] = util_mat.std(axis=1)
    # number of resources over 80%
    out["q_stressed_resources"] = (util_mat > 0.8).sum(axis=1)

    # === Capacity margin ===
    out["q_slack_robot"] = (df["robot_idle"].fillna(0) + df["robot_charging"].fillna(0)) / (df["robot_total"].fillna(1))
    out["q_saturation"] = np.exp(5 * rho)  # exponential pressure near saturation

    # === Reliability / stability ===
    faults = df["fault_count_15m"].fillna(0).astype(np.float64)
    out["q_reliability"] = 1.0 / (1.0 + faults)
    out["q_fault_per_robot"] = faults / robots
    out["q_effective_util"] = util * (1 - faults / (df["robot_total"].fillna(1) + 1)).clip(0, 1)

    # === Order complexity vs capacity ===
    out["q_complexity_load"] = (
        df["unique_sku_15m"].fillna(0) * item_per_order / robots
    )
    out["q_urgent_pressure"] = (
        lam * df["urgent_order_ratio"].fillna(0) / (df["pack_station_count"].fillna(1) + 1)
    )

    # === Downstream / upstream bottleneck ===
    if "outbound_truck_wait_min" in df.columns:
        out["q_dock_pressure"] = (
            df["loading_dock_util"].fillna(0) * df["outbound_truck_wait_min"].fillna(0)
        )
        out["q_outbound_backlog"] = lam / (1 / (df["outbound_truck_wait_min"].fillna(1) + 1))

    # === Time-decay accumulation ===
    # cumulative faults this scenario (different from cummean)
    grp = df.groupby(["layout_id", "scenario_id"], sort=False)
    if "fault_count_15m" in df.columns:
        out["q_cum_faults"] = grp["fault_count_15m"].apply(lambda s: s.fillna(0).cumsum()).values
    if "blocked_path_15m" in df.columns:
        out["q_cum_blocked"] = grp["blocked_path_15m"].apply(lambda s: s.fillna(0).cumsum()).values

    # === Congestion × time ===
    ts = df["timeslot"].fillna(0).astype(np.float64)
    out["q_congestion_x_time"] = df["congestion_score"].fillna(0) * (ts / 24.0)
    out["q_congestion_x_util"] = df["congestion_score"].fillna(0) * rho

    # === Shift / time of day effects ===
    if "shift_hour" in df.columns:
        out["q_shift_sin"] = np.sin(2 * np.pi * df["shift_hour"].fillna(0) / 24)
        out["q_shift_cos"] = np.cos(2 * np.pi * df["shift_hour"].fillna(0) / 24)

    # === Charge / energy dynamics ===
    if "battery_mean" in df.columns and "charge_queue_length" in df.columns:
        out["q_charge_stress"] = df["charge_queue_length"].fillna(0) / (df["battery_mean"].fillna(50) + 1)
        out["q_charge_bottleneck"] = df["charge_queue_length"].fillna(0) * (1 - df["battery_mean"].fillna(50) / 100)

    return out

queue_train = build_queue_features(train_raw)
queue_test = build_queue_features(test_raw)
queue_cols = list(queue_train.columns)
print(f"  built {len(queue_cols)} queuing features")

# --- Combine with v23 features ---
X_tr_base = train_fe[feat_cols].values.astype(np.float32)
X_te_base = test_fe[feat_cols].values.astype(np.float32)
X_tr = np.hstack([X_tr_base, queue_train.values.astype(np.float32)])
X_te = np.hstack([X_te_base, queue_test.values.astype(np.float32)])
all_feat_names = feat_cols + queue_cols
del X_tr_base, X_te_base
gc.collect()
print(f"  X_tr: {X_tr.shape}, X_te: {X_te.shape}")

# --- Train LGB-Huber 5-fold ---
print("\n[2] Training LGB-Huber (v23 + queuing) 5-fold...")
PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

q_oof = np.zeros(len(y))
q_test = np.zeros(len(X_te))
fold_rows = []
imp_accum = np.zeros(X_tr.shape[1])

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr], y_log[tr],
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    q_oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    q_test += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
    imp_accum += m.feature_importances_
    fm = mean_absolute_error(y[val], q_oof[val])
    fold_rows.append({"fold": f, "mae": float(fm), "best_iter": int(m.best_iteration_ or 5000)})
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tv:.0f}s)")

q_mae = mean_absolute_error(y, q_oof)
print(f"\nQueuing-LGB OOF: {q_mae:.5f}")
np.save(f"{OUT}/queue_oof.npy", q_oof)
np.save(f"{OUT}/queue_test.npy", q_test)
pd.DataFrame(fold_rows).to_csv(f"{OUT}/queue_fold_metrics.csv", index=False)

# --- Feature importance: check queuing features ---
print("\n[3] Top 20 features by importance (looking for queuing features)")
imp_series = pd.Series(imp_accum / 5, index=all_feat_names).sort_values(ascending=False)
for i, (name, imp) in enumerate(imp_series.head(20).items()):
    tag = " ★" if name.startswith("q_") else ""
    print(f"  {i+1:2d}. {name:40s} imp={imp:.0f}{tag}")

# --- Diversity check ---
r_mega = y - mega33_oof
r_q = y - q_oof
corr = float(np.corrcoef(r_mega, r_q)[0, 1])
print(f"\n[4] residual_corr(mega33, queuing): {corr:.4f}")

best_w, best_mae = 0, float(baseline_mae)
scan = []
for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    blend = (1 - w) * mega33_oof + w * q_oof
    mae = float(mean_absolute_error(y, blend))
    scan.append({"w": w, "mae": mae, "delta": mae - baseline_mae})
    tag = " ***" if mae - baseline_mae < -0.01 else (" **" if mae - baseline_mae < -0.005 else "")
    print(f"  blend w={w:.2f}: mae={mae:.5f} delta={mae - baseline_mae:+.5f}{tag}")
    if mae < best_mae:
        best_w, best_mae = w, mae

verdict = "STRONG_GO" if best_mae - baseline_mae < -0.02 else (
    "GO" if best_mae - baseline_mae < -0.008 else (
    "WEAK" if best_mae - baseline_mae < -0.003 else "NO_GO"))

if best_w > 0 and best_mae < baseline_mae:
    blend_test = (1 - best_w) * mega33_test + best_w * q_test
    test_ids = pd.read_csv("test.csv")["ID"].values
    sorted_ids = test_raw["ID"].values
    pred_map = dict(zip(sorted_ids, blend_test))
    sub_preds = np.array([pred_map[id_] for id_ in test_ids])
    pd.DataFrame({"ID": test_ids, TARGET: np.clip(sub_preds, 0, None)}).to_csv(
        f"{OUT}/submission_mega33_queue_w{int(best_w*100)}.csv", index=False)
    print(f"\nSubmission: {OUT}/submission_mega33_queue_w{int(best_w*100)}.csv")

json.dump({
    "queue_oof_mae": float(q_mae),
    "residual_corr": corr,
    "best_blend_w": float(best_w),
    "best_blend_mae": float(best_mae),
    "best_delta": float(best_mae - baseline_mae),
    "verdict": verdict,
    "n_queue_features": len(queue_cols),
    "queue_features": queue_cols,
    "elapsed": round(time.time() - t0, 1),
}, open(f"{OUT}/queue_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"VERDICT: {verdict}  corr={corr:.4f}  delta={best_mae - baseline_mae:+.5f}")
print(f"elapsed: {time.time() - t0:.0f}s")
print(f"{'='*64}")
