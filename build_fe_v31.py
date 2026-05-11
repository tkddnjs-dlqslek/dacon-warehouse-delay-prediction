"""
Feature Engineering v31 — v30 확장판.
v30에서 미개발된 컬럼들의 SC aggregates + lag/lead/diff 추가.

v30 대비 추가:
1. extra_cols (5개)에 SC aggregates (mean,std,max,min,range,rank,dev) 추가 → +35개
2. avg_charge_wait, urgent_order_ratio 완전 개발 (key_cols 수준) → +34개
3. 새 cross-interactions (collision×blocked, cold_chain×temp, system 신뢰도) → +12개
4. NaN row count feature → +1개

결과: ~149 + 82 = 약 230개 → feature importance 기반 pruning 가능
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pickle, os
import numpy as np, pandas as pd

ROOT = r"C:\Users\user\Desktop\데이콘 4월"
OUT  = os.path.join(ROOT, "results", "eda_v31")
os.makedirs(OUT, exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"
SORT_KEY = ["layout_id", "scenario_id"]

print("Loading data...", flush=True)
train = pd.read_csv(os.path.join(ROOT, "train.csv"))
test  = pd.read_csv(os.path.join(ROOT, "test.csv"))
layout_df = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))

train = train.sort_values(SORT_KEY).reset_index(drop=True)
test  = test.sort_values(SORT_KEY).reset_index(drop=True)

# Load v30 feat_cols as reference
with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
    v30 = pickle.load(f)
v30_feat_cols = list(v30['feat_cols'])
print(f"v30 feat_cols: {len(v30_feat_cols)} features", flush=True)


def engineer_features_v31(df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(layout_df, on="layout_id", how="left")
    df["timeslot"]      = df.groupby(["layout_id", "scenario_id"]).cumcount()
    df["timeslot_sq"]   = df["timeslot"] ** 2
    df["timeslot_norm"] = df["timeslot"] / 24.0
    df = df.sort_values(["layout_id", "scenario_id", "timeslot"]).reset_index(drop=True)
    group = df.groupby(["layout_id", "scenario_id"])

    # ----------------------------------------------------------------
    # v30 key_cols: 완전 개발 (lag, lead, diff, rolling, cummean, SC)
    # ----------------------------------------------------------------
    key_cols = [
        "order_inflow_15m", "congestion_score", "robot_utilization",
        "battery_mean", "fault_count_15m", "blocked_path_15m",
        "pack_utilization", "charge_queue_length",
    ]
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lag1"]  = g.shift(1)
        df[f"{col}_lag2"]  = g.shift(2)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rmean3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_rstd3"]  = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f"{col}_rmean5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{col}_cummean"] = g.transform(lambda x: x.expanding().mean())
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_lead2"] = g.shift(-2)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]
        df[f"{col}_sc_mean"]  = g.transform("mean")
        df[f"{col}_sc_std"]   = g.transform("std")
        df[f"{col}_sc_max"]   = g.transform("max")
        df[f"{col}_sc_min"]   = g.transform("min")
        df[f"{col}_sc_range"] = df[f"{col}_sc_max"] - df[f"{col}_sc_min"]
        df[f"{col}_sc_rank"]  = g.rank(pct=True)
        df[f"{col}_sc_dev"]   = df[col] - df[f"{col}_sc_mean"]

    # ----------------------------------------------------------------
    # v30 extra_cols: lag/lead/diff만 있었음 → SC aggregates 추가 (NEW)
    # ----------------------------------------------------------------
    extra_cols = [
        "max_zone_density", "robot_charging", "low_battery_ratio",
        "robot_idle", "near_collision_15m",
    ]
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lag1"]  = g.shift(1)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]
        # NEW: SC aggregates
        df[f"{col}_sc_mean"]  = g.transform("mean")
        df[f"{col}_sc_std"]   = g.transform("std")
        df[f"{col}_sc_max"]   = g.transform("max")
        df[f"{col}_sc_min"]   = g.transform("min")
        df[f"{col}_sc_range"] = df[f"{col}_sc_max"] - df[f"{col}_sc_min"]
        df[f"{col}_sc_rank"]  = g.rank(pct=True)
        df[f"{col}_sc_dev"]   = df[col] - df[f"{col}_sc_mean"]

    # ----------------------------------------------------------------
    # NEW key_cols: avg_charge_wait, urgent_order_ratio 완전 개발
    # ----------------------------------------------------------------
    new_key_cols = ["avg_charge_wait", "urgent_order_ratio"]
    for col in new_key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lag1"]  = g.shift(1)
        df[f"{col}_lag2"]  = g.shift(2)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rmean3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_rstd3"]  = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f"{col}_cummean"] = g.transform(lambda x: x.expanding().mean())
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]
        df[f"{col}_sc_mean"]  = g.transform("mean")
        df[f"{col}_sc_std"]   = g.transform("std")
        df[f"{col}_sc_max"]   = g.transform("max")
        df[f"{col}_sc_min"]   = g.transform("min")
        df[f"{col}_sc_range"] = df[f"{col}_sc_max"] - df[f"{col}_sc_min"]
        df[f"{col}_sc_rank"]  = g.rank(pct=True)
        df[f"{col}_sc_dev"]   = df[col] - df[f"{col}_sc_mean"]

    # ----------------------------------------------------------------
    # v30 cross-interactions (복원)
    # ----------------------------------------------------------------
    df["order_per_robot"] = df["order_inflow_15m"] / (df["robot_active"] + 1)
    rta = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["robot_available_ratio"] = df["robot_idle"] / (rta + 1)
    df["robot_charging_ratio"]  = df["robot_charging"] / (rta + 1)
    df["battery_risk"]          = df["low_battery_ratio"] * df["charge_queue_length"]
    df["congestion_x_utilization"] = df["congestion_score"] * df["robot_utilization"]
    df["congestion_x_order"]    = df["congestion_score"] * df["order_inflow_15m"]
    df["order_complexity"]      = df["unique_sku_15m"] * df["avg_items_per_order"]
    df["urgent_order_volume"]   = df["order_inflow_15m"] * df["urgent_order_ratio"]
    df["dock_pressure"]         = df["loading_dock_util"] * df["outbound_truck_wait_min"]
    df["staff_per_order"]       = df["staff_on_floor"] / (df["order_inflow_15m"] + 1)
    df["total_utilization"]     = (
        df["pack_utilization"] + df["staging_area_util"] + df["loading_dock_util"]
    ) / 3
    df["fault_x_congestion"]    = df["fault_count_15m"] * df["congestion_score"]
    df["battery_charge_pressure"] = df["low_battery_ratio"] * df["avg_charge_wait"]
    df["congestion_per_robot"]  = df["congestion_score"] / (df["robot_active"] + 1)
    df["order_per_staff"]       = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)
    df["order_per_area"]        = df["order_inflow_15m"] / (df["floor_area_sqm"] + 1) * 1000
    df["congestion_per_area"]   = df["congestion_score"] / (df["floor_area_sqm"] + 1) * 1000
    df["fault_per_robot_total"] = df["fault_count_15m"] / (df["robot_total"] + 1)
    df["blocked_per_robot_total"] = df["blocked_path_15m"] / (df["robot_total"] + 1)
    df["collision_per_robot_total"] = df["near_collision_15m"] / (df["robot_total"] + 1)
    df["pack_util_per_station"] = df["pack_utilization"] / (df["pack_station_count"] + 1)
    df["charge_queue_per_charger"] = df["charge_queue_length"] / (df["charger_count"] + 1)
    df["order_per_pack_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1)
    df["floor_area_per_robot"]  = df["floor_area_sqm"] / (df["robot_total"] + 1)
    df["charger_ratio"]         = df["charger_count"] / (df["robot_total"] + 1)
    df["robot_density"]         = df["robot_total"] / (df["floor_area_sqm"] + 1) * 1000
    df["active_vs_total"]       = df["robot_active"] / (df["robot_total"] + 1)
    df["congestion_x_aisle_width"] = df["congestion_score"] * df["aisle_width_avg"]
    df["congestion_x_compactness"] = df["congestion_score"] * df["layout_compactness"]
    df["blocked_x_one_way"]     = df["blocked_path_15m"] * df["one_way_ratio"]
    df["utilization_x_compactness"] = df["robot_utilization"] * df["layout_compactness"]

    # ----------------------------------------------------------------
    # NEW cross-interactions (v31 신규)
    # ----------------------------------------------------------------
    df["blocked_x_collision"]   = df["blocked_path_15m"] * df["near_collision_15m"]
    df["charge_wait_x_queue"]   = df["avg_charge_wait"] * df["charge_queue_length"]
    df["urgent_x_congestion"]   = df["urgent_order_ratio"] * df["congestion_score"]
    df["fault_x_blocked"]       = df["fault_count_15m"] * df["blocked_path_15m"]
    df["battery_std_x_util"]    = df["battery_std"] * df["robot_utilization"] if "battery_std" in df.columns else 0.0
    df["low_batt_x_charge_q"]   = df["low_battery_ratio"] * df["charge_queue_per_charger"]
    df["collision_x_congestion"]= df["near_collision_15m"] * df["congestion_score"]
    df["zone_density_x_cong"]   = df["max_zone_density"] * df["congestion_score"]
    # NaN 카운트 feature
    raw_cols = [c for c in df.columns if c not in ["ID","layout_id","scenario_id","timeslot","timeslot_sq","timeslot_norm",TARGET]]
    df["null_count"] = df[raw_cols].isnull().sum(axis=1)

    # ----------------------------------------------------------------
    # Layout static columns 제거
    # ----------------------------------------------------------------
    layout_static = [
        "layout_type", "aisle_width_avg", "intersection_count", "one_way_ratio",
        "pack_station_count", "charger_count", "layout_compactness", "zone_dispersion",
        "robot_total", "building_age_years", "floor_area_sqm", "ceiling_height_m",
        "fire_sprinkler_count", "emergency_exit_count",
    ]
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors="ignore")

    # ----------------------------------------------------------------
    # v30 corr_remove (동일하게 제거)
    # ----------------------------------------------------------------
    corr_remove = [
        "battery_mean_rmean3", "charge_queue_length_rmean3",
        "battery_mean_rmean5", "charge_queue_length_rmean5",
        "pack_utilization_rmean5", "battery_mean_lag1",
        "charge_queue_length_lag1", "congestion_score_rmean3",
        "order_inflow_15m_cummean", "robot_utilization_rmean5",
        "robot_utilization_rmean3", "order_inflow_15m_rmean5",
        "battery_risk", "congestion_score_rmean5",
        "pack_utilization_rmean3", "order_inflow_15m_rmean3",
        "charge_queue_length_lag2", "blocked_path_15m_rmean5",
    ]
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors="ignore")
    return df


print("Building v31 FE for train...", flush=True)
train_fe = engineer_features_v31(train.copy(), layout_df)
print("Building v31 FE for test...", flush=True)
test_fe  = engineer_features_v31(test.copy(),  layout_df)

# feat_cols: non-ID, non-target, numeric columns present in both
exclude = {"ID", "layout_id", "scenario_id", "timeslot", TARGET}
candidate = [c for c in train_fe.columns if c not in exclude
             and train_fe[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]
feat_cols = [c for c in candidate if c in test_fe.columns]

# Fill NaN
train_fe[feat_cols] = train_fe[feat_cols].fillna(0)
test_fe[feat_cols]  = test_fe[feat_cols].fillna(0)

print(f"v31 feat_cols: {len(feat_cols)} (v30 had {len(v30_feat_cols)}, +{len(feat_cols)-len(v30_feat_cols)})", flush=True)
new_feats = [c for c in feat_cols if c not in v30_feat_cols]
print(f"New features ({len(new_feats)}): {new_feats[:20]}...", flush=True)

# Save fold_idx from v30 (reuse same folds)
fold_idx = None
fold_path = os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy")
if os.path.exists(fold_path):
    fold_idx = np.load(fold_path)
    print("Reusing v30 fold_idx", flush=True)

blob = {
    "train_fe":  train_fe,
    "test_fe":   test_fe,
    "feat_cols": feat_cols,
    "fold_idx":  fold_idx,
    "new_feats": new_feats,
}
out_path = os.path.join(OUT, "v31_fe_cache.pkl")
with open(out_path, "wb") as f:
    pickle.dump(blob, f, protocol=4)
print(f"Saved: {out_path}", flush=True)
print("Done.", flush=True)
