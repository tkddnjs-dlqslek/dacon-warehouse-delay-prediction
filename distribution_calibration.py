"""
Distribution analysis + calibration WITHOUT LB probing.

Compare:
- y_train distribution
- our prediction distribution on test
- If systematic shift → apply correction

Also: apply classical calibrations (shift, scale, monotone) and check OOF impact.
"""
import pickle, numpy as np, pandas as pd, os, json
from sklearn.metrics import mean_absolute_error
import warnings; warnings.filterwarnings("ignore")

OUT = "results/dist_calibration"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

# Load
train = pd.read_csv("train.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
y = train[TARGET].values.astype(np.float64)
fold_ids = np.load("results/eda_v30/fold_idx.npy")

with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline_oof = mean_absolute_error(y, mega33_oof)

# Load our FIXED multi-blend (current best submission)
fixed_df = pd.read_csv("results/final_blend/submission_final_multiblend_FIXED.csv")
# Load the sorted ids to align fixed_test with our train order
sorted_test_ids = pd.read_csv("results/eda_v30/v30_test_fe_cache.pkl"
                              .replace(".pkl", ""), engine='c') if False else None
# Use the sorted test order from v30 test cache
with open("results/eda_v30/v30_test_fe_cache.pkl","rb") as f:
    test_fe = pickle.load(f)
sorted_test_ids = test_fe["ID"].values

fixed_test_sorted = fixed_df.set_index("ID").loc[sorted_test_ids][TARGET].values

print("=" * 64)
print("Distribution Calibration Analysis")
print("=" * 64)

# === Step 1: Distribution comparison ===
print("\n[1] Distribution comparison (y_train vs predictions)")
def describe_dist(vals, name):
    v = np.asarray(vals).astype(np.float64)
    print(f"  {name}:")
    print(f"    mean:    {v.mean():.4f}")
    print(f"    median:  {np.median(v):.4f}")
    print(f"    std:     {v.std():.4f}")
    print(f"    q10:     {np.quantile(v, 0.1):.4f}")
    print(f"    q25:     {np.quantile(v, 0.25):.4f}")
    print(f"    q75:     {np.quantile(v, 0.75):.4f}")
    print(f"    q90:     {np.quantile(v, 0.9):.4f}")
    print(f"    q99:     {np.quantile(v, 0.99):.4f}")
    print(f"    max:     {v.max():.4f}")

describe_dist(y, "y_train")
describe_dist(mega33_oof, "mega33_oof (train)")
describe_dist(mega33_test, "mega33_test")
describe_dist(fixed_test_sorted, "fixed_test (our current best)")

# === Step 2: Check train vs test prediction distribution ===
print("\n[2] Train prediction vs Test prediction (same model, different data)")
print(f"  mega33_oof mean:  {mega33_oof.mean():.4f}")
print(f"  mega33_test mean: {mega33_test.mean():.4f}")
print(f"  diff: {mega33_test.mean() - mega33_oof.mean():+.4f}")
print(f"  (if positive, test predictions higher than train OOF predictions)")

# If test predictions have SAME distribution as train OOF, but different from y_train,
# then mean(y_test) likely similar to mean(mega33_test) too

# === Step 3: Classical shift/scale calibration on OOF ===
print("\n[3] Classical calibrations on OOF (find optimal shift/scale)")

# global shift: pred + c
print("\n  Global shift:")
baseline = baseline_oof
best_shift, best_mae = 0, float(baseline)
for c in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
    shifted = mega33_oof + c
    mae = mean_absolute_error(y, np.clip(shifted, 0, None))
    d = mae - baseline
    print(f"    shift={c:+.1f}: mae={mae:.5f} delta={d:+.5f}")
    if mae < best_mae: best_shift, best_mae = c, mae
print(f"  best_shift: {best_shift:+.2f} delta: {best_mae - baseline:+.5f}")

# global scale: pred * s
print("\n  Global scale:")
best_scale, best_scale_mae = 1, float(baseline)
for s in [0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3]:
    scaled = mega33_oof * s
    mae = mean_absolute_error(y, np.clip(scaled, 0, None))
    d = mae - baseline
    print(f"    scale={s:.2f}: mae={mae:.5f} delta={d:+.5f}")
    if mae < best_scale_mae: best_scale, best_scale_mae = s, mae
print(f"  best_scale: {best_scale:.3f} delta: {best_scale_mae - baseline:+.5f}")

# combined shift+scale optimization
from scipy.optimize import minimize
def obj_ss(params):
    s, c = params
    p = np.clip(s * mega33_oof + c, 0, None)
    return mean_absolute_error(y, p)
res = minimize(obj_ss, [1.0, 0.0], method='Nelder-Mead', options={'xatol': 1e-5})
opt_s, opt_c = res.x
print(f"\n  Combined shift+scale: scale={opt_s:.4f}, shift={opt_c:+.4f}")
print(f"    mae={res.fun:.5f}  delta={res.fun - baseline:+.5f}")

# === Step 4: Per-bucket calibration ===
print("\n[4] Per-bucket shift calibration")
layout = pd.read_csv("layout_info.csv")[["layout_id","layout_type"]]
train_meta = train[["layout_id"]].merge(layout, on="layout_id", how="left")
layout_type_train = train_meta["layout_type"].values
timeslot_train = train.groupby(["layout_id","scenario_id"]).cumcount().values

types = ["narrow","grid","hybrid","hub_spoke"]
print(f"  per-layout_type optimal shifts:")
cal_oof = mega33_oof.copy()
for lt in types:
    mask = layout_type_train == lt
    def obj_shift(c):
        return mean_absolute_error(y[mask], np.clip(mega33_oof[mask] + c[0], 0, None))
    r = minimize(obj_shift, [0.0], method='Nelder-Mead')
    opt_c = r.x[0]
    cal_oof[mask] = np.clip(mega33_oof[mask] + opt_c, 0, None)
    print(f"    {lt:10s}: opt shift = {opt_c:+.4f}  single-mae {r.fun:.5f}")
cal_mae = mean_absolute_error(y, cal_oof)
print(f"  total per-layout_type calibration: mae {cal_mae:.5f}  delta {cal_mae - baseline:+.5f}")

# per layout_type x ts_bucket
print("\n  per (layout_type × ts_bucket) optimal shifts:")
ts_bucket = (timeslot_train // 5).astype(np.int8)
cal_oof2 = mega33_oof.copy()
for lt in types:
    for tb in range(5):
        mask = (layout_type_train == lt) & (ts_bucket == tb)
        if mask.sum() < 100: continue
        def obj_s(c):
            return mean_absolute_error(y[mask], np.clip(mega33_oof[mask] + c[0], 0, None))
        r = minimize(obj_s, [0.0], method='Nelder-Mead')
        opt_c = r.x[0]
        cal_oof2[mask] = np.clip(mega33_oof[mask] + opt_c, 0, None)
cal_mae2 = mean_absolute_error(y, cal_oof2)
print(f"  total per-(layout_type × ts_bucket) calibration: mae {cal_mae2:.5f}  delta {cal_mae2 - baseline:+.5f}")

# === Step 5: Apply to test ===
print("\n[5] Apply best calibration to test + generate submissions")

# test bucket info
test_raw = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
test_raw["timeslot"] = test_raw.groupby(["layout_id","scenario_id"]).cumcount()
test_raw = test_raw.merge(layout, on="layout_id", how="left")
layout_type_test = test_raw["layout_type"].values
ts_bucket_test = (test_raw["timeslot"].values // 5).astype(np.int8)

# Apply global shift+scale (best combined)
mega33_cal_test = np.clip(opt_s * mega33_test + opt_c, 0, None)
fixed_cal_test = np.clip(opt_s * fixed_test_sorted + opt_c, 0, None)

# Apply per-bucket shifts learned on OOF
shifts_lt = {}
for lt in types:
    mask_train = layout_type_train == lt
    def obj_s(c):
        return mean_absolute_error(y[mask_train], np.clip(mega33_oof[mask_train] + c[0], 0, None))
    shifts_lt[lt] = minimize(obj_s, [0.0], method='Nelder-Mead').x[0]

shifts_lt_ts = {}
for lt in types:
    for tb in range(5):
        mask_train = (layout_type_train == lt) & (ts_bucket == tb)
        if mask_train.sum() < 100:
            shifts_lt_ts[(lt,tb)] = 0
            continue
        def obj_s(c):
            return mean_absolute_error(y[mask_train], np.clip(mega33_oof[mask_train] + c[0], 0, None))
        shifts_lt_ts[(lt,tb)] = minimize(obj_s, [0.0], method='Nelder-Mead').x[0]

# Apply per-bucket to fixed test
fixed_lt_cal = fixed_test_sorted.copy()
for lt in types:
    mask = layout_type_test == lt
    fixed_lt_cal[mask] = fixed_test_sorted[mask] + shifts_lt[lt]
fixed_lt_cal = np.clip(fixed_lt_cal, 0, None)

fixed_lt_ts_cal = fixed_test_sorted.copy()
for lt in types:
    for tb in range(5):
        mask = (layout_type_test == lt) & (ts_bucket_test == tb)
        if mask.sum() > 0:
            fixed_lt_ts_cal[mask] = fixed_test_sorted[mask] + shifts_lt_ts[(lt,tb)]
fixed_lt_ts_cal = np.clip(fixed_lt_ts_cal, 0, None)

# save
sorted_ids = sorted_test_ids
pd.DataFrame({"ID": sorted_ids, TARGET: fixed_cal_test}).to_csv(
    f"{OUT}/submission_fixed_scale_shift.csv", index=False)
pd.DataFrame({"ID": sorted_ids, TARGET: fixed_lt_cal}).to_csv(
    f"{OUT}/submission_fixed_lt_cal.csv", index=False)
pd.DataFrame({"ID": sorted_ids, TARGET: fixed_lt_ts_cal}).to_csv(
    f"{OUT}/submission_fixed_lt_ts_cal.csv", index=False)

print(f"\n  Submissions:")
print(f"    results/dist_calibration/submission_fixed_scale_shift.csv")
print(f"    results/dist_calibration/submission_fixed_lt_cal.csv")
print(f"    results/dist_calibration/submission_fixed_lt_ts_cal.csv")

# === Summary ===
print(f"\n{'='*64}")
print(f"Summary:")
print(f"  best global shift: {best_shift:+.2f} → delta {best_mae - baseline:+.5f}")
print(f"  best global scale: {best_scale:.3f} → delta {best_scale_mae - baseline:+.5f}")
print(f"  combined:          s={opt_s:.3f} c={opt_c:+.3f} → delta {res.fun - baseline:+.5f}")
print(f"  per-layout_type:   delta {cal_mae - baseline:+.5f}")
print(f"  per-(lt × ts):     delta {cal_mae2 - baseline:+.5f}")

json.dump({
    "baseline": float(baseline),
    "best_global_shift": float(best_shift),
    "best_global_shift_delta": float(best_mae - baseline),
    "best_global_scale": float(best_scale),
    "best_global_scale_delta": float(best_scale_mae - baseline),
    "combined_ss_scale": float(opt_s),
    "combined_ss_shift": float(opt_c),
    "combined_ss_delta": float(res.fun - baseline),
    "per_lt_delta": float(cal_mae - baseline),
    "per_lt_ts_delta": float(cal_mae2 - baseline),
    "shifts_lt": {lt: float(shifts_lt[lt]) for lt in shifts_lt},
    "mega33_oof_mean": float(mega33_oof.mean()),
    "mega33_test_mean": float(mega33_test.mean()),
    "y_train_mean": float(y.mean()),
    "y_train_median": float(np.median(y)),
}, open(f"{OUT}/summary.json","w"), indent=2)

print(f"\n  Saved: {OUT}/summary.json")
