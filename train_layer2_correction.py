"""
Layer 2 Meta-Correction: train a meta-learner using all our model OOFs as features,
with sample-weight emphasis on hub_spoke × late-timeslot (the worst bucket).

Design:
- Features: [mega33_oof, rank_orig, iter_r1, iter_r2, iter_r3, res_rank, bucket flags, key v23 cols]
- Target: y (direct)
- Sample weight: 3.0 for hub_spoke ts3+4, 1.0 otherwise
- 5-fold GroupKFold on layout_id

Goal: meta-learner learns when to override mega33, focused on worst bucket.
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/layer2"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Layer 2 Meta-Correction (hub_spoke late-ts focus)")
print("=" * 64)
t0 = time.time()

with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline:.5f}")

# --- Identify hub_spoke + late_ts subset ---
layout = pd.read_csv("layout_info.csv")[["layout_id", "layout_type"]]
meta = train_fe[["layout_id"]].merge(layout, on="layout_id", how="left")
layout_type_train = meta["layout_type"].values

meta_te = test_fe[["layout_id"]].merge(layout, on="layout_id", how="left")
layout_type_test = meta_te["layout_type"].values

timeslot_train = train_fe["timeslot"].values
timeslot_test = test_fe["timeslot"].values

# hub_spoke AND ts >= 15 (ts_bucket 3 or 4)
hs_latets_train = (layout_type_train == "hub_spoke") & (timeslot_train >= 15)
hs_latets_test = (layout_type_test == "hub_spoke") & (timeslot_test >= 15)
print(f"hub_spoke+late_ts: train {hs_latets_train.sum()} ({100*hs_latets_train.mean():.1f}%), "
      f"test {hs_latets_test.sum()} ({100*hs_latets_test.mean():.1f}%)")

# Current MAE on this subset
sub_mae_before = mean_absolute_error(y[hs_latets_train], mega33_oof[hs_latets_train])
print(f"mega33 MAE on hub_spoke late_ts: {sub_mae_before:.5f}")

# --- Load all model OOFs and test predictions ---
print("\n[1] Loading model OOFs/tests as meta-features...")
oof_dict, test_dict = {}, {}
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("iter_r1", np.load("results/iter_pseudo/round1_oof.npy"),
     np.load("results/iter_pseudo/round1_test.npy")),
    ("iter_r2", np.load("results/iter_pseudo/round2_oof.npy"),
     np.load("results/iter_pseudo/round2_test.npy")),
    ("iter_r3", np.load("results/iter_pseudo/round3_oof.npy"),
     np.load("results/iter_pseudo/round3_test.npy")),
    ("res_rank", np.load("results/residual_ranking/res_rank_oof.npy"),
     np.load("results/residual_ranking/res_rank_test.npy")),
    ("cluster", np.load("results/cluster_spec/cluster_oof.npy"),
     np.load("results/cluster_spec/cluster_test.npy")),
    ("vorth", np.load("results/vorth/vorth_oof.npy"),
     np.load("results/vorth/vorth_test.npy")),
]
meta_feat_names = [s[0] for s in sources]
meta_feats_train = np.column_stack([s[1] for s in sources])
meta_feats_test = np.column_stack([s[2] for s in sources])

# Add context features: hs_flag, ts_bucket, timeslot, ts_norm, layout_type codes
context_train = np.column_stack([
    hs_latets_train.astype(np.float32),
    timeslot_train.astype(np.float32) / 24.0,
    (timeslot_train // 5).astype(np.float32),
    (layout_type_train == "hub_spoke").astype(np.float32),
    (layout_type_train == "hybrid").astype(np.float32),
    (layout_type_train == "grid").astype(np.float32),
    (layout_type_train == "narrow").astype(np.float32),
])
context_test = np.column_stack([
    hs_latets_test.astype(np.float32),
    timeslot_test.astype(np.float32) / 24.0,
    (timeslot_test // 5).astype(np.float32),
    (layout_type_test == "hub_spoke").astype(np.float32),
    (layout_type_test == "hybrid").astype(np.float32),
    (layout_type_test == "grid").astype(np.float32),
    (layout_type_test == "narrow").astype(np.float32),
])
context_names = ["hs_late_flag", "ts_norm", "ts_bucket", "is_hs", "is_hybrid", "is_grid", "is_narrow"]

# Add some key v23 features (top importance)
key_v23 = ["battery_mean_sc_max", "avg_trip_distance", "pack_utilization_sc_mean",
           "order_inflow_15m_sc_min", "battery_mean_sc_std", "floor_area_per_robot",
           "congestion_score", "fault_count_15m", "charge_queue_length", "robot_utilization"]
key_v23_avail = [c for c in key_v23 if c in train_fe.columns and c in test_fe.columns]
v23_feat_train = train_fe[key_v23_avail].fillna(0).values.astype(np.float32)
v23_feat_test = test_fe[key_v23_avail].fillna(0).values.astype(np.float32)

X_tr = np.hstack([meta_feats_train, context_train, v23_feat_train])
X_te = np.hstack([meta_feats_test, context_test, v23_feat_test])
all_feat_names = meta_feat_names + context_names + key_v23_avail
print(f"  X_tr: {X_tr.shape}  features: {len(all_feat_names)}")

# --- Train layer 2 with sample_weight emphasizing hub_spoke late_ts ---
print("\n[2] Training layer-2 LGB with hub_spoke late_ts sample weight 3x...")
y_log = np.log1p(y)
sample_w = np.ones(len(y), dtype=np.float32)
sample_w[hs_latets_train] = 3.0
print(f"  total weight: base {hs_latets_train.sum() * 3 + (~hs_latets_train).sum():,}")

PARAMS = dict(
    objective="huber", n_estimators=3000, learning_rate=0.03,
    num_leaves=31, max_depth=5, min_child_samples=100,  # smaller model (meta-learner)
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

layer2_oof = np.zeros(len(y))
layer2_test = np.zeros(len(X_te))
imp_accum = np.zeros(X_tr.shape[1])

for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr], y_log[tr], sample_weight=sample_w[tr],
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    layer2_oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    layer2_test += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
    imp_accum += m.feature_importances_
    print(f"  fold {f}: best_iter={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

l2_mae = mean_absolute_error(y, layer2_oof)
l2_corr = float(np.corrcoef(y - mega33_oof, y - layer2_oof)[0, 1])
print(f"\n  Layer-2 OOF full: {l2_mae:.5f}")
print(f"  residual_corr(mega33, layer2): {l2_corr:.4f}")

# check subset performance
l2_sub_mae = mean_absolute_error(y[hs_latets_train], layer2_oof[hs_latets_train])
print(f"  Layer-2 MAE on hub_spoke late_ts: {l2_sub_mae:.5f}  (mega33: {sub_mae_before:.5f})")
print(f"  subset delta: {l2_sub_mae - sub_mae_before:+.5f}")

np.save(f"{OUT}/layer2_oof.npy", layer2_oof)
np.save(f"{OUT}/layer2_test.npy", layer2_test)

# Feature importance
imp = pd.Series(imp_accum / 5, index=all_feat_names).sort_values(ascending=False)
print(f"\n[3] Top 15 features:")
for i, (n_, v_) in enumerate(imp.head(15).items()):
    print(f"  {i+1:2d}. {n_:30s}: {v_:.0f}")

# --- Substitute only in target subset ---
print(f"\n[4] Substitute layer2 predictions only in hub_spoke late_ts subset:")
swap_oof = mega33_oof.copy()
swap_oof[hs_latets_train] = layer2_oof[hs_latets_train]
swap_mae = mean_absolute_error(y, swap_oof)
print(f"  swap OOF: {swap_mae:.5f}  delta: {swap_mae - baseline:+.5f}")

# Try partial substitution (blend on subset)
print(f"  partial substitution:")
for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    partial_oof = mega33_oof.copy()
    partial_oof[hs_latets_train] = (1 - alpha) * mega33_oof[hs_latets_train] + alpha * layer2_oof[hs_latets_train]
    p_mae = mean_absolute_error(y, partial_oof)
    d = p_mae - baseline
    tag = " ***" if d < -0.003 else (" **" if d < -0.001 else "")
    print(f"    alpha={alpha:.1f}: mae={p_mae:.5f} delta={d:+.5f}{tag}")

# Full multi-blend v6 with layer2 as new candidate
print("\n[5] Full mega-blend v6 (adding layer2)...")
from scipy.optimize import minimize
sources2 = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("layer2", layer2_oof, layer2_test),
    ("iter_r2", np.load("results/iter_pseudo/round2_oof.npy"),
     np.load("results/iter_pseudo/round2_test.npy")),
    ("iter_r3", np.load("results/iter_pseudo/round3_oof.npy"),
     np.load("results/iter_pseudo/round3_test.npy")),
]
names2 = [s[0] for s in sources2]
oofs_M = np.column_stack([s[1] for s in sources2])
tests_M = np.column_stack([s[2] for s in sources2])
n = len(sources2)

def obj_fn(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs_M @ w)

x0 = np.zeros(n); x0[0] = 0.75
for i in range(1, n): x0[i] = 0.25 / (n - 1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol": 1e-7, "maxiter": 100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline

print(f"\nFinal mega-blend v6 weights:")
for nm, w_ in zip(names2, w_opt):
    if w_ > 0.001:
        print(f"  {nm:12s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2: -0.005434")
print(f"Improvement: {delta - (-0.005434):+.6f}")

# submission
blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v6.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v6.csv")

json.dump({
    "baseline": float(baseline),
    "layer2_oof_mae": float(l2_mae),
    "layer2_corr": l2_corr,
    "layer2_subset_mae": float(l2_sub_mae),
    "mega33_subset_mae": float(sub_mae_before),
    "swap_delta": float(swap_mae - baseline),
    "full_blend_weights": dict(zip(names2, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/layer2_summary.json", "w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
