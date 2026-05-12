"""
Phase 1: Simulation reverse engineering - find deterministic patterns.

Strategy:
1. Identify "simple" scenarios (low load, no faults, no congestion)
2. Check if y is predictable by simple formulas
3. Fit queuing theory: Little's Law, M/M/c wait time
4. Compare actual vs predicted → identify deterministic signal
"""
import pickle, numpy as np, pandas as pd, os, json
import warnings; warnings.filterwarnings("ignore")
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error

OUT = "results/phase1_sim"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

train = pd.read_csv("train.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
train["timeslot"] = train.groupby(["layout_id", "scenario_id"]).cumcount()
layout = pd.read_csv("layout_info.csv")
train = train.merge(layout, on="layout_id", how="left")
y = train[TARGET].values

print("=" * 64)
print("Phase 1: Simulation Reverse Engineering")
print("=" * 64)
print(f"train shape: {train.shape}")
print(f"y: mean={y.mean():.3f} median={np.median(y):.3f} std={y.std():.3f}")

# ===== Step 1: Examine the simplest regime =====
print("\n[1] Finding SIMPLE scenarios (low stress)...")

# Define "simple" = low load + no faults + low congestion
simple_cond = (
    (train["order_inflow_15m"] <= train["order_inflow_15m"].quantile(0.2)) &
    (train["fault_count_15m"] == 0) &
    (train["blocked_path_15m"] == 0) &
    (train["congestion_score"] <= train["congestion_score"].quantile(0.2)) &
    (train["robot_utilization"] <= 0.3)
)
simple_df = train[simple_cond].copy()
print(f"simple rows: {len(simple_df)} ({100*len(simple_df)/len(train):.2f}%)")
print(f"simple y: mean={simple_df[TARGET].mean():.3f} median={simple_df[TARGET].median():.3f}")
print(f"simple y dist:\n  q10={simple_df[TARGET].quantile(0.1):.2f} q50={simple_df[TARGET].quantile(0.5):.2f} "
      f"q90={simple_df[TARGET].quantile(0.9):.2f}")

# Check: is y deterministic (constant) in simple regime? Or still highly variable?
print(f"\n  simple y CV: {simple_df[TARGET].std() / (simple_df[TARGET].mean() + 1e-6):.3f}")
print(f"  all y CV:    {y.std() / (y.mean() + 1e-6):.3f}")
# if simple CV << all CV → deterministic signal in simple regime
# if similar → y is inherently stochastic

# ===== Step 2: Check linear relationships in simple regime =====
print("\n[2] Linear relationships in simple regime:")
candidate_vars = [
    "order_inflow_15m", "robot_active", "robot_utilization",
    "battery_mean", "charge_queue_length", "pack_utilization",
    "avg_trip_distance", "timeslot",
]
for v in candidate_vars:
    if v in simple_df.columns:
        r_s = spearmanr(simple_df[v], simple_df[TARGET]).statistic
        r_p = pearsonr(simple_df[v].fillna(0), simple_df[TARGET])[0]
        print(f"  {v:30s}: spearman={r_s:+.4f} pearson={r_p:+.4f}")

# ===== Step 3: Try Little's Law prediction =====
print("\n[3] Little's Law prediction test: y ~ λ/μ")
# Treat order_inflow_15m as λ (per 15 min), robot_active·speed as μ
# Convert to consistent units: λ=orders per minute, μ=orders per minute served
lam_full = train["order_inflow_15m"].values / 15.0  # orders per min
# service rate estimate: robots × (1 / avg_trip_distance) assuming 1 order per trip
mu_full = train["robot_active"].values / (train["avg_trip_distance"].fillna(1).values + 1e-6)
# utilization
rho_full = lam_full / (mu_full + 1e-6)
rho_full = np.clip(rho_full, 0, 0.99)

# Little's law predicts: L (queue) = λ*W. If W is delay and L is queue length proxy,
# then W = L / λ
# For M/M/1 formula: W = 1/(μ - λ) = 1/μ · 1/(1-ρ)
mm1_W = 1.0 / (mu_full * (1 - rho_full) + 1e-6)   # minutes
# convert delay minutes to match scale
# Check correlation with actual delay
valid_mask = (mu_full > 0) & np.isfinite(mm1_W) & (mm1_W < 1000)
r_s_mm1 = spearmanr(mm1_W[valid_mask], y[valid_mask]).statistic
print(f"  M/M/1 wait prediction vs y: spearman={r_s_mm1:+.4f}")

# M/M/c approximation (c = robot_active)
# Erlang C approximation: P(wait) ≈ ρ^c, W_q ≈ P(wait)·1/(c·μ·(1-ρ))
c_rob = train["robot_active"].values + 1
pi_wait = rho_full ** c_rob
mmc_W = pi_wait / (c_rob * mu_full * (1 - rho_full) + 1e-6)
r_s_mmc = spearmanr(mmc_W[valid_mask], y[valid_mask]).statistic
print(f"  M/M/c wait prediction vs y: spearman={r_s_mmc:+.4f}")

# ===== Step 4: Check if simulation has nearly constant unexplained noise =====
print("\n[4] Irreducible noise test:")
# Group by identical feature vectors → if same features → same y, → no noise
# Group by (layout_id, some coarse feature bin)
bins_inflow = pd.cut(train["order_inflow_15m"], bins=20, labels=False)
bins_util = pd.cut(train["robot_utilization"], bins=10, labels=False)
bins_cong = pd.cut(train["congestion_score"], bins=10, labels=False)
group_key = (
    bins_inflow.astype(str) + "_" + bins_util.astype(str) + "_" +
    bins_cong.astype(str) + "_" + train["layout_type"].astype(str)
)
group_y_std = train.groupby(group_key)[TARGET].agg(["mean","std","count"])
group_y_std = group_y_std[group_y_std["count"] > 20]
print(f"  groups with >20 samples: {len(group_y_std)}")
print(f"  avg within-group std: {group_y_std['std'].mean():.3f}")
print(f"  (global y std: {y.std():.3f})")
print(f"  ratio: {group_y_std['std'].mean() / y.std():.3f}")
# if ratio small → y is mostly determined by group → deterministic
# if ratio large → high within-group noise → stochastic

# ===== Step 5: Per-scenario variance vs overall =====
print("\n[5] Scenario-level deterministic test:")
sc_stats = train.groupby(["layout_id","scenario_id"])[TARGET].agg(["mean","std","min","max","count"])
sc_stats = sc_stats[sc_stats["count"] >= 20]
print(f"  scenarios with >=20 ts: {len(sc_stats)}")
print(f"  avg within-scenario std: {sc_stats['std'].mean():.3f}")
print(f"  avg scenario range: {(sc_stats['max']-sc_stats['min']).mean():.3f}")
print(f"  (scenario is 25 timesteps; within-scenario std {sc_stats['std'].mean():.3f}"
      f" vs global {y.std():.3f})")

# ===== Step 6: Residual pattern of mega33 in simple regime =====
print("\n[6] Does mega33 make simple-regime errors?")
with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
simple_idx = np.where(simple_cond)[0]
mega_simple_mae = mean_absolute_error(y[simple_idx], mega33_oof[simple_idx])
mega_all_mae = mean_absolute_error(y, mega33_oof)
print(f"  mega33 MAE on simple: {mega_simple_mae:.4f}")
print(f"  mega33 MAE overall:   {mega_all_mae:.4f}")

# ===== Step 7: Fit Erlang C formula coefficients in simple regime =====
print("\n[7] Fitting simple linear model on simple regime with queuing vars:")
from sklearn.linear_model import LinearRegression
Xs = np.column_stack([
    lam_full[simple_idx],
    mu_full[simple_idx],
    rho_full[simple_idx],
    rho_full[simple_idx] ** 2,
    mm1_W[simple_idx],
    1.0 / (mu_full[simple_idx] + 0.1),
    train["order_inflow_15m"].values[simple_idx],
    train["robot_active"].values[simple_idx],
])
Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
ys = y[simple_idx]
lr = LinearRegression().fit(Xs, ys)
pred_lr = lr.predict(Xs)
mae_lr = mean_absolute_error(ys, pred_lr)
print(f"  linear regression on simple: MAE={mae_lr:.4f} (mega33 on simple: {mega_simple_mae:.4f})")
print(f"  coeffs: {lr.coef_.round(4)}")
print(f"  intercept: {lr.intercept_:.4f}")

# Summary
print(f"\n{'='*64}")
print("Phase 1 DIAGNOSTIC SUMMARY")
print(f"{'='*64}")

# Is there deterministic signal?
if group_y_std['std'].mean() / y.std() < 0.3:
    print("✓ STRONG deterministic signal (within-group std << global)")
    print("  → Phase 2 should find analytical formula improvements")
elif group_y_std['std'].mean() / y.std() < 0.6:
    print("~ MODERATE determinism (some irreducible noise)")
    print("  → Phase 2 queuing features may help")
else:
    print("✗ HIGH stochasticity (likely simulation inherent noise)")
    print("  → Analytical approach limited; noise floor near current LB")

print(f"\n  simple regime: y CV = {simple_df[TARGET].std() / (simple_df[TARGET].mean() + 1e-6):.3f}")
print(f"  mega33 simple MAE = {mega_simple_mae:.4f}, overall = {mega_all_mae:.4f}")
print(f"  if simple MAE much lower → simple regime is more deterministic")
print(f"  ratio: {mega_simple_mae / mega_all_mae:.3f}")

summary = {
    "simple_y_cv": float(simple_df[TARGET].std() / (simple_df[TARGET].mean()+1e-6)),
    "all_y_cv": float(y.std() / (y.mean()+1e-6)),
    "within_group_std_ratio": float(group_y_std['std'].mean() / y.std()),
    "mega33_mae_simple": float(mega_simple_mae),
    "mega33_mae_all": float(mega_all_mae),
    "mm1_spearman": float(r_s_mm1),
    "mmc_spearman": float(r_s_mmc),
    "simple_regime_linear_mae": float(mae_lr),
    "simple_n": int(len(simple_df)),
}
json.dump(summary, open(f"{OUT}/summary.json","w"), indent=2)
print(f"\nSaved: {OUT}/summary.json")
