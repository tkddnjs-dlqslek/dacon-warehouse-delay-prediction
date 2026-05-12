"""
Leak hunt v2: Deep dive into scenario_id / ID / column-combo patterns.

Goal: find any hidden identifier or number pattern that encodes target info.
Checked in EDA v30 S5 only briefly (numeric range overlap). Here we go deeper:
- modulo, bit patterns, prime factors
- column combinations as "de facto" identifiers
- scenario_id vs timeslot × target joint distribution
"""
import pickle, numpy as np, pandas as pd, time, warnings
warnings.filterwarnings("ignore")
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import os

OUT = "results/leak_hunt"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("Leak Hunt v2: deep number pattern + column combo analysis")
print("=" * 64)
t0 = time.time()

train = pd.read_csv("train.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
test = pd.read_csv("test.csv").sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
y = train[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
residual = y - mega33_oof
baseline_mae = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline_mae:.5f}")

# --- parse numeric parts ---
def parse(s):
    return int(s.split("_")[-1])

sc_num = train["scenario_id"].apply(parse).values
id_num = train["ID"].apply(parse).values
lay_num = train["layout_id"].apply(parse).values

sc_num_te = test["scenario_id"].apply(parse).values
id_num_te = test["ID"].apply(parse).values
lay_num_te = test["layout_id"].apply(parse).values

print(f"\n[1] Basic ranges")
print(f"  train scenario_num: {sc_num.min()} ~ {sc_num.max()}")
print(f"  test  scenario_num: {sc_num_te.min()} ~ {sc_num_te.max()}")
print(f"  train layout_num: {lay_num.min()} ~ {lay_num.max()}")
print(f"  test  layout_num: {lay_num_te.min()} ~ {lay_num_te.max()}")

# --- modulo patterns ---
print(f"\n[2] Modulo patterns on scenario_num vs target")
for m in [2, 3, 5, 7, 10, 11, 13, 17, 25, 50, 100, 500, 1000]:
    mod = sc_num % m
    # target mean by mod
    grp_mean = pd.DataFrame({"mod": mod, "y": y, "res": residual}).groupby("mod").agg(
        y_mean=("y", "mean"), y_std=("y", "std"),
        res_mean=("res", "mean"), res_std=("res", "std"),
        n=("y", "count"))
    y_cv = grp_mean["y_mean"].std() / grp_mean["y_mean"].mean()
    res_std_of_mean = grp_mean["res_mean"].std()
    rho_y = spearmanr(mod, y).statistic
    rho_res = spearmanr(mod, residual).statistic
    if abs(rho_y) > 0.02 or abs(rho_res) > 0.02 or res_std_of_mean > 0.5:
        print(f"  mod={m}: ρ(y)={rho_y:+.4f} ρ(res)={rho_res:+.4f} "
              f"res_mean_std={res_std_of_mean:.4f} y_mean_cv={y_cv:.4f}")

# --- bit patterns ---
print(f"\n[3] Bit patterns on scenario_num")
for bit in range(14):
    b = (sc_num >> bit) & 1
    if b.sum() == 0 or b.sum() == len(b):
        continue
    rho = spearmanr(b, y).statistic
    rho_r = spearmanr(b, residual).statistic
    if abs(rho) > 0.01 or abs(rho_r) > 0.01:
        print(f"  bit {bit}: ρ(y)={rho:+.4f} ρ(res)={rho_r:+.4f} ones={b.sum()}")

# --- number of divisors / factor patterns ---
print(f"\n[4] Divisor count patterns")
def divisor_count(n):
    if n <= 0: return 0
    c = 0
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            c += 1
            if i != n // i: c += 1
    return c

# sample 10000 for speed
np.random.seed(42)
idx = np.random.choice(len(sc_num), 10000, replace=False)
divs = np.array([divisor_count(int(sc_num[i])) for i in idx])
y_sample = y[idx]
res_sample = residual[idx]
rho = spearmanr(divs, y_sample).statistic
rho_r = spearmanr(divs, res_sample).statistic
print(f"  divisor_count vs y: ρ={rho:+.4f}")
print(f"  divisor_count vs residual: ρ={rho_r:+.4f}")

# --- column combinations as identifier ---
print(f"\n[5] Column combos as implicit IDs (checking cardinality)")
# check if any small set of columns uniquely identifies (layout, scenario)
layout_cols = ["floor_area_sqm", "robot_total", "charger_count", "pack_station_count",
               "layout_compactness", "aisle_width_avg"]
valid_cols = [c for c in layout_cols if c in train.columns]
for n_cols in [1, 2, 3]:
    from itertools import combinations
    for combo in combinations(valid_cols, n_cols):
        combo_vals = train[list(combo)].astype(str).agg("_".join, axis=1)
        unique_per_layout = combo_vals.groupby(train["layout_id"]).nunique()
        # if this combo is 1 per layout → ID proxy
        if (unique_per_layout == 1).all():
            # then combo == layout_id
            combo_unique = combo_vals.nunique()
            if combo_unique == train["layout_id"].nunique():
                print(f"  {'+'.join(combo)}: **unique layout ID** ({combo_unique} values)")
            else:
                print(f"  {'+'.join(combo)}: consistent per layout but not unique ({combo_unique} vals for {train['layout_id'].nunique()} layouts)")

# --- scenario_num within layout pattern ---
print(f"\n[6] scenario_num within layout: is it (layout_seed) or (global_seed)?")
# for each layout, are scenario_ids unique range, or repeated?
lay_df = train[["layout_id", "scenario_id"]].drop_duplicates()
lay_sc_count = lay_df.groupby("layout_id").size()
print(f"  scenarios per layout: min={lay_sc_count.min()} max={lay_sc_count.max()} mean={lay_sc_count.mean():.1f}")
# check if scenario_ids are shared or disjoint across layouts
sc_uniq_global = train["scenario_id"].nunique()
print(f"  total unique scenario_ids: {sc_uniq_global}")
# if scenarios are disjoint (each scenario is in only 1 layout), scenario_id = (layout × scenario_num) encoded
sc_per_layout = lay_df.groupby("scenario_id")["layout_id"].nunique()
print(f"  scenarios in exactly 1 layout: {(sc_per_layout == 1).sum()} / {len(sc_per_layout)}")

# --- timeslot × scenario_num pattern ---
print(f"\n[7] timeslot × scenario_num joint pattern")
train["timeslot"] = train.groupby(["layout_id", "scenario_id"]).cumcount()
train["ts_x_sc"] = train["timeslot"] * sc_num
rho = spearmanr(train["ts_x_sc"], y).statistic
print(f"  timeslot * scenario_num vs y: ρ={rho:+.4f}")
# joint mean
joint = train.groupby(["timeslot"]).agg(y_mean=(TARGET, "mean"), res_mean=("timeslot", "count"))
# Wait that's wrong. Let me redo

# --- target distribution by scenario_num bucket ---
print(f"\n[8] Target quantiles by scenario_num decile")
deciles = pd.qcut(sc_num, 10, labels=False, duplicates="drop")
dec_df = pd.DataFrame({"decile": deciles, "y": y, "res": residual})
for d in sorted(dec_df["decile"].unique()):
    sub = dec_df[dec_df["decile"] == d]
    print(f"  decile {d}: y_mean={sub['y'].mean():.3f} res_mean={sub['res'].mean():.3f} "
          f"n={len(sub)} sc_range={sc_num[deciles==d].min()}~{sc_num[deciles==d].max()}")

# --- ID_num consecutive diff / gap patterns ---
print(f"\n[9] ID number patterns within train")
sorted_id = np.sort(id_num)
gaps = np.diff(sorted_id)
print(f"  ID gaps: min={gaps.min()} max={gaps.max()} mean={gaps.mean():.2f} uniq={np.unique(gaps).shape}")

# --- done ---
print(f"\nTotal elapsed: {time.time() - t0:.0f}s")
print("=" * 64)
print("Leak Hunt done. Look for any rho > 0.02 or anomalous patterns above.")
print("=" * 64)
