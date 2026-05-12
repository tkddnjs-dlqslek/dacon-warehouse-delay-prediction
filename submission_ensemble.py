"""
Ensemble of LB-tested submissions + exotic blends.

Test multiple combinations of existing submissions to find LB improvement.
"""
import pandas as pd, numpy as np, os
from sklearn.metrics import mean_absolute_error
import pickle

OUT = "results/submission_ensemble"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

# Load submissions
subs = {
    "mega33_avg": pd.read_csv("submission_mega33_avg.csv"),  # LB 9.7759
    "fixed": pd.read_csv("results/final_blend/submission_final_multiblend_FIXED.csv"),  # LB 9.77116
    "v8": pd.read_csv("results/layout_ranking/submission_megablend_v8.csv"),  # LB 9.77116
}

# Align by ID
ref_ids = subs["mega33_avg"]["ID"].values
for k, df in subs.items():
    df = df.set_index("ID").loc[ref_ids].reset_index()
    subs[k] = df[TARGET].values

print(f"aligned, n_rows={len(ref_ids)}")

# Compare pairs
for a in subs:
    for b in subs:
        if a >= b: continue
        diff = np.abs(subs[a] - subs[b]).mean()
        print(f"  mean_abs_diff({a}, {b}) = {diff:.4f}")

# Load ground truth OOF to estimate LB proxy
# But we don't have ground truth for test, so we can only compute:
# - OOF correlations
# But we already have those. Let me skip and just generate blend CSVs.

# Various blends
blends = {
    "fixed_plus_mega33_10": 0.9 * subs["fixed"] + 0.1 * subs["mega33_avg"],
    "fixed_plus_mega33_20": 0.8 * subs["fixed"] + 0.2 * subs["mega33_avg"],
    "fixed_plus_mega33_30": 0.7 * subs["fixed"] + 0.3 * subs["mega33_avg"],
    "fixed_plus_v8_50": 0.5 * subs["fixed"] + 0.5 * subs["v8"],
    "all_three_avg": (subs["fixed"] + subs["mega33_avg"] + subs["v8"]) / 3,
    # geometric mean (for positive predictions)
    "fixed_mega_geom": np.exp(0.7 * np.log(subs["fixed"] + 1e-3) + 0.3 * np.log(subs["mega33_avg"] + 1e-3)) - 1e-3,
}

for name, preds in blends.items():
    preds_clipped = np.clip(preds, 0, None)
    pd.DataFrame({"ID": ref_ids, TARGET: preds_clipped}).to_csv(
        f"{OUT}/submission_{name}.csv", index=False)
    # Report diff from FIXED (our current best)
    diff = np.abs(preds_clipped - subs["fixed"]).mean()
    print(f"  {name}: saved, mean_abs_diff_vs_fixed={diff:.4f}")

print(f"\nAll blends saved to {OUT}/")
print("Try LB of 'fixed_plus_mega33_20' first (0.8 FIXED + 0.2 mega33) - most safe variation")
