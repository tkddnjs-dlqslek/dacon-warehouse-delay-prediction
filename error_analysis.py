"""
Approach 1: Error-driven feature discovery.

Goal: Find what's specific about rows where mega33 fails most.

Steps:
  1. Compute |residual| on train (y - mega33_oof)
  2. Split: worst 5% (n~12500) vs rest
  3. For each feature (v23 + raw 94 columns), compute distribution shift:
     - Cohen's d (standardized mean difference)
     - KS statistic
     - Mean ratio, std ratio
  4. Rank features by effect size
  5. Meta-level: which layout_type / ts_bucket / scenario patterns dominate worst rows
  6. Output hypothesis list for feature generation
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "error_analysis")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Approach 1: Error-driven feature discovery", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    oof = mega["meta_avg_oof"]
    resid = y - oof
    abs_resid = np.abs(resid)

    print(f"Residual stats: mean={resid.mean():+.3f}, std={resid.std():.3f}", flush=True)
    print(f"|Residual| stats: mean={abs_resid.mean():.3f}, median={np.median(abs_resid):.3f}, max={abs_resid.max():.1f}", flush=True)

    # Worst 5%
    thresh = np.quantile(abs_resid, 0.95)
    worst_mask = abs_resid >= thresh
    print(f"\nTop 5% threshold: |resid| >= {thresh:.3f} → {int(worst_mask.sum())} rows", flush=True)
    print(f"Top 5% MAE contribution: {abs_resid[worst_mask].mean() * worst_mask.mean():.3f} / total {abs_resid.mean():.3f}", flush=True)
    print(f"  = {abs_resid[worst_mask].mean() * worst_mask.mean() / abs_resid.mean() * 100:.1f}% of total MAE", flush=True)

    # Split direction: under-predict vs over-predict within worst
    worst_under = (resid[worst_mask] > 0).sum()  # y > pred → under-predict
    worst_over = (resid[worst_mask] < 0).sum()
    print(f"\nDirection in worst: under-pred={worst_under} ({worst_under/worst_mask.sum()*100:.1f}%), over-pred={worst_over} ({worst_over/worst_mask.sum()*100:.1f}%)", flush=True)

    # y distribution in worst
    y_worst = y[worst_mask]
    print(f"\ny in worst: mean={y_worst.mean():.2f}, median={np.median(y_worst):.2f}, min={y_worst.min():.2f}, max={y_worst.max():.2f}", flush=True)
    print(f"y in normal: mean={y[~worst_mask].mean():.2f}, median={np.median(y[~worst_mask]):.2f}", flush=True)

    # ─── Load v23 features + raw + static ───
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])

    # Raw columns (from train.csv)
    exclude = {"ID", "layout_id", "scenario_id", "avg_delay_minutes_next_30m"}
    raw_cols = [c for c in train.columns if c not in exclude]

    # All features to analyze: v23 + raw (some overlap)
    all_features = {}
    for c in feat_cols:
        if c in train_fe.columns:
            all_features[f"v23_{c}"] = train_fe[c].values
    for c in raw_cols:
        if c in train.columns:
            all_features[f"raw_{c}"] = train[c].values.astype(np.float64)

    # Static layout_info
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    train_li = train[["layout_id"]].merge(layout_info, on="layout_id", how="left")
    for c in layout_info.columns:
        if c in ("layout_id", "layout_type"):
            continue
        all_features[f"static_{c}"] = train_li[c].values.astype(np.float64)

    print(f"\nFeatures to analyze: {len(all_features)}", flush=True)

    # ─── Distribution shift analysis ───
    results = []
    for name, x in all_features.items():
        mask_valid = np.isfinite(x)
        xw = x[mask_valid & worst_mask]
        xn = x[mask_valid & ~worst_mask]
        if len(xw) < 50 or len(xn) < 50:
            continue
        mw, mn = xw.mean(), xn.mean()
        sw, sn = xw.std(), xn.std()
        # Pooled std for Cohen's d
        pooled = np.sqrt((sw**2 + sn**2) / 2) if (sw**2 + sn**2) > 0 else 1e-9
        cohen_d = (mw - mn) / (pooled + 1e-9)
        # KS test
        try:
            ks_stat, _ = ks_2samp(xw, xn)
        except Exception:
            ks_stat = 0.0
        results.append(dict(
            feature=name, mean_worst=mw, mean_normal=mn,
            std_worst=sw, std_normal=sn,
            cohen_d=float(cohen_d), abs_d=float(abs(cohen_d)),
            ks=float(ks_stat),
        ))
    res = pd.DataFrame(results).sort_values("abs_d", ascending=False)
    print("\n=== Top 30 features by Cohen's d (distribution shift worst vs normal) ===", flush=True)
    print(res.head(30)[["feature", "mean_worst", "mean_normal", "cohen_d", "ks"]].to_string(index=False), flush=True)

    res.to_csv(os.path.join(OUT, "feature_shift_ranking.csv"), index=False)

    # ─── Meta-level: layout_type / ts_bucket distribution in worst ───
    print("\n=== Meta breakdown of worst 5% ===", flush=True)
    # layout_type
    train_li["_y"] = y
    train_li["_worst"] = worst_mask
    lt_frac_worst = train_li[train_li["_worst"]]["layout_type"].value_counts(normalize=True)
    lt_frac_all = train_li["layout_type"].value_counts(normalize=True)
    print("\nlayout_type fraction (worst vs all):", flush=True)
    for lt in lt_frac_all.index:
        print(f"  {lt:<10}: worst {lt_frac_worst.get(lt, 0)*100:5.1f}%  all {lt_frac_all[lt]*100:5.1f}%  ratio={lt_frac_worst.get(lt, 0)/lt_frac_all[lt]:.2f}", flush=True)

    # ts_bucket (0-24 row-in-scenario)
    ts_idx = np.tile(np.arange(25), len(train) // 25)
    ts_bucket = np.where(ts_idx <= 7, "early", np.where(ts_idx <= 15, "mid", "late"))
    tb_frac_worst = pd.Series(ts_bucket[worst_mask]).value_counts(normalize=True)
    tb_frac_all = pd.Series(ts_bucket).value_counts(normalize=True)
    print("\nts_bucket fraction (worst vs all):", flush=True)
    for tb in ["early", "mid", "late"]:
        print(f"  {tb:<7}: worst {tb_frac_worst.get(tb, 0)*100:5.1f}%  all {tb_frac_all.get(tb, 0)*100:5.1f}%  ratio={tb_frac_worst.get(tb, 0)/tb_frac_all.get(tb, 1):.2f}", flush=True)

    # ─── Binary "worst" classifier AUC per top feature ───
    # Check if individual features can distinguish worst vs normal
    from sklearn.metrics import roc_auc_score
    print("\n=== Per-feature AUC (worst=1, normal=0), top 15 by Cohen's d ===", flush=True)
    for _, row in res.head(15).iterrows():
        name = row["feature"]
        x = all_features[name]
        valid = np.isfinite(x)
        try:
            auc = roc_auc_score(worst_mask[valid], x[valid])
            # Handle direction: AUC < 0.5 means lower values predict worst
            auc_adj = auc if auc >= 0.5 else 1 - auc
            print(f"  {name:<50s} cohen_d={row['cohen_d']:+.3f} AUC={auc_adj:.3f}", flush=True)
        except Exception:
            pass

    # ─── Scenario-level pattern: do worst rows cluster in specific scenarios? ───
    # Count worst rows per scenario (each scenario has 25 rows)
    n_sc = len(train) // 25
    worst_matrix = worst_mask.reshape(n_sc, 25)
    worst_per_sc = worst_matrix.sum(axis=1)
    print(f"\n=== Scenario-level clustering ===", flush=True)
    print(f"  Scenarios with 0 worst rows: {(worst_per_sc == 0).sum()} / {n_sc}", flush=True)
    print(f"  Scenarios with 1-5 worst rows: {((worst_per_sc > 0) & (worst_per_sc <= 5)).sum()}", flush=True)
    print(f"  Scenarios with 6-15 worst rows: {((worst_per_sc > 5) & (worst_per_sc <= 15)).sum()}", flush=True)
    print(f"  Scenarios with 16-25 worst rows: {(worst_per_sc > 15).sum()}", flush=True)
    # Binomial null: P(row in worst) = 0.05. Expected per scenario: 1.25
    # Upper 95%CI: poisson ppf(0.99, 1.25) ≈ 4
    # So scenarios with 6+ worst rows are statistically concentrated
    # What fraction of all worst rows are in those concentrated scenarios?
    concentrated_mask = worst_per_sc >= 6  # suspicious scenarios
    n_conc = concentrated_mask.sum()
    rows_in_conc = worst_matrix[concentrated_mask].sum()
    print(f"  Concentrated scenarios (≥6 worst rows): {n_conc} ({n_conc/n_sc*100:.2f}%)", flush=True)
    print(f"    → {rows_in_conc} worst rows ({rows_in_conc/worst_mask.sum()*100:.1f}% of all worst)", flush=True)

    # Layout-level clustering
    layout_worst_rate = pd.Series(worst_mask, index=train["layout_id"]).groupby(level=0).mean()
    print(f"\n=== Layout-level clustering ===", flush=True)
    print(f"  Expected rate per layout: 5.00%", flush=True)
    print(f"  Mean rate per layout: {layout_worst_rate.mean()*100:.2f}%", flush=True)
    print(f"  Max rate: {layout_worst_rate.max()*100:.2f}%", flush=True)
    print(f"  Layouts with >15% worst rate: {(layout_worst_rate > 0.15).sum()} / {len(layout_worst_rate)}", flush=True)
    print(f"  Top 5 layouts by worst rate:", flush=True)
    top5 = layout_worst_rate.sort_values(ascending=False).head(5)
    for lid, rate in top5.items():
        print(f"    {lid}: {rate*100:.2f}%", flush=True)

    print("\n✓ Analysis complete. See feature_shift_ranking.csv for full list.", flush=True)


if __name__ == "__main__":
    main()
