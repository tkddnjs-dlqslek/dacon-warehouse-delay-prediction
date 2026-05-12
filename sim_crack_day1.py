"""
Simulator Crack Day 1 — y self micro-structure hunt.

Scope:
  A. y distribution micro-observation (multi-resolution histograms, log1p, zero-spike, small-y zoom)
  B. Layout-specific stats (250 train layouts) — anomaly detection
  C. Scenario-level structure (25 ts sequence — step/jump patterns)
  D. 2D cell bimodality (congestion x util, robot_active x order_inflow) — mixture hint
  E. LOWESS top-20 features only (by mega33 importance, not all 90)

Output: results/sim_crack/*
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.stats import skew

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "sim_crack")
os.makedirs(OUT, exist_ok=True)


def section_A(y):
    """y distribution micro-observation."""
    print("\n[A] y distribution micro-observation")
    results = {}

    # zero spike
    n_zero = int((y == 0).sum())
    results["zero_count"] = n_zero
    results["zero_frac"] = float(n_zero / len(y))
    # very-small y (e.g., y < 0.1)
    tiny_count = int((y < 0.1).sum())
    nonzero_tiny = int(((y > 0) & (y < 0.1)).sum())
    results["tiny_count"] = tiny_count
    results["nonzero_tiny_count"] = nonzero_tiny
    print(f"  y == 0: {n_zero} ({n_zero/len(y)*100:.2f}%)")
    print(f"  0 < y < 0.1: {nonzero_tiny}")

    # Multi-resolution histograms
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    y_trunc = y[y < np.quantile(y, 0.99)]  # cap at Q99 for visibility
    for ax, bins in zip(axes.flat[:3], [1000, 5000, 10000]):
        ax.hist(y_trunc, bins=bins, edgecolor="none")
        ax.set_title(f"y histogram (0..Q99, {bins} bins)")
        ax.set_xlabel("y")
    # small y zoom
    ax = axes.flat[3]
    ax.hist(y[y < 5], bins=500)
    ax.set_title("y in [0, 5]")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "A_y_hist.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    # log1p(y)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(np.log1p(y), bins=400)
    ax.set_title("log1p(y) histogram")
    fig.savefig(os.path.join(OUT, "A_log1p_hist.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    # Peak detection on 5000-bin hist
    hist, edges = np.histogram(y_trunc, bins=5000)
    centers = 0.5 * (edges[:-1] + edges[1:])
    peaks, props = find_peaks(hist, height=len(y) * 0.0002, distance=10, prominence=len(y) * 0.0001)
    results["n_peaks_5000bin"] = int(len(peaks))
    results["top_peaks"] = [
        (float(centers[i]), int(hist[i])) for i in np.argsort(hist[peaks])[-10:][::-1]
    ] if len(peaks) > 0 else []
    print(f"  peaks (5000 bin, prom>{len(y)*0.0001:.0f}): {len(peaks)}")
    if len(peaks) > 0:
        print(f"  top 5 peak locations: {results['top_peaks'][:5]}")

    return results


def section_B(train, y):
    """Layout-specific."""
    print("\n[B] Layout-specific analysis")
    df = pd.DataFrame({"layout_id": train["layout_id"].values, "y": y})
    lo_stats = (
        df.groupby("layout_id")["y"]
        .agg(["count", "mean", "std", "min", "max", "median"])
        .reset_index()
    )
    lo_stats.columns = ["layout_id", "n", "y_mean", "y_std", "y_min", "y_max", "y_median"]

    # Attach layout_type
    li = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    lo_stats = lo_stats.merge(li[["layout_id", "layout_type"]], on="layout_id")

    print(f"  layouts: {len(lo_stats)}  unique types: {lo_stats['layout_type'].value_counts().to_dict()}")
    for t in lo_stats["layout_type"].unique():
        sub = lo_stats[lo_stats["layout_type"] == t]
        print(
            f"  type {t:<10}: y_mean range [{sub['y_mean'].min():.2f}, {sub['y_mean'].max():.2f}], "
            f"y_max range [{sub['y_max'].min():.2f}, {sub['y_max'].max():.2f}]"
        )

    # Anomaly detection: within each type, z-score of y_mean
    lo_stats["z_mean_within_type"] = lo_stats.groupby("layout_type")["y_mean"].transform(
        lambda s: (s - s.mean()) / s.std()
    )
    lo_stats["z_max_within_type"] = lo_stats.groupby("layout_type")["y_max"].transform(
        lambda s: (s - s.mean()) / s.std()
    )
    anomaly = lo_stats[
        (lo_stats["z_mean_within_type"].abs() > 3) | (lo_stats["z_max_within_type"].abs() > 3)
    ].sort_values("z_mean_within_type", key=lambda s: s.abs(), ascending=False)
    print(f"  anomaly layouts (|z| > 3 in mean or max): {len(anomaly)}")
    if len(anomaly) > 0:
        print(anomaly[["layout_id", "layout_type", "y_mean", "y_max", "z_mean_within_type"]].head(10).to_string(index=False))

    lo_stats.to_csv(os.path.join(OUT, "B_layout_stats.csv"), index=False)
    anomaly.to_csv(os.path.join(OUT, "B_layout_anomalies.csv"), index=False)

    # Histogram of y_mean per layout, colored by type
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for t in lo_stats["layout_type"].unique():
        sub = lo_stats[lo_stats["layout_type"] == t]
        ax.hist(sub["y_mean"], bins=40, alpha=0.5, label=t)
    ax.set_title("y_mean per layout by type")
    ax.set_xlabel("y_mean")
    ax.legend()
    fig.savefig(os.path.join(OUT, "B_layout_ymean.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    return dict(
        n_layouts=int(len(lo_stats)),
        n_anomalies=int(len(anomaly)),
        type_means={t: float(lo_stats[lo_stats.layout_type == t]["y_mean"].mean()) for t in lo_stats["layout_type"].unique()},
    )


def section_C(y):
    """Scenario-level structure."""
    print("\n[C] Scenario-level structure")
    n_sc = len(y) // 25
    Y = y.reshape(n_sc, 25)

    # Shape statistics
    sc_max = Y.max(axis=1)
    sc_mean = Y.mean(axis=1)
    sc_range = sc_max - Y.min(axis=1)
    sc_std = Y.std(axis=1)
    sc_mono_up = (np.diff(Y, axis=1) > 0).mean(axis=1)  # fraction of positive diffs
    sc_argmax = Y.argmax(axis=1)

    print(f"  n_scenarios: {n_sc}")
    print(f"  y_argmax distribution: early(0-7)={int((sc_argmax<=7).sum())} mid(8-15)={int(((sc_argmax>7)&(sc_argmax<=15)).sum())} late(16-24)={int((sc_argmax>=16).sum())}")
    print(f"  monotonic up (>80%): {int((sc_mono_up > 0.8).sum())} scenarios ({(sc_mono_up>0.8).mean()*100:.1f}%)")
    print(f"  monotonic down (<20%): {int((sc_mono_up < 0.2).sum())} scenarios ({(sc_mono_up<0.2).mean()*100:.1f}%)")

    # Mean trajectory with quartiles
    ts_mean = Y.mean(axis=0)
    ts_q25 = np.quantile(Y, 0.25, axis=0)
    ts_q75 = np.quantile(Y, 0.75, axis=0)
    ts_q90 = np.quantile(Y, 0.90, axis=0)

    # Step-function detection: look at median absolute diff per scenario
    # If generation is step-like, sc_std should have bimodal distribution (flat vs stepped)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(ts_mean, label="mean")
    axes[0, 0].plot(ts_q25, label="Q25", alpha=0.5)
    axes[0, 0].plot(ts_q75, label="Q75", alpha=0.5)
    axes[0, 0].plot(ts_q90, label="Q90", alpha=0.5)
    axes[0, 0].set_title("Cross-scenario y trajectory")
    axes[0, 0].legend()

    # Sample 100 scenarios
    np.random.seed(0)
    samples = np.random.choice(n_sc, 100, replace=False)
    for i in samples:
        axes[0, 1].plot(Y[i], alpha=0.15, color="blue")
    axes[0, 1].set_title("100 random scenario trajectories")

    axes[1, 0].hist(sc_mono_up, bins=50)
    axes[1, 0].set_title("per-scenario monotonic-up fraction")

    axes[1, 1].hist(sc_std, bins=100, range=(0, 50))
    axes[1, 1].set_title("per-scenario std")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "C_scenario.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    return dict(
        n_scenarios=n_sc,
        mono_up_frac=float((sc_mono_up > 0.8).mean()),
        argmax_late_frac=float((sc_argmax >= 16).mean()),
    )


def section_D(train_fe, y):
    """2D cell bimodality detection."""
    print("\n[D] 2D cell bimodality (mixture hint)")
    results = {"bimodal_cells": []}

    # Focus on high-value features (congestion, utilization, order inflow, robot_active)
    pairs = [
        ("congestion_score", "robot_utilization"),
        ("order_inflow_15m", "robot_active"),
        ("congestion_score", "order_inflow_15m"),
        ("blocked_path_15m", "robot_utilization"),
    ]
    for f1, f2 in pairs:
        if f1 not in train_fe.columns or f2 not in train_fe.columns:
            continue
        x1 = train_fe[f1].values
        x2 = train_fe[f2].values
        # Drop nan rows for these columns
        mask = np.isfinite(x1) & np.isfinite(x2)
        if mask.sum() < 1000:
            continue
        x1c = x1[mask]
        x2c = x2[mask]
        yc = y[mask]

        # Quantile bins per feature
        b1 = pd.qcut(x1c, 5, labels=False, duplicates="drop")
        b2 = pd.qcut(x2c, 5, labels=False, duplicates="drop")

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        # Heatmap of y_mean per cell
        grid_mean = np.zeros((5, 5))
        grid_n = np.zeros((5, 5), dtype=int)
        cell_bimodal = []
        for i in range(5):
            for j in range(5):
                m = (b1 == i) & (b2 == j)
                n = int(m.sum())
                grid_n[i, j] = n
                if n > 0:
                    grid_mean[i, j] = yc[m].mean()
                # Bimodality check: Hartigan-like via histogram
                if n > 100:
                    yc_cell = yc[m]
                    # Simple bimodality metric: does yc_cell look bimodal?
                    # Use: sort, find gap relative to std
                    sorted_y = np.sort(yc_cell)
                    # Compute kernel density peaks
                    hist, _ = np.histogram(yc_cell, bins=50)
                    p_idx, _ = find_peaks(hist, prominence=n * 0.03)
                    if len(p_idx) >= 2:
                        cell_bimodal.append((i, j, int(len(p_idx)), int(n), float(yc_cell.mean())))

        im = ax.imshow(grid_mean, cmap="viridis", aspect="auto")
        ax.set_title(f"mean y: {f1} x {f2}")
        ax.set_xlabel(f"{f2} bin")
        ax.set_ylabel(f"{f1} bin")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"D_{f1}_x_{f2}.png"), dpi=90, bbox_inches="tight")
        plt.close(fig)

        print(f"  {f1} x {f2}: {len(cell_bimodal)} bimodal cells (>=2 peaks)")
        if cell_bimodal:
            for c in cell_bimodal[:5]:
                print(f"    cell ({c[0]},{c[1]}): peaks={c[2]} n={c[3]} y_mean={c[4]:.2f}")
            results["bimodal_cells"].extend([
                dict(f1=f1, f2=f2, bin1=c[0], bin2=c[1], peaks=c[2], n=c[3], y_mean=c[4])
                for c in cell_bimodal
            ])

    return results


def section_E(train_fe, y, feat_cols):
    """LOWESS top 20 features."""
    print("\n[E] LOWESS top-20 features (skipping 70 redundant)")
    # Use simple correlation ranking (proxy for mega33 importance)
    # Compute |rho| between each feature and y
    imp = []
    for c in feat_cols:
        x = train_fe[c].values
        m = np.isfinite(x)
        if m.sum() < 1000:
            continue
        xv = x[m]
        yv = y[m]
        rho = float(np.corrcoef(xv, yv)[0, 1]) if xv.std() > 0 else 0.0
        imp.append((c, abs(rho), rho))
    imp.sort(key=lambda r: -r[1])
    top20 = imp[:20]
    print(f"  top-20 features by |rho| with y:")
    for c, absrho, rho in top20:
        print(f"    {c:<40} rho={rho:+.4f}")

    # Plot scatter + moving median (LOWESS-like) for top 8 only
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for k, (c, _, _) in enumerate(top20[:8]):
        ax = axes.flat[k]
        x = train_fe[c].values
        m = np.isfinite(x)
        xv, yv = x[m], y[m]
        # Subsample for speed
        idx = np.random.default_rng(0).choice(len(xv), size=min(10000, len(xv)), replace=False)
        ax.scatter(xv[idx], yv[idx], s=1, alpha=0.2)
        # Moving median
        order = np.argsort(xv[idx])
        xs = xv[idx][order]
        ys = yv[idx][order]
        w = 500
        if len(xs) > w:
            med = np.array([np.median(ys[i : i + w]) for i in range(0, len(xs) - w, 50)])
            xmed = np.array([xs[i + w // 2] for i in range(0, len(xs) - w, 50)])
            ax.plot(xmed, med, color="red", lw=1.5, label="rolling median")
        ax.set_title(f"{c[:30]}")
        ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "E_lowess_top8.png"), dpi=90, bbox_inches="tight")
    plt.close(fig)

    return dict(
        top20=[dict(feat=c, abs_rho=ar, rho=r) for c, ar, r in top20],
    )


def main():
    print("=" * 60)
    print("Simulator Crack Day 1 — y self micro-structure")
    print("=" * 60)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]

    findings = {}
    findings["A"] = section_A(y)
    findings["B"] = section_B(train, y)
    findings["C"] = section_C(y)
    findings["D"] = section_D(train_fe, y)
    findings["E"] = section_E(train_fe, y, feat_cols)

    with open(os.path.join(OUT, "findings.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    # GO/NO-GO assessment
    signals = {
        "zero_spike": findings["A"]["zero_frac"] > 0.02,
        "peaks_in_y": findings["A"]["n_peaks_5000bin"] > 5,
        "layout_anomalies": findings["B"]["n_anomalies"] > 3,
        "bimodal_cells": len(findings["D"]["bimodal_cells"]) > 0,
    }
    print("\n" + "=" * 60)
    print("Signal summary:")
    for k, v in signals.items():
        print(f"  {k}: {v}")
    n_sig = sum(signals.values())
    verdict = "GO_DAY2_PYSR" if n_sig >= 2 else "NO_GO"
    print(f"\nSignal count: {n_sig} / 4")
    print(f"VERDICT: {verdict}")
    findings["signals"] = signals
    findings["n_signal"] = n_sig
    findings["verdict"] = verdict
    with open(os.path.join(OUT, "findings.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    # pattern_candidates.md (short)
    lines = [
        "# Simulator Crack Day 1 — Pattern Candidates",
        "",
        f"**Verdict: {verdict} ({n_sig}/4 signals)**",
        "",
        "## Signals detected",
    ]
    for k, v in signals.items():
        mark = "✅" if v else "❌"
        lines.append(f"- {mark} `{k}`")
    lines.append("")
    lines.append("## Highlights")
    lines.append(f"- y==0 spike: {findings['A']['zero_count']} rows ({findings['A']['zero_frac']*100:.2f}%)")
    lines.append(f"- 5000-bin peaks: {findings['A']['n_peaks_5000bin']}")
    lines.append(f"- Layout anomalies (|z|>3): {findings['B']['n_anomalies']}")
    lines.append(f"- Bimodal cells across 4 pair grids: {len(findings['D']['bimodal_cells'])}")
    lines.append("")
    lines.append("## Interpretation")
    if verdict == "GO_DAY2_PYSR":
        lines.append("- 2+ signals found. Day 2 PySR may be worth 1 symbolic regression pass on the strongest pattern.")
        lines.append("- Recommended PySR target: conditional y on the top bimodal cell.")
    else:
        lines.append("- Insufficient signals. Day 2 PySR unlikely to find exploitable formula.")
        lines.append("- 22+ prior experiments (esp. queuing 14, Q-sweep 33) already ruled out formula-space recovery.")
    with open(os.path.join(OUT, "pattern_candidates.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
