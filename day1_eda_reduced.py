"""
Day 1 Reduced EDA — 4 untried analyses.

Scope:
  1. y mod 1 histogram (discrete step hint)
  2. y find_peaks / cliff detection
  3. Scenario-internal autocorrelation (lag 1..24)
  4. Extreme tail (y > 50) conditional distribution

Kill gate:
  If 0-1 of 4 show meaningful signal -> abort Day 2 PySR, pivot.

Output:
  results/crack_eda/day1_findings.json
  results/crack_eda/y_mod_hist.png
  results/crack_eda/y_autocorr.png
  results/crack_eda/y_peaks.png
  results/crack_eda/day1_summary.md
"""
import os
import json
import pickle
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "crack_eda")
os.makedirs(OUT, exist_ok=True)


def analysis_1_y_mod(y):
    """y mod 1 — discrete step hint."""
    frac = y - np.floor(y)
    # Compare to uniform [0,1)
    hist, edges = np.histogram(frac, bins=100, range=(0.0, 1.0))
    uniform_count = len(y) / 100
    chi2 = float(np.sum((hist - uniform_count) ** 2 / uniform_count))
    # Top-5 bin concentrations
    top5 = np.argsort(hist)[::-1][:5]
    peaks = [(float(edges[i]), int(hist[i]), float(hist[i] / uniform_count)) for i in top5]

    # Also y mod 0.5, 0.25, 0.125 (in case step is subinteger)
    mod_patterns = {}
    for step in [1.0, 0.5, 0.25, 0.1, 0.01]:
        m = np.mod(y, step)
        # If systematically 0, all values are multiples of step
        zero_frac = float((np.abs(m) < 1e-6).mean())
        mod_patterns[str(step)] = dict(zero_frac=zero_frac)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(edges[:-1], hist, width=0.01, align="edge")
    ax.axhline(uniform_count, color="r", linestyle="--", label=f"uniform={uniform_count:.0f}")
    ax.set_title(f"y mod 1 histogram (chi2={chi2:.1f}, expected ~100 for uniform)")
    ax.set_xlabel("fractional part")
    ax.legend()
    fig.savefig(os.path.join(OUT, "y_mod_hist.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Signal criterion: chi2 > 500 (5x uniform variance) OR any step zero_frac > 0.5
    signal = bool(
        chi2 > 500
        or mod_patterns["0.5"]["zero_frac"] > 0.5
        or mod_patterns["0.25"]["zero_frac"] > 0.5
        or mod_patterns["0.1"]["zero_frac"] > 0.5
    )
    return dict(
        chi2_vs_uniform=chi2,
        uniform_expected=float(uniform_count),
        top5_bins=peaks,
        mod_zero_fracs=mod_patterns,
        signal=signal,
    )


def analysis_2_y_peaks(y):
    """Peak detection on y density for cliffs."""
    # Use fine-grained histogram
    hist, edges = np.histogram(y, bins=800, range=(0, float(np.quantile(y, 0.995))))
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Smooth lightly to suppress 1-bin noise
    w = 3
    kernel = np.ones(w) / w
    hist_smooth = np.convolve(hist, kernel, mode="same")

    # Peaks in density
    peaks, props = find_peaks(hist_smooth, height=np.percentile(hist_smooth, 90), distance=5)
    peak_locs = [(float(centers[i]), int(hist_smooth[i])) for i in peaks]

    # Cliffs: steep drops in cumulative. Look at diff of hist
    dhist = np.diff(hist_smooth)
    # Negative cliff = strong drop
    cliff_threshold = np.percentile(dhist, 1)  # 1%ile most negative
    cliffs_idx = np.where(dhist < cliff_threshold * 3)[0]
    cliffs = [(float(centers[i]), float(dhist[i])) for i in cliffs_idx[:5]]

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.plot(centers, hist_smooth, lw=1.0, label="smoothed density")
    for pl, _ in peak_locs[:10]:
        ax.axvline(pl, color="g", alpha=0.3)
    ax.set_title(
        f"y density peaks (n_peaks={len(peak_locs)}, top peaks green)"
    )
    ax.set_xlabel("y (minutes)")
    fig.savefig(os.path.join(OUT, "y_peaks.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Signal: >= 3 strong peaks OR a clear cliff (dhist < -500)
    signal = bool(len(peak_locs) >= 5 or len(cliffs) >= 2)
    return dict(
        n_peaks=len(peak_locs),
        top_peaks=peak_locs[:10],
        n_cliffs=len(cliffs),
        top_cliffs=cliffs,
        signal=signal,
    )


def analysis_3_autocorr(train, y):
    """Within-scenario autocorrelation lag 1..24."""
    # 10,000 scenarios, 25 rows each, sorted
    n_sc = len(train) // 25
    Y = y.reshape(n_sc, 25)

    # Mean-center per scenario
    Yc = Y - Y.mean(axis=1, keepdims=True)
    var = Yc.var(axis=1, keepdims=True)  # (n_sc, 1)
    var[var < 1e-9] = 1e-9

    acf = np.zeros(25)
    acf[0] = 1.0
    for L in range(1, 25):
        num = (Yc[:, :-L] * Yc[:, L:]).mean(axis=1, keepdims=True)
        rho = num / var
        acf[L] = float(rho.mean())

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.bar(np.arange(25), acf)
    ax.axhline(0, color="k", lw=0.5)
    ax.axhline(0.1, color="r", lw=0.5, linestyle="--", label="|0.1|")
    ax.axhline(-0.1, color="r", lw=0.5, linestyle="--")
    ax.set_title("Within-scenario y autocorrelation")
    ax.set_xlabel("lag")
    ax.set_ylabel("rho")
    ax.legend()
    fig.savefig(os.path.join(OUT, "y_autocorr.png"), dpi=100, bbox_inches="tight")
    plt.close(fig)

    # Signal: |rho| > 0.2 at any lag (strong temporal structure not modeled)
    max_abs_rho = float(np.max(np.abs(acf[1:])))
    signal = bool(max_abs_rho > 0.2)
    return dict(
        acf=[float(x) for x in acf],
        max_abs_rho_lag1_24=max_abs_rho,
        signal=signal,
    )


def analysis_4_extreme_tail(train, y, mega_oof):
    """Extreme tail (y > 50) conditional distribution vs mega33 residual pattern."""
    mask_tail = y > 50
    n_tail = int(mask_tail.sum())
    tail_frac = n_tail / len(y)

    r = y - mega_oof
    r_tail = r[mask_tail]
    r_body = r[~mask_tail]

    # Tail residual stats
    tail_stats = dict(
        n=n_tail,
        frac=float(tail_frac),
        r_mean=float(r_tail.mean()),
        r_median=float(np.median(r_tail)),
        r_std=float(r_tail.std()),
        r_p10=float(np.percentile(r_tail, 10)),
        r_p90=float(np.percentile(r_tail, 90)),
        mega_mean=float(mega_oof[mask_tail].mean()),
        y_mean=float(y[mask_tail].mean()),
    )
    body_stats = dict(
        n=len(r_body),
        r_mean=float(r_body.mean()),
        r_median=float(np.median(r_body)),
        r_std=float(r_body.std()),
    )

    # Is mega33 systematically under-predicting the tail?
    # Residual mean in tail = y_mean - mega_mean. If large positive, yes.
    bias_gap = tail_stats["r_mean"] - body_stats["r_mean"]

    # What fraction of tail is missed by mega33 (pred < 50 but truth > 50)?
    missed = float(((mega_oof < 50) & mask_tail).mean() * 100 / tail_frac) if tail_frac > 0 else 0
    # = among tail samples, fraction where mega33 pred < 50
    missed = float(((mega_oof[mask_tail] < 50)).mean())

    # Signal: bias_gap > 5 (major systematic underprediction NOT absorbed by mega)
    # AND missed > 0.5 (detection failure)
    signal = bool(bias_gap > 5 and missed > 0.5)

    return dict(
        tail=tail_stats,
        body=body_stats,
        bias_gap_tail_vs_body=float(bias_gap),
        tail_missed_rate=float(missed),
        signal=signal,
    )


def main():
    print("=" * 60)
    print("Day 1 Reduced EDA — crack hunt (4 analyses)")
    print("=" * 60)
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]

    print(f"y stats: mean={y.mean():.3f}, median={np.median(y):.3f}, std={y.std():.3f}")
    print(f"y min={y.min():.3f}, max={y.max():.3f}")
    print()

    print("[1/4] y mod 1 / step size detection ...")
    r1 = analysis_1_y_mod(y)
    print(f"      chi2 vs uniform = {r1['chi2_vs_uniform']:.1f} (expected ~100 if uniform)")
    print(f"      mod 0.5 zero_frac = {r1['mod_zero_fracs']['0.5']['zero_frac']:.4f}")
    print(f"      mod 0.1 zero_frac = {r1['mod_zero_fracs']['0.1']['zero_frac']:.4f}")
    print(f"      SIGNAL: {r1['signal']}")

    print("\n[2/4] y density peaks / cliffs ...")
    r2 = analysis_2_y_peaks(y)
    print(f"      n_peaks = {r2['n_peaks']}")
    print(f"      n_cliffs = {r2['n_cliffs']}")
    print(f"      top peaks: {r2['top_peaks'][:5]}")
    print(f"      SIGNAL: {r2['signal']}")

    print("\n[3/4] Within-scenario autocorrelation ...")
    r3 = analysis_3_autocorr(train, y)
    acf = r3["acf"]
    print(f"      max |rho| (lag 1..24) = {r3['max_abs_rho_lag1_24']:.4f}")
    print(f"      rho(lag=1..5) = {[f'{x:.3f}' for x in acf[1:6]]}")
    print(f"      rho(lag=6..12)= {[f'{x:.3f}' for x in acf[6:13]]}")
    print(f"      SIGNAL: {r3['signal']}")

    print("\n[4/4] Extreme tail conditional residual ...")
    r4 = analysis_4_extreme_tail(train, y, mega_oof)
    print(f"      tail n={r4['tail']['n']} ({r4['tail']['frac']*100:.2f}%)")
    print(f"      tail r_mean={r4['tail']['r_mean']:+.3f}  body r_mean={r4['body']['r_mean']:+.3f}")
    print(f"      bias_gap = {r4['bias_gap_tail_vs_body']:+.3f}")
    print(f"      tail_missed_rate (mega<50 | y>50) = {r4['tail_missed_rate']:.3f}")
    print(f"      SIGNAL: {r4['signal']}")

    findings = {
        "y_mod": r1,
        "y_peaks": r2,
        "autocorr": r3,
        "extreme_tail": r4,
    }
    signals = [r1["signal"], r2["signal"], r3["signal"], r4["signal"]]
    n_signal = int(sum(signals))
    print("\n" + "=" * 60)
    print(f"SIGNAL COUNT: {n_signal} / 4")
    if n_signal >= 2:
        verdict = "PROCEED_DAY2_EVALUATE"
        note = "2+ signals — Day 2 재평가 가치 있음. 단 PySR 전에 실험 4/14 결과 대조 필수."
    elif n_signal == 1:
        verdict = "MARGINAL_PIVOT"
        note = "1 signal only. Day 2 PySR 기대값 여전히 낮음. TabPFN/NODE 피벗 권장."
    else:
        verdict = "ABORT_PIVOT"
        note = "0 signals. Day 2 ~ Day 5 포기, 즉시 TabPFN/NODE 또는 현 제출 유지."
    print(f"VERDICT: {verdict}")
    print(f"NOTE: {note}")
    print("=" * 60)

    findings["verdict"] = verdict
    findings["note"] = note
    findings["n_signal"] = n_signal

    with open(os.path.join(OUT, "day1_findings.json"), "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2, ensure_ascii=False)

    # Also write a short markdown summary
    lines = [
        "# Day 1 Reduced EDA — Findings",
        "",
        f"- y stats: mean={y.mean():.3f}, median={np.median(y):.3f}, std={y.std():.3f}, max={y.max():.3f}",
        "",
        "## 1. y mod 1",
        f"- chi2 vs uniform: {r1['chi2_vs_uniform']:.1f} (uniform ~ 100 per bin)",
        f"- mod 0.5 zero_frac: {r1['mod_zero_fracs']['0.5']['zero_frac']:.4f}",
        f"- mod 0.1 zero_frac: {r1['mod_zero_fracs']['0.1']['zero_frac']:.4f}",
        f"- SIGNAL: **{r1['signal']}**",
        "",
        "## 2. y density peaks / cliffs",
        f"- n_peaks: {r2['n_peaks']}, n_cliffs: {r2['n_cliffs']}",
        f"- top peaks (value, count): {r2['top_peaks'][:5]}",
        f"- SIGNAL: **{r2['signal']}**",
        "",
        "## 3. Within-scenario autocorrelation",
        f"- max |rho| (lag 1..24): {r3['max_abs_rho_lag1_24']:.4f}",
        f"- rho(1..5): {[f'{x:.3f}' for x in acf[1:6]]}",
        f"- SIGNAL: **{r3['signal']}**",
        "",
        "## 4. Extreme tail (y > 50)",
        f"- tail n={r4['tail']['n']} ({r4['tail']['frac']*100:.2f}%)",
        f"- bias_gap (tail r_mean - body r_mean): {r4['bias_gap_tail_vs_body']:+.3f}",
        f"- tail_missed_rate: {r4['tail_missed_rate']:.3f}",
        f"- SIGNAL: **{r4['signal']}**",
        "",
        f"## VERDICT: {verdict}",
        f"> {note}",
    ]
    with open(os.path.join(OUT, "day1_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
