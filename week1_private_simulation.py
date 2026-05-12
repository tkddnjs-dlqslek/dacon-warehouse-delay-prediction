"""
Week 1 Day 5-6: Private LB shake-up simulation.

Assumption: Public LB = 30% of test, Private = 70%.
Use OOF residuals as proxy for unknown test residuals.
Bootstrap: resample test rows with different Public/Private splits to estimate
variance in Public LB vs Private LB for each candidate submission.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "private_sim")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Week 1 Day 5-6: Private LB Shake-up Simulation", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)

    # Candidate OOFs (use OOF as proxy for test distribution)
    candidates = {}

    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    candidates["FIXED_mega33"] = mega["meta_avg_oof"]

    # v6 (3-level)
    p = os.path.join(ROOT, "results", "task_c", "meta_avg_oof_3level.npy")
    if os.path.exists(p):
        candidates["v6_3level_meta"] = np.load(p)

    # Simulated FIXED full blend from known component weights
    # FIXED: mega33 0.77, rank 0.16, iter_r1/r2/r3 0.07
    rank = np.load(os.path.join(ROOT, "results", "ranking", "rank_adj_oof.npy"))
    iter_r1 = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round1_oof.npy"))
    iter_r2 = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round2_oof.npy"))
    iter_r3 = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round3_oof.npy"))

    fixed_oof = 0.7637 * mega["meta_avg_oof"] + 0.1589 * rank + 0.0119 * iter_r1 + 0.0346 * iter_r2 + 0.0310 * iter_r3
    candidates["FIXED_full_blend"] = fixed_oof

    # v7 combined: mega_3level 0.7099 + meta9q 0.1368 + rank 0.1534
    meta9q = np.load(os.path.join(ROOT, "results", "task_2_joint", "meta_oof.npy"))
    if os.path.exists(os.path.join(ROOT, "results", "task_c", "meta_avg_oof_3level.npy")):
        mega_3l = np.load(os.path.join(ROOT, "results", "task_c", "meta_avg_oof_3level.npy"))
        v7_oof = 0.7099 * mega_3l + 0.1368 * meta9q + 0.1534 * rank
        candidates["v7_combined"] = v7_oof

    print(f"\nCandidates loaded: {list(candidates.keys())}", flush=True)
    for name, oof in candidates.items():
        mae = mean_absolute_error(y, oof)
        print(f"  {name:<25s} OOF MAE = {mae:.5f}", flush=True)

    # ─── Bootstrap simulation ───
    # Assume test is 50000 rows. Public = 30% = 15000, Private = 70% = 35000.
    # Our OOF is 250000 rows. We'll bootstrap 50000-row "mock test" subsets
    # then split into public/private.
    print("\n=== Bootstrap simulation (n=1000 trials) ===", flush=True)
    n_trials = 1000
    n_test = 50000
    n_public = int(n_test * 0.3)
    n_private = n_test - n_public

    rng = np.random.default_rng(42)
    results = {name: dict(public=[], private=[]) for name in candidates}

    for trial in range(n_trials):
        # Sample 50k rows from 250k OOF as "mock test"
        # Better: respect layout structure — sample by layout group
        mock_test_idx = rng.choice(len(y), size=n_test, replace=False)
        # Split into public/private
        public_idx = mock_test_idx[:n_public]
        private_idx = mock_test_idx[n_public:]

        for name, oof in candidates.items():
            mae_pub = mean_absolute_error(y[public_idx], oof[public_idx])
            mae_priv = mean_absolute_error(y[private_idx], oof[private_idx])
            results[name]["public"].append(mae_pub)
            results[name]["private"].append(mae_priv)

    # ─── Analysis ───
    print("\n=== Shake-up statistics per candidate ===", flush=True)
    summary = {}
    names = list(candidates.keys())

    for name in names:
        pubs = np.array(results[name]["public"])
        privs = np.array(results[name]["private"])
        rank_priv_better_count = (privs < pubs).sum()

        summary[name] = dict(
            public_mean=float(pubs.mean()),
            public_std=float(pubs.std()),
            private_mean=float(privs.mean()),
            private_std=float(privs.std()),
            priv_better_than_pub_pct=float((privs < pubs).mean() * 100),
            priv_minus_pub_mean=float((privs - pubs).mean()),
        )
        print(f"\n  {name}:", flush=True)
        print(f"    Public   MAE: {pubs.mean():.5f} ± {pubs.std():.5f}", flush=True)
        print(f"    Private  MAE: {privs.mean():.5f} ± {privs.std():.5f}", flush=True)
        print(f"    P(Private < Public): {(privs < pubs).mean()*100:.1f}%", flush=True)
        print(f"    Mean (Priv - Pub): {(privs - pubs).mean():+.5f}", flush=True)

    # ─── Relative ranking: FIXED vs v7 ───
    if "FIXED_full_blend" in candidates and "v7_combined" in candidates:
        print("\n=== FIXED vs v7 relative performance (bootstrap) ===", flush=True)
        fixed_pubs = np.array(results["FIXED_full_blend"]["public"])
        v7_pubs = np.array(results["v7_combined"]["public"])
        fixed_privs = np.array(results["FIXED_full_blend"]["private"])
        v7_privs = np.array(results["v7_combined"]["private"])

        # How often v7 wins Public? Private?
        v7_wins_public = (v7_pubs < fixed_pubs).mean()
        v7_wins_private = (v7_privs < fixed_privs).mean()
        print(f"  v7 wins Public: {v7_wins_public*100:.1f}%", flush=True)
        print(f"  v7 wins Private: {v7_wins_private*100:.1f}%", flush=True)
        print(f"  Mean Public gap (v7 - FIXED): {(v7_pubs - fixed_pubs).mean():+.5f}", flush=True)
        print(f"  Mean Private gap (v7 - FIXED): {(v7_privs - fixed_privs).mean():+.5f}", flush=True)

        # Real LB data
        print(f"\n  Real LB FIXED: 9.7711, v7: 9.7740, diff: +0.0029", flush=True)
        print(f"  Bootstrap prediction Private gap: {(v7_privs - fixed_privs).mean():+.5f}", flush=True)

        summary["v7_vs_FIXED"] = dict(
            v7_wins_public_pct=float(v7_wins_public * 100),
            v7_wins_private_pct=float(v7_wins_private * 100),
            mean_public_gap=float((v7_pubs - fixed_pubs).mean()),
            mean_private_gap=float((v7_privs - fixed_privs).mean()),
        )

    with open(os.path.join(OUT, "bootstrap_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT}/bootstrap_summary.json", flush=True)


if __name__ == "__main__":
    main()
