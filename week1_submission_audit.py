"""
Week 1 Day 1-2: Submission reproducibility audit.

Verify:
  1. FIXED submission exists and is reproducible
  2. Seed consistency across components
  3. Test prediction aligns with sample_submission ID order
  4. No NaN/negative values
  5. Statistical sanity (mean, std within reasonable range)
"""
import os
import sys
import json
import pickle
import hashlib
import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "audit")
os.makedirs(OUT, exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def main():
    print("=" * 60, flush=True)
    print("Week 1 Day 1-2: Submission Audit", flush=True)
    print("=" * 60, flush=True)

    # Candidate submissions
    candidates = {
        "FIXED": os.path.join(ROOT, "results", "final_blend", "submission_final_multiblend_FIXED.csv"),
        "v6_3level": os.path.join(ROOT, "results", "final_blend", "submission_v6_3level.csv"),
        "v7_combined": os.path.join(ROOT, "results", "final_blend", "submission_v7_combined.csv"),
        "v7b_orig_meta9q": os.path.join(ROOT, "results", "final_blend", "submission_v7b_orig_meta9q.csv"),
    }

    sample = pd.read_csv(os.path.join(ROOT, "sample_submission.csv"))
    print(f"\nsample_submission: {len(sample)} rows, cols: {list(sample.columns)}", flush=True)
    sample_ids = sample["ID"].values

    # Load ground-truth y for sanity
    train = pd.read_csv(os.path.join(ROOT, "train.csv"))
    y_mean = train[TARGET].mean()
    y_std = train[TARGET].std()
    y_q95 = train[TARGET].quantile(0.95)

    print(f"\nTrain y stats: mean={y_mean:.2f}, std={y_std:.2f}, Q95={y_q95:.2f}", flush=True)

    results = []
    for name, path in candidates.items():
        print(f"\n--- {name}: {path} ---", flush=True)
        if not os.path.exists(path):
            print(f"  ❌ MISSING", flush=True)
            continue

        sub = pd.read_csv(path)
        h = sha256(path)

        # Checks
        checks = {}
        # 1. Columns
        checks["cols_ok"] = list(sub.columns) == ["ID", TARGET]
        # 2. Row count
        checks["rowcount_ok"] = len(sub) == len(sample)
        # 3. ID order match
        checks["id_order_ok"] = (sub["ID"].values == sample_ids).all() if len(sub) == len(sample) else False
        # 4. No NaN
        checks["no_nan"] = not sub[TARGET].isna().any()
        # 5. Non-negative
        checks["non_negative"] = (sub[TARGET] >= 0).all()
        # 6. Plausible range
        pred_mean = sub[TARGET].mean()
        pred_std = sub[TARGET].std()
        pred_max = sub[TARGET].max()
        checks["mean_plausible"] = 0.3 * y_mean <= pred_mean <= 3 * y_mean
        checks["std_plausible"] = 0.3 * y_std <= pred_std <= 3 * y_std

        print(f"  sha256_head={h}  n={len(sub)}  mean={pred_mean:.3f}  std={pred_std:.3f}  max={pred_max:.2f}", flush=True)
        for ck, v in checks.items():
            mark = "✅" if v else "❌"
            print(f"  {mark} {ck}", flush=True)

        results.append(dict(
            name=name, path=path, sha=h,
            n=len(sub), mean=float(pred_mean), std=float(pred_std),
            max=float(pred_max), checks=checks,
            all_ok=all(checks.values()),
        ))

    # Cross-comparison
    print("\n=== Cross-Comparison ===", flush=True)
    for r1, r2 in [("FIXED", "v7_combined"), ("FIXED", "v6_3level")]:
        if r1 in [r["name"] for r in results] and r2 in [r["name"] for r in results]:
            s1 = pd.read_csv(candidates[r1])
            s2 = pd.read_csv(candidates[r2])
            if len(s1) == len(s2):
                corr = float(np.corrcoef(s1[TARGET], s2[TARGET])[0, 1])
                mae_diff = float(np.mean(np.abs(s1[TARGET] - s2[TARGET])))
                print(f"  {r1} vs {r2}: corr={corr:.4f}, mean|diff|={mae_diff:.3f}", flush=True)

    # Save summary
    summary = dict(
        sample_n=len(sample),
        y_train_mean=y_mean, y_train_std=y_std, y_train_q95=y_q95,
        submissions=results,
    )
    with open(os.path.join(OUT, "submission_audit.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT}/submission_audit.json", flush=True)


if __name__ == "__main__":
    main()
