"""
Experiment B - Phase 1: Bucket Diagnosis (30-min cap)

Goal: Find worst bucket (layout_type x ts_bucket) in mega33 OOF.
Kill gate: any worst bucket row share < 1% -> abort experiment B.

Output:
- results/bucket_specialist/bucket_mae.csv
- results/bucket_specialist/diagnosis_summary.txt
- worst3 buckets printed to stdout
"""
import os
import pickle
import numpy as np
import pandas as pd

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT_DIR = os.path.join(ROOT, "results", "bucket_specialist")
os.makedirs(OUT_DIR, exist_ok=True)


def main():
    print("=" * 60)
    print("Experiment B Phase 1: Bucket Diagnosis")
    print("=" * 60)

    # Load train, sort (MANDATORY alignment with mega33_oof)
    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values
    print(f"train rows: {len(train)}")

    # Load mega33 OOF
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    oof = mega["meta_avg_oof"]
    assert len(oof) == len(train), f"mega33 OOF length {len(oof)} != train {len(train)}"
    print(f"mega33 OOF loaded. shape={oof.shape}")

    # Sanity: overall MAE
    overall_mae = np.mean(np.abs(y - oof))
    print(f"mega33 overall OOF MAE = {overall_mae:.4f}")
    # Expected ~8.3989 per STUDY_SUMMARY

    # Build timeslot index within scenario (25 rows per scenario, already sorted)
    # After sort by layout_id, scenario_id, within-scenario row order = ts_idx 0..24
    ts_idx = np.tile(np.arange(25), len(train) // 25)
    assert len(ts_idx) == len(train), "ts_idx length mismatch"
    train["_ts_idx"] = ts_idx
    # Bucket: early(0-7), mid(8-15), late(16-24)
    train["_ts_bucket"] = pd.cut(
        train["_ts_idx"],
        bins=[-1, 7, 15, 24],
        labels=["early", "mid", "late"],
    )

    # Load layout_type
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    print(f"layout_info cols: {list(layout_info.columns)[:8]}...")
    lt_col = "layout_type"
    assert lt_col in layout_info.columns, f"layout_type not found"
    train = train.merge(layout_info[["layout_id", lt_col]], on="layout_id", how="left")
    print(f"layout_type null rows: {train[lt_col].isna().sum()}")
    print(f"layout_type values: {train[lt_col].unique()}")

    # Compute bucket MAE
    train["_y"] = y
    train["_pred"] = oof
    train["_abs_err"] = np.abs(y - oof)

    grp = (
        train.groupby([lt_col, "_ts_bucket"], observed=True)
        .agg(
            bucket_mae=("_abs_err", "mean"),
            n_rows=("_abs_err", "size"),
            y_mean=("_y", "mean"),
            y_std=("_y", "std"),
        )
        .reset_index()
        .sort_values("bucket_mae", ascending=False)
    )
    grp["row_share_pct"] = grp["n_rows"] / len(train) * 100
    print("\nBucket MAE (sorted worst to best):")
    print(grp.to_string(index=False))

    # Save
    grp.to_csv(os.path.join(OUT_DIR, "bucket_mae.csv"), index=False)
    print(f"\nSaved bucket_mae.csv to {OUT_DIR}")

    # Worst 3
    worst3 = grp.head(3).reset_index(drop=True)
    print("\nWorst 3 buckets:")
    print(worst3.to_string(index=False))

    # Kill gate: if top worst bucket row share < 1% -> abort
    top_share = worst3.iloc[0]["row_share_pct"]
    abort = bool(top_share < 1.0)

    summary = []
    summary.append(f"Overall mega33 OOF MAE: {overall_mae:.4f}")
    summary.append(f"Top worst bucket: {worst3.iloc[0][lt_col]} x {worst3.iloc[0]['_ts_bucket']}")
    summary.append(
        f"  - bucket MAE: {worst3.iloc[0]['bucket_mae']:.4f}, row share: {top_share:.2f}%"
    )
    summary.append("")
    summary.append("Worst 3:")
    for i, r in worst3.iterrows():
        summary.append(
            f"  {i+1}. {r[lt_col]} x {r['_ts_bucket']}: MAE {r['bucket_mae']:.4f}, "
            f"share {r['row_share_pct']:.2f}%"
        )
    summary.append("")
    summary.append(f"Kill gate (share < 1%): {'ABORT' if abort else 'PASS'}")
    with open(os.path.join(OUT_DIR, "diagnosis_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print("\n" + "\n".join(summary))

    if abort:
        print("\n[ABORT] Top worst bucket row share < 1%. Experiment B stopped.")
    else:
        print("\n[PASS] Proceed to Phase 2-a.")


if __name__ == "__main__":
    main()
