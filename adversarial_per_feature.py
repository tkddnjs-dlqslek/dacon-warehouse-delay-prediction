"""
Per-feature adversarial diagnostic.

For each of 149 v23 features:
  1. KS statistic between train and test marginal distribution
  2. LGB adversarial AUC (single-feature classifier)
  3. Static/dynamic classification (unique values per layout)
  4. Correlation with y (to know if removal hurts predictive power)

Output:
  results/adv_per_feat/diagnosis.csv  (feature, ks, auc_single, is_static, n_unique_per_layout, rho_y, action)
  results/adv_per_feat/summary.md
"""
import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "adv_per_feat")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60)
    print("Per-feature adversarial diagnostic (149 v23 features)")
    print("=" * 60)

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = d["feat_cols"]
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)

    # y for train (for rho check)
    train_csv = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train_csv["avg_delay_minutes_next_30m"].values

    print(f"train_fe: {train_fe.shape}  test_fe: {test_fe.shape}")
    print(f"feat_cols: {len(feat_cols)}")

    results = []
    # Pre-check static-ness: for each feature, how many unique values per layout?
    print("\nScanning static-ness per feature...")
    # Group train by layout_id
    for i, c in enumerate(feat_cols):
        if c not in train_fe.columns or c not in test_fe.columns:
            continue
        xt = train_fe[c].values
        xv = test_fe[c].values
        # filter nan
        mt = np.isfinite(xt)
        mv = np.isfinite(xv)
        xt_f = xt[mt]
        xv_f = xv[mv]

        # KS
        if len(xt_f) < 100 or len(xv_f) < 100:
            ks_stat, ks_p = np.nan, np.nan
        else:
            ks_stat, ks_p = ks_2samp(xt_f, xv_f)

        # Static detection: if per-layout mean std is ~0, static
        # Quick proxy: global std vs std of (row-level) - group-mean
        # Simpler: check if feature is constant within each layout by sampling
        layouts = train_fe["layout_id"].values
        grp_stds = train_fe.groupby("layout_id")[c].std(ddof=0)
        grp_stds_nonan = grp_stds.dropna()
        is_static = bool((grp_stds_nonan.mean() < 1e-6)) if len(grp_stds_nonan) > 0 else False

        # Per-feature adversarial AUC (LGB, shallow, fast)
        if mt.sum() < 100 or mv.sum() < 100:
            auc = np.nan
        else:
            # Build balanced-ish dataset: use ALL train + ALL test (train ~5x larger)
            X_adv = np.concatenate([xt_f, xv_f]).reshape(-1, 1)
            y_adv = np.concatenate([np.zeros(len(xt_f)), np.ones(len(xv_f))]).astype(np.int32)
            try:
                m = lgb.LGBMClassifier(
                    n_estimators=80,
                    learning_rate=0.1,
                    num_leaves=15,
                    min_child_samples=100,
                    verbose=-1,
                ).fit(X_adv, y_adv)
                p = m.predict_proba(X_adv)[:, 1]
                auc = float(roc_auc_score(y_adv, p))
            except Exception as e:
                auc = np.nan

        # rho with y on train
        mt_ally = np.isfinite(xt) & np.isfinite(y)
        if mt_ally.sum() > 100 and xt[mt_ally].std() > 0:
            rho = float(np.corrcoef(xt[mt_ally], y[mt_ally])[0, 1])
        else:
            rho = np.nan

        results.append(dict(
            feature=c,
            ks_stat=float(ks_stat) if np.isfinite(ks_stat) else None,
            adv_auc=float(auc) if np.isfinite(auc) else None,
            is_static=is_static,
            rho_y=rho,
        ))
        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(feat_cols)}] done")

    df = pd.DataFrame(results).sort_values("adv_auc", ascending=False, na_position="last")
    df.to_csv(os.path.join(OUT, "diagnosis.csv"), index=False)
    print(f"\nSaved to {OUT}/diagnosis.csv")

    # Summary
    print("\n=== Summary ===")
    n_total = len(df)
    n_static = int(df["is_static"].sum())
    n_dynamic = n_total - n_static
    print(f"total features: {n_total}")
    print(f"static (constant per layout): {n_static}")
    print(f"dynamic: {n_dynamic}")

    # Shift breakdown
    def pct(mask):
        return f"{int(mask.sum()):>3} ({mask.mean()*100:5.1f}%)"

    print("\nAdversarial AUC distribution (ALL features):")
    print(f"  AUC >= 0.9: {pct(df['adv_auc'] >= 0.9)}")
    print(f"  AUC >= 0.8: {pct((df['adv_auc'] >= 0.8) & (df['adv_auc'] < 0.9))}")
    print(f"  AUC >= 0.7: {pct((df['adv_auc'] >= 0.7) & (df['adv_auc'] < 0.8))}")
    print(f"  AUC >= 0.6: {pct((df['adv_auc'] >= 0.6) & (df['adv_auc'] < 0.7))}")
    print(f"  AUC <  0.6: {pct(df['adv_auc'] < 0.6)}")

    print("\n=== AUC > 0.7 in DYNAMIC features only (actionable) ===")
    dyn_high = df[(~df["is_static"]) & (df["adv_auc"] >= 0.7)].sort_values("adv_auc", ascending=False)
    print(f"count: {len(dyn_high)}")
    if len(dyn_high) > 0:
        print(dyn_high.head(20).to_string(index=False))

    print("\n=== AUC > 0.7 in STATIC features (explains AUC 0.9994) ===")
    stat_high = df[(df["is_static"]) & (df["adv_auc"] >= 0.7)].sort_values("adv_auc", ascending=False)
    print(f"count: {len(stat_high)}")
    if len(stat_high) > 0:
        print(stat_high.head(10).to_string(index=False))

    # Summary markdown
    lines = [
        "# Per-feature Adversarial Diagnosis",
        "",
        f"- Total features: {n_total}",
        f"- Static (constant per layout): {n_static}",
        f"- Dynamic: {n_dynamic}",
        "",
        "## Adversarial AUC distribution",
        f"- AUC >= 0.9: {int((df['adv_auc']>=0.9).sum())}",
        f"- AUC >= 0.8: {int(((df['adv_auc']>=0.8)&(df['adv_auc']<0.9)).sum())}",
        f"- AUC >= 0.7: {int(((df['adv_auc']>=0.7)&(df['adv_auc']<0.8)).sum())}",
        f"- AUC >= 0.6: {int(((df['adv_auc']>=0.6)&(df['adv_auc']<0.7)).sum())}",
        f"- AUC <  0.6: {int((df['adv_auc']<0.6).sum())}",
        "",
        f"## Actionable: DYNAMIC features with AUC >= 0.7 ({len(dyn_high)})",
        "These are candidates for removal (shift in time-varying signal):",
        "",
    ]
    if len(dyn_high) > 0:
        lines.append("| feature | adv_auc | ks | rho_y |")
        lines.append("|---|---|---|---|")
        for _, r in dyn_high.head(25).iterrows():
            lines.append(f"| {r['feature']} | {r['adv_auc']:.3f} | {r['ks_stat']:.3f} | {r['rho_y']:+.3f} |")
    else:
        lines.append("_None — all shift is in static layout features._")

    lines.append("")
    lines.append(f"## Static features with AUC >= 0.7 ({len(stat_high)})")
    lines.append("These explain overall AUC 0.9994 (unseen-layout artifact). Removing these won't help since GroupKFold already handles the unseen-layout problem.")
    lines.append("")
    with open(os.path.join(OUT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSummary: {OUT}/summary.md")


if __name__ == "__main__":
    main()
