"""
Phase 3: Soft gating of mega33 <-> Q90 via p_ext classifier.

Two families to search:
  Power blend:   pred = (1-p)^beta * mega33 + (1 - (1-p)^beta) * Q90
  Linear:        pred = mega33 + alpha * p * (Q90 - mega33)

Also try p from different thresholds (y>30 vs y>50 vs y>100).

Kill gate: OOF improvement vs mega33 < 0.05 -> discard.
"""
import os
import json
import pickle
import numpy as np
import pandas as pd

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "soft_gate")
os.makedirs(OUT, exist_ok=True)


def mae(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    print("=" * 60)
    print("Phase 3: Soft gating")
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
    mega_test = mega["meta_avg_test"]

    q90_oof = np.load(os.path.join(ROOT, "results", "q90", "q90_oof.npy"))
    q90_test = np.load(os.path.join(ROOT, "results", "q90", "q90_test.npy"))

    p_oofs = {
        "gt30": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt30_oof.npy")),
        "gt50": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt50_oof.npy")),
        "gt100": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt100_oof.npy")),
    }
    p_tests = {
        "gt30": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt30_test.npy")),
        "gt50": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt50_test.npy")),
        "gt100": np.load(os.path.join(ROOT, "results", "tail_cls", "p_gt100_test.npy")),
    }

    mega_mae = mae(y, mega_oof)
    print(f"mega33 OOF MAE: {mega_mae:.4f}")

    # Grid search: for each p source, try power blend and linear
    results = []
    best = dict(mae=mega_mae, method="mega33_only", params={}, delta=0.0)

    # Power blend search
    print("\n--- Power blend: (1-p)^beta * mega33 + (1-(1-p)^beta) * Q90 ---")
    for p_name, p_oof in p_oofs.items():
        for beta in [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]:
            w = 1.0 - (1.0 - p_oof) ** beta
            pred = (1 - w) * mega_oof + w * q90_oof
            m = mae(y, pred)
            results.append(("power", p_name, beta, m))
            if m < best["mae"]:
                best = dict(mae=m, method="power", params=dict(p_src=p_name, beta=beta), delta=m - mega_mae)
            if beta in [0.5, 1.0, 2.0, 3.0, 5.0]:
                print(f"  p={p_name}, beta={beta:>4}: MAE={m:.4f}  delta={m - mega_mae:+.4f}")

    # Linear blend search
    print("\n--- Linear blend: mega33 + alpha * p * (Q90 - mega33) ---")
    for p_name, p_oof in p_oofs.items():
        for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
            pred = mega_oof + alpha * p_oof * (q90_oof - mega_oof)
            m = mae(y, pred)
            results.append(("linear", p_name, alpha, m))
            if m < best["mae"]:
                best = dict(mae=m, method="linear", params=dict(p_src=p_name, alpha=alpha), delta=m - mega_mae)
            if alpha in [0.3, 0.5, 0.7, 1.0]:
                print(f"  p={p_name}, alpha={alpha:>4}: MAE={m:.4f}  delta={m - mega_mae:+.4f}")

    # Nested (two-level): use p_gt30 * p_gt50 etc -> optional
    # Threshold hard gating (for comparison against soft)
    print("\n--- Hard threshold gating (sanity check, not for submission) ---")
    for p_name, p_oof in p_oofs.items():
        for pt in [0.3, 0.5, 0.7, 0.9]:
            mask = p_oof > pt
            pred = mega_oof.copy()
            pred[mask] = q90_oof[mask]
            m = mae(y, pred)
            if pt in [0.5, 0.7]:
                print(f"  hard p={p_name} > {pt}: MAE={m:.4f}  delta={m-mega_mae:+.4f}  flag_rate={mask.mean()*100:.2f}%")

    # Pure q90 for reference
    print(f"\nmega33 only:     {mega_mae:.4f}")
    print(f"pure q90:        {mae(y, q90_oof):.4f}")

    print("\n" + "=" * 60)
    print(f"BEST: method={best['method']}, params={best['params']}")
    print(f"  OOF MAE={best['mae']:.4f}, delta={best['delta']:+.4f}")

    # Kill gate check
    pass_gate = best["delta"] <= -0.05
    print(f"Kill gate (delta <= -0.05): {'PASS' if pass_gate else 'ABORT'}")

    # Build final test prediction using best config
    if best["method"] == "power":
        beta = best["params"]["beta"]
        p_src = best["params"]["p_src"]
        p_oof = p_oofs[p_src]
        p_test = p_tests[p_src]
        w_oof = 1.0 - (1.0 - p_oof) ** beta
        w_test = 1.0 - (1.0 - p_test) ** beta
        pred_oof = (1 - w_oof) * mega_oof + w_oof * q90_oof
        pred_test = (1 - w_test) * mega_test + w_test * q90_test
    elif best["method"] == "linear":
        alpha = best["params"]["alpha"]
        p_src = best["params"]["p_src"]
        p_oof = p_oofs[p_src]
        p_test = p_tests[p_src]
        pred_oof = mega_oof + alpha * p_oof * (q90_oof - mega_oof)
        pred_test = mega_test + alpha * p_test * (q90_test - mega_test)
    else:
        pred_oof = mega_oof.copy()
        pred_test = mega_test.copy()

    # Residual correlation vs mega33 on best gated prediction
    resid_mega = y - mega_oof
    resid_gated = y - pred_oof
    corr = float(np.corrcoef(resid_mega, resid_gated)[0, 1])
    print(f"residual_corr(mega33, gated): {corr:.4f}")

    # tail/body MAE break-down
    tail = y > 50
    print(f"gated tail MAE: {mae(y[tail], pred_oof[tail]):.4f}   (mega: {mae(y[tail], mega_oof[tail]):.4f})")
    print(f"gated body MAE: {mae(y[~tail], pred_oof[~tail]):.4f}   (mega: {mae(y[~tail], mega_oof[~tail]):.4f})")

    # Save
    np.save(os.path.join(OUT, "gated_oof.npy"), pred_oof)
    np.save(os.path.join(OUT, "gated_test.npy"), pred_test)
    summary = dict(
        best=best,
        mega_mae=mega_mae,
        residual_corr_vs_mega=corr,
        gated_tail_mae=mae(y[tail], pred_oof[tail]),
        gated_body_mae=mae(y[~tail], pred_oof[~tail]),
        pass_gate=bool(pass_gate),
        # Top 10 configs for transparency
        top_configs=sorted(results, key=lambda r: r[3])[:10],
    )
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    if not pass_gate:
        print("[ABORT] Phase 3 insufficient improvement.")


if __name__ == "__main__":
    main()
