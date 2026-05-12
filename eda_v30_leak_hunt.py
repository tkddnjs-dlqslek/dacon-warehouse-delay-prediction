"""
eda_v30_leak_hunt.py
====================
"1등의 +0.08 점프 추적" — v22가 건드리지 않은 4개 공백을 파고드는 EDA.

Plan: C:\\Users\\user\\.claude\\plans\\witty-twirling-bear.md

Sections:
  0) Setup & baseline load  (assert baseline_mae == 8.398946 ± 0.001)
  1) Missing Pattern Informativeness
  2) Extreme Regime Subset Regression
  3) Test-only Drift Bucket Matrix
  4) hub_spoke Subset Submodel
  5) scenario_id / ID 숫자 디코딩 (optional)
  6) Timeslot trajectory (optional)

DO NOT:
  - HP tuning (v26 실패)
  - 3-seed averaging (미세 악화)
  - mlp_deep 복제 (mega37 실패)
  - sample_weight on whole train (v17 악화)
  - pseudo-labeling (v22_pl 악화)
  - lag/rolling 확장 (v22 Section 6 완결)

각 섹션은 독립 함수 + checkpoint 파일 (재실행 안전).
"""
from __future__ import annotations
import os
import sys
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
ROOT = Path(r"c:/Users/user/Desktop/데이콘 4월")
DATA_TRAIN = ROOT / "train.csv"
DATA_TEST = ROOT / "test.csv"
DATA_LAYOUT = ROOT / "layout_info.csv"

RESULTS = ROOT / "results"
MEGA33_PKL = RESULTS / "mega33_final.pkl"
V23_PHASE1_PKL = RESULTS / "v23_phase1.pkl"

OUT = RESULTS / "eda_v30"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = "avg_delay_minutes_next_30m"
SORT_KEY = ["layout_id", "scenario_id"]   # OOF 정렬 기준 (mega33 학습 시 사용)

EXPECTED_BASELINE_MAE = 8.398946  # mega33_final.pkl / meta_avg_oof
BASELINE_TOL = 0.001


def ckpt_path(n: int) -> Path:
    return OUT / f"checkpoint_{n}.done"


def is_done(n: int) -> bool:
    return ckpt_path(n).exists()


def mark_done(n: int, note: str = "") -> None:
    ckpt_path(n).write_text(note or "ok", encoding="utf-8")


# ==================================================================
# Section 0: Setup & baseline load
# ==================================================================
def section_0() -> dict:
    """Load train/test, OOF, fold indices, assert baseline_mae."""
    print("=" * 60)
    print("Section 0: Setup & baseline load")
    print("=" * 60)

    # --- load train ---
    assert DATA_TRAIN.exists(), f"missing {DATA_TRAIN}"
    train = pd.read_csv(DATA_TRAIN)
    print(f"train.csv shape = {train.shape}")
    assert TARGET in train.columns, f"missing target column {TARGET}"

    # --- critical sort to match OOF order ---
    train = train.sort_values(SORT_KEY).reset_index(drop=True)
    print(f"sorted by {SORT_KEY}: first ID = {train['ID'].iloc[0]!r}")

    # --- load OOF baseline ---
    assert MEGA33_PKL.exists(), f"missing {MEGA33_PKL}"
    with open(MEGA33_PKL, "rb") as f:
        mega = pickle.load(f)
    oof = np.asarray(mega["meta_avg_oof"])
    assert oof.shape == (len(train),), (
        f"oof shape mismatch: {oof.shape} vs {len(train)}"
    )
    assert not np.isnan(oof).any(), "oof has NaN"

    y = train[TARGET].values.astype(np.float64)
    from sklearn.metrics import mean_absolute_error
    baseline_mae = mean_absolute_error(y, oof)
    print(f"baseline MAE (mega33_avg OOF) = {baseline_mae:.6f}")
    assert abs(baseline_mae - EXPECTED_BASELINE_MAE) < BASELINE_TOL, (
        f"baseline drift! got {baseline_mae:.6f}, "
        f"expected {EXPECTED_BASELINE_MAE:.6f}"
    )

    # --- GroupKFold indices (for downstream reuse) ---
    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=5)
    fold_ids = np.full(len(train), -1, dtype=np.int8)
    for fi, (_, val_idx) in enumerate(gkf.split(train, y, groups=train["layout_id"])):
        fold_ids[val_idx] = fi
    assert (fold_ids >= 0).all(), "some rows not assigned to fold"
    per_fold = np.bincount(fold_ids, minlength=5).tolist()
    print(f"per-fold val sizes: {per_fold}")

    np.save(OUT / "fold_idx.npy", fold_ids)

    # --- load test (for later sections) ---
    test = pd.read_csv(DATA_TEST)
    print(f"test.csv shape = {test.shape}")

    # --- load v23 features ---
    with open(V23_PHASE1_PKL, "rb") as f:
        v23 = pickle.load(f)
    selected_feats = list(v23["selected_features"])
    print(f"v23 selected_features: {len(selected_feats)}")

    # --- write outputs ---
    setup_csv = pd.DataFrame(
        [
            ("train_rows", len(train)),
            ("train_cols", train.shape[1]),
            ("test_rows", len(test)),
            ("test_cols", test.shape[1]),
            ("target_mean", float(y.mean())),
            ("target_std", float(y.std())),
            ("target_min", float(y.min())),
            ("target_max", float(y.max())),
            ("baseline_mae", float(baseline_mae)),
            ("expected_baseline_mae", EXPECTED_BASELINE_MAE),
            ("baseline_tolerance", BASELINE_TOL),
            ("n_folds", 5),
            ("fold_sort_key", "|".join(SORT_KEY)),
            ("n_v23_features", len(selected_feats)),
            ("mega33_n_meta", len(mega["meta_oofs"])),
        ],
        columns=["key", "value"],
    )
    setup_csv.to_csv(OUT / "section0_setup.csv", index=False)

    with open(OUT / "section0_setup.txt", "w", encoding="utf-8") as f:
        f.write(f"train shape: {train.shape}\n")
        f.write(f"test shape: {test.shape}\n")
        f.write(f"target: {TARGET}\n")
        f.write(f"baseline MAE: {baseline_mae:.6f}\n")
        f.write(f"per-fold sizes: {per_fold}\n")
        f.write(f"sort key: {SORT_KEY}\n")
        f.write(f"v23 n features: {len(selected_feats)}\n")

    mark_done(0, f"baseline_mae={baseline_mae:.6f}")
    print(f"[OK] section 0 complete - baseline MAE {baseline_mae:.6f}")

    return dict(
        train=train,
        test=test,
        y=y,
        oof=oof,
        baseline_mae=baseline_mae,
        fold_ids=fold_ids,
        selected_feats=selected_feats,
    )


# ==================================================================
# Section 1: Missing Pattern Informativeness
# ==================================================================
def _get_numeric_feature_cols(train: pd.DataFrame) -> list[str]:
    """numeric columns excluding ID/keys/target."""
    skip = {"ID", "layout_id", "scenario_id", TARGET}
    cols = []
    for c in train.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(train[c]):
            cols.append(c)
    return cols


def section_1(ctx: dict | None = None) -> dict:
    """Missing Pattern Informativeness.

    Does the NULL location itself correlate with target or residual?
    """
    print("=" * 60)
    print("Section 1: Missing Pattern Informativeness")
    print("=" * 60)

    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    test: pd.DataFrame = ctx["test"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]

    num_cols = _get_numeric_feature_cols(train)
    print(f"numeric feature cols: {len(num_cols)}")

    # --- build null-mask matrix ---
    Mtr = train[num_cols].isnull().to_numpy().astype(np.float32)   # (n, p)
    Mte = test[num_cols].isnull().to_numpy().astype(np.float32)

    null_ratio_tr = Mtr.mean(axis=0)
    print(
        f"cols with missing in train: "
        f"{(null_ratio_tr > 0).sum()} / {len(num_cols)}; "
        f"max ratio = {null_ratio_tr.max():.3f}"
    )

    # --- Spearman(null_mask, y) via Pearson on rank(y) ---
    from scipy.stats import rankdata, spearmanr
    ry = rankdata(y).astype(np.float64)
    ry_c = ry - ry.mean()
    ry_std = ry_c.std()

    # Spearman(null_mask, residual)
    residual = y - oof
    rr = rankdata(residual).astype(np.float64)
    rr_c = rr - rr.mean()
    rr_std = rr_c.std()

    rhos_y = np.zeros(len(num_cols), dtype=np.float64)
    rhos_res = np.zeros(len(num_cols), dtype=np.float64)
    for j, col in enumerate(num_cols):
        m = Mtr[:, j]
        if null_ratio_tr[j] == 0.0 or null_ratio_tr[j] == 1.0:
            rhos_y[j] = 0.0
            rhos_res[j] = 0.0
            continue
        m_c = m - m.mean()
        m_std = m_c.std()
        if m_std < 1e-12:
            continue
        rhos_y[j] = float((m_c * ry_c).mean() / (m_std * ry_std + 1e-12))
        rhos_res[j] = float((m_c * rr_c).mean() / (m_std * rr_std + 1e-12))

    null_info = pd.DataFrame(
        {
            "feature": num_cols,
            "null_ratio": null_ratio_tr,
            "spearman_rho_target": rhos_y,
            "abs_rho_target": np.abs(rhos_y),
            "spearman_rho_residual": rhos_res,
            "abs_rho_residual": np.abs(rhos_res),
        }
    )
    null_info = null_info.sort_values("abs_rho_target", ascending=False).reset_index(
        drop=True
    )
    null_info.insert(0, "rank", np.arange(1, len(null_info) + 1))
    null_info.to_csv(OUT / "null_informativeness.csv", index=False)
    print(f"wrote null_informativeness.csv (rows={len(null_info)})")

    # --- row-level null count ---
    row_null_tr = Mtr.sum(axis=1).astype(np.int32)
    row_null_te = Mte.sum(axis=1).astype(np.int32)
    print(
        f"row_null_count train: mean={row_null_tr.mean():.2f} "
        f"min={row_null_tr.min()} max={row_null_tr.max()}"
    )
    print(
        f"row_null_count test:  mean={row_null_te.mean():.2f} "
        f"min={row_null_te.min()} max={row_null_te.max()}"
    )

    rho_row_y = spearmanr(row_null_tr, y).statistic
    rho_row_res = spearmanr(row_null_tr, residual).statistic
    print(f"Spearman(row_null_count, y)        = {rho_row_y:+.5f}")
    print(f"Spearman(row_null_count, residual) = {rho_row_res:+.5f}")

    # --- scatter plot: row_null_count vs residual (subsample 10k) ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(42)
    idx = rng.choice(len(train), size=10000, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(row_null_tr[idx], residual[idx], s=3, alpha=0.25, c="tab:blue")
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("row_null_count")
    ax[0].set_ylabel("residual (y - mega33_oof)")
    ax[0].set_title(f"rho={rho_row_res:+.4f}")
    ax[1].scatter(row_null_tr[idx], y[idx], s=3, alpha=0.25, c="tab:orange")
    ax[1].set_xlabel("row_null_count")
    ax[1].set_ylabel("y (target)")
    ax[1].set_title(f"rho={rho_row_y:+.4f}")
    fig.tight_layout()
    fig.savefig(OUT / "null_row_scatter.png", dpi=100)
    plt.close(fig)
    print("wrote null_row_scatter.png")

    # --- adversarial AUC using null-mask only ---
    print("running adversarial (null-mask only) ...")
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.linear_model import LogisticRegression

    X_adv = np.vstack([Mtr, Mte])
    y_adv = np.concatenate(
        [np.zeros(len(Mtr), dtype=np.int8), np.ones(len(Mte), dtype=np.int8)]
    )
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    oof_adv = np.zeros(len(y_adv), dtype=np.float64)
    for fi, (tri, vai) in enumerate(skf.split(X_adv, y_adv)):
        clf = LogisticRegression(
            max_iter=500, solver="liblinear", C=1.0, n_jobs=None
        )
        clf.fit(X_adv[tri], y_adv[tri])
        oof_adv[vai] = clf.predict_proba(X_adv[vai])[:, 1]
    auc_null_only = roc_auc_score(y_adv, oof_adv)
    print(f"adversarial AUC (null-mask only) = {auc_null_only:.4f}")
    print("  (v22 full-feature adversarial AUC was 0.5632)")

    # --- summary txt ---
    top = null_info.head(10)
    lines = [
        "Section 1 summary - Missing Pattern Informativeness",
        "",
        f"numeric feature columns: {len(num_cols)}",
        f"cols with any null: {(null_ratio_tr > 0).sum()}",
        f"max null_ratio: {null_ratio_tr.max():.4f}",
        "",
        f"row_null_count train: mean={row_null_tr.mean():.2f} "
        f"min={row_null_tr.min()} max={row_null_tr.max()}",
        f"row_null_count test:  mean={row_null_te.mean():.2f} "
        f"min={row_null_te.min()} max={row_null_te.max()}",
        f"  train/test mean delta: {row_null_te.mean() - row_null_tr.mean():+.4f}",
        "",
        f"Spearman(row_null_count, y)        = {rho_row_y:+.5f}",
        f"Spearman(row_null_count, residual) = {rho_row_res:+.5f}",
        f"Adversarial AUC (null-mask only)   = {auc_null_only:.4f}",
        f"  (reference: v22 full-feature AUC = 0.5632)",
        "",
        "TOP 10 |Spearman(null_mask, target)|:",
    ]
    for _, r in top.iterrows():
        lines.append(
            f"  [{int(r['rank']):3d}] {r['feature']:40s} "
            f"null_ratio={r['null_ratio']:.4f} "
            f"rho_y={r['spearman_rho_target']:+.5f} "
            f"rho_res={r['spearman_rho_residual']:+.5f}"
        )
    lines.append("")

    # --- decision per plan ---
    max_abs_y = float(null_info["abs_rho_target"].max())
    max_abs_res = float(null_info["abs_rho_residual"].max())
    passes_feat = max_abs_y > 0.05 or max_abs_res > 0.05
    passes_row = abs(rho_row_res) > 0.03 or abs(rho_row_y) > 0.03
    verdict = "PROMOTE" if (passes_feat or passes_row) else "DROP"
    lines.append(
        f"DECISION (plan threshold: feat |rho|>0.05 OR row |rho|>0.03): {verdict}"
    )
    lines.append(
        f"  feat max |rho_y|={max_abs_y:.5f}  feat max |rho_res|={max_abs_res:.5f}"
    )
    lines.append(
        f"  row |rho_y|={abs(rho_row_y):.5f}  row |rho_res|={abs(rho_row_res):.5f}"
    )
    (OUT / "null_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("wrote null_summary.txt")
    print("-" * 60)
    for ln in lines[-6:]:
        print(ln)

    mark_done(
        1,
        f"verdict={verdict} auc_null={auc_null_only:.4f} "
        f"feat_max_y={max_abs_y:.5f} row_res={rho_row_res:+.5f}",
    )
    print(f"[OK] section 1 complete - verdict: {verdict}")

    return dict(
        null_info=null_info,
        row_null_tr=row_null_tr,
        row_null_te=row_null_te,
        rho_row_y=rho_row_y,
        rho_row_res=rho_row_res,
        auc_null_only=auc_null_only,
        verdict=verdict,
    )


# ==================================================================
# v23 feature engineering (ported from train_v23.py, no extreme_prob)
# ==================================================================
def engineer_features_v23(df: pd.DataFrame, layout_df: pd.DataFrame) -> pd.DataFrame:
    """Same as train_v23.py:engineer_features_v23, minus extreme_prob dep."""
    df = df.merge(layout_df, on="layout_id", how="left")
    df["timeslot"] = df.groupby(["layout_id", "scenario_id"]).cumcount()
    df["timeslot_sq"] = df["timeslot"] ** 2
    df["timeslot_norm"] = df["timeslot"] / 24.0
    df = df.sort_values(["layout_id", "scenario_id", "timeslot"]).reset_index(drop=True)
    group = df.groupby(["layout_id", "scenario_id"])

    key_cols = [
        "order_inflow_15m", "congestion_score", "robot_utilization",
        "battery_mean", "fault_count_15m", "blocked_path_15m",
        "pack_utilization", "charge_queue_length",
    ]
    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        df[f"{col}_lag1"] = g.shift(1)
        df[f"{col}_lag2"] = g.shift(2)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rmean3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_rstd3"] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f"{col}_rmean5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{col}_cummean"] = g.transform(lambda x: x.expanding().mean())

    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_lead2"] = g.shift(-2)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]

    extra_cols = [
        "max_zone_density", "robot_charging", "low_battery_ratio",
        "robot_idle", "near_collision_15m",
    ]
    for col in extra_cols:
        if col not in df.columns:
            continue
        g = group[col]
        df[f"{col}_lag1"] = g.shift(1)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]

    for col in key_cols:
        if col not in df.columns:
            continue
        g = group[col]
        df[f"{col}_sc_mean"] = g.transform("mean")
        df[f"{col}_sc_std"] = g.transform("std")
        df[f"{col}_sc_max"] = g.transform("max")
        df[f"{col}_sc_min"] = g.transform("min")
        df[f"{col}_sc_range"] = df[f"{col}_sc_max"] - df[f"{col}_sc_min"]
        df[f"{col}_sc_rank"] = g.rank(pct=True)
        df[f"{col}_sc_dev"] = df[col] - df[f"{col}_sc_mean"]

    df["order_per_robot"] = df["order_inflow_15m"] / (df["robot_active"] + 1)
    rta = df["robot_active"] + df["robot_idle"] + df["robot_charging"]
    df["robot_available_ratio"] = df["robot_idle"] / (rta + 1)
    df["robot_charging_ratio"] = df["robot_charging"] / (rta + 1)
    df["battery_risk"] = df["low_battery_ratio"] * df["charge_queue_length"]
    df["congestion_x_utilization"] = df["congestion_score"] * df["robot_utilization"]
    df["congestion_x_order"] = df["congestion_score"] * df["order_inflow_15m"]
    df["order_complexity"] = df["unique_sku_15m"] * df["avg_items_per_order"]
    df["urgent_order_volume"] = df["order_inflow_15m"] * df["urgent_order_ratio"]
    df["dock_pressure"] = df["loading_dock_util"] * df["outbound_truck_wait_min"]
    df["staff_per_order"] = df["staff_on_floor"] / (df["order_inflow_15m"] + 1)
    df["total_utilization"] = (
        df["pack_utilization"] + df["staging_area_util"] + df["loading_dock_util"]
    ) / 3
    df["fault_x_congestion"] = df["fault_count_15m"] * df["congestion_score"]
    df["battery_charge_pressure"] = df["low_battery_ratio"] * df["avg_charge_wait"]
    df["congestion_per_robot"] = df["congestion_score"] / (df["robot_active"] + 1)
    df["order_per_staff"] = df["order_inflow_15m"] / (df["staff_on_floor"] + 1)

    df["order_per_area"] = df["order_inflow_15m"] / (df["floor_area_sqm"] + 1) * 1000
    df["congestion_per_area"] = df["congestion_score"] / (df["floor_area_sqm"] + 1) * 1000
    df["fault_per_robot_total"] = df["fault_count_15m"] / (df["robot_total"] + 1)
    df["blocked_per_robot_total"] = df["blocked_path_15m"] / (df["robot_total"] + 1)
    df["collision_per_robot_total"] = df["near_collision_15m"] / (df["robot_total"] + 1)
    df["pack_util_per_station"] = df["pack_utilization"] / (df["pack_station_count"] + 1)
    df["charge_queue_per_charger"] = df["charge_queue_length"] / (df["charger_count"] + 1)
    df["order_per_pack_station"] = df["order_inflow_15m"] / (df["pack_station_count"] + 1)
    df["floor_area_per_robot"] = df["floor_area_sqm"] / (df["robot_total"] + 1)
    df["charger_ratio"] = df["charger_count"] / (df["robot_total"] + 1)
    df["robot_density"] = df["robot_total"] / (df["floor_area_sqm"] + 1) * 1000
    df["active_vs_total"] = df["robot_active"] / (df["robot_total"] + 1)
    df["congestion_x_aisle_width"] = df["congestion_score"] * df["aisle_width_avg"]
    df["congestion_x_compactness"] = df["congestion_score"] * df["layout_compactness"]
    df["blocked_x_one_way"] = df["blocked_path_15m"] * df["one_way_ratio"]
    df["utilization_x_compactness"] = df["robot_utilization"] * df["layout_compactness"]

    layout_static = [
        "layout_type", "aisle_width_avg", "intersection_count", "one_way_ratio",
        "pack_station_count", "charger_count", "layout_compactness", "zone_dispersion",
        "robot_total", "building_age_years", "floor_area_sqm", "ceiling_height_m",
        "fire_sprinkler_count", "emergency_exit_count",
    ]
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors="ignore")

    corr_remove = [
        "battery_mean_rmean3", "charge_queue_length_rmean3",
        "battery_mean_rmean5", "charge_queue_length_rmean5",
        "pack_utilization_rmean5", "battery_mean_lag1",
        "charge_queue_length_lag1", "congestion_score_rmean3",
        "order_inflow_15m_cummean", "robot_utilization_rmean5",
        "robot_utilization_rmean3", "order_inflow_15m_rmean5",
        "battery_risk", "congestion_score_rmean5",
        "pack_utilization_rmean3", "order_inflow_15m_rmean3",
        "charge_queue_length_lag2", "blocked_path_15m_rmean5",
    ]
    df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors="ignore")
    return df


def _load_fe_cached(ctx: dict) -> tuple[pd.DataFrame, list[str]]:
    """Build (or load cached) v23 FE matrix for train. Aligned with ctx['train']."""
    cache_pkl = OUT / "v30_fe_cache.pkl"
    if cache_pkl.exists():
        print(f"loading cached FE from {cache_pkl.name} ...")
        with open(cache_pkl, "rb") as f:
            blob = pickle.load(f)
        return blob["train_fe"], blob["feat_cols"]

    print("building v23 features for train (fresh) ...")
    layout = pd.read_csv(DATA_LAYOUT)
    # use raw train (unsorted) because engineer_features_v23 re-sorts internally
    train_raw = pd.read_csv(DATA_TRAIN)
    train_fe = engineer_features_v23(train_raw, layout)
    # Confirm the sort matches Section 0 ordering
    assert (
        train_fe["ID"].values == ctx["train"]["ID"].values
    ).all(), "train_fe row order drift from Section 0 train"

    selected_feats: list[str] = list(ctx["selected_feats"])
    available = [c for c in selected_feats if c in train_fe.columns]
    missing = [c for c in selected_feats if c not in train_fe.columns]
    print(f"v23 feats available: {len(available)} / {len(selected_feats)}")
    if missing:
        print(f"  (excluded, not in FE): {missing}")

    with open(cache_pkl, "wb") as f:
        pickle.dump({"train_fe": train_fe, "feat_cols": available}, f)
    print(f"cached -> {cache_pkl.name}")
    return train_fe, available


# ==================================================================
# Section 2: Extreme Regime Subset Regression
# ==================================================================
def section_2(ctx: dict | None = None) -> dict:
    """Train LGB-Huber on y > Q95 subset. Blend with mega33 OOF."""
    print("=" * 60)
    print("Section 2: Extreme Regime Subset Regression")
    print("=" * 60)
    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]
    fold_ids: np.ndarray = ctx["fold_ids"]

    train_fe, feat_cols = _load_fe_cached(ctx)
    X = train_fe[feat_cols].to_numpy(dtype=np.float32)
    print(f"X shape: {X.shape}")

    # --- define extreme subset ---
    q95 = float(np.quantile(y, 0.95))
    extreme_mask = y > q95
    n_ext = int(extreme_mask.sum())
    print(f"Q95 = {q95:.3f}, n_extreme = {n_ext} ({100*n_ext/len(y):.2f}%)")
    assert n_ext > 5000, "extreme subset too small"

    # baseline slice metrics
    from sklearn.metrics import mean_absolute_error
    mae_mega33_full = mean_absolute_error(y, oof)
    mae_mega33_ext = mean_absolute_error(y[extreme_mask], oof[extreme_mask])
    print(f"baseline MAE on full    = {mae_mega33_full:.5f}")
    print(f"baseline MAE on extreme = {mae_mega33_ext:.5f}")

    # --- 5-fold LGB-Huber on extreme subset (using shared fold_ids) ---
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    except ImportError as e:
        raise SystemExit(f"lightgbm not available: {e}")

    extreme_oof = np.zeros(len(y), dtype=np.float64)
    fold_metrics = []
    for f in range(5):
        tr_mask = (fold_ids != f) & extreme_mask
        val_idx = np.where(fold_ids == f)[0]
        val_ext_idx = np.where((fold_ids == f) & extreme_mask)[0]
        n_tr = int(tr_mask.sum())
        n_val_ext = len(val_ext_idx)

        # hold out a small chunk of tr for early stopping
        rng = np.random.default_rng(100 + f)
        tr_idx = np.where(tr_mask)[0]
        rng.shuffle(tr_idx)
        cut = int(0.9 * len(tr_idx))
        fit_idx = tr_idx[:cut]
        es_idx = tr_idx[cut:]

        model = LGBMRegressor(
            n_estimators=2000,
            learning_rate=0.03,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="huber",
            alpha=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[es_idx], y[es_idx])],
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)],
        )
        pred_val = model.predict(X[val_idx])
        extreme_oof[val_idx] = pred_val

        fmae_ext = mean_absolute_error(y[val_ext_idx], extreme_oof[val_ext_idx])
        fmae_all = mean_absolute_error(y[val_idx], extreme_oof[val_idx])
        fold_metrics.append(
            {
                "fold": f,
                "n_train_extreme": n_tr,
                "n_val_all": len(val_idx),
                "n_val_extreme": n_val_ext,
                "best_iter": int(model.best_iteration_ or 0),
                "mae_on_val_extreme": float(fmae_ext),
                "mae_on_val_all": float(fmae_all),
            }
        )
        print(
            f"  fold {f}: n_tr_ext={n_tr} best_it={model.best_iteration_} "
            f"mae_on_ext={fmae_ext:.4f} mae_on_all={fmae_all:.4f}"
        )

    pd.DataFrame(fold_metrics).to_csv(OUT / "extreme_fold_metrics.csv", index=False)
    np.save(OUT / "extreme_oof.npy", extreme_oof)

    mae_ext_model_on_ext = mean_absolute_error(y[extreme_mask], extreme_oof[extreme_mask])
    print(
        f"extreme-only model MAE on extreme subset = {mae_ext_model_on_ext:.5f} "
        f"(mega33 = {mae_mega33_ext:.5f})"
    )

    # --- global blend scan ---
    weights = [0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]
    blend_rows = []
    for w in weights:
        blend = (1 - w) * oof + w * extreme_oof
        mae = mean_absolute_error(y, blend)
        blend_rows.append(
            {"type": "global", "weight": w, "oof_mae": float(mae),
             "delta_vs_baseline": float(mae - mae_mega33_full)}
        )

    # --- gated blend scan: only blend when mega33 pred > gate quantile ---
    gates = [0.80, 0.90, 0.95]
    for gq in gates:
        gate_val = float(np.quantile(oof, gq))
        mask = oof > gate_val
        for w in weights:
            blend = oof.copy()
            blend[mask] = (1 - w) * oof[mask] + w * extreme_oof[mask]
            mae = mean_absolute_error(y, blend)
            blend_rows.append(
                {
                    "type": f"gated_q{int(gq*100)}",
                    "weight": w,
                    "oof_mae": float(mae),
                    "delta_vs_baseline": float(mae - mae_mega33_full),
                    "gate_value": gate_val,
                    "n_gated": int(mask.sum()),
                }
            )

    blend_df = pd.DataFrame(blend_rows)
    blend_df.to_csv(OUT / "extreme_oof_mae.csv", index=False)
    print(f"wrote extreme_oof_mae.csv ({len(blend_df)} rows)")

    # best overall
    best = blend_df.sort_values("oof_mae").iloc[0]
    print(
        f"BEST: type={best['type']} w={best['weight']:.2f} "
        f"mae={best['oof_mae']:.5f} delta={best['delta_vs_baseline']:+.5f}"
    )

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for t, sub in blend_df.groupby("type"):
        sub = sub.sort_values("weight")
        ax[0].plot(sub["weight"], sub["oof_mae"], marker="o", label=t, lw=1.2)
    ax[0].axhline(mae_mega33_full, color="k", lw=0.7, linestyle="--",
                  label=f"baseline {mae_mega33_full:.4f}")
    ax[0].axhline(mae_mega33_full - 0.05, color="r", lw=0.7, linestyle="--",
                  label="-0.05 target")
    ax[0].set_xlabel("blend weight")
    ax[0].set_ylabel("OOF MAE")
    ax[0].set_title("Extreme blend sweep")
    ax[0].legend(fontsize=8)

    # subset performance bar
    ax[1].bar(
        ["mega33\n(on ext)", "extreme_only\n(on ext)"],
        [mae_mega33_ext, mae_ext_model_on_ext],
        color=["gray", "tab:red"],
    )
    ax[1].set_ylabel("MAE on y>Q95 subset")
    ax[1].set_title(f"n_extreme={n_ext}")
    fig.tight_layout()
    fig.savefig(OUT / "extreme_blend_curve.png", dpi=100)
    plt.close(fig)

    # --- decision ---
    passes = bool((blend_df["delta_vs_baseline"] <= -0.05).any())
    weak = bool((blend_df["delta_vs_baseline"] <= -0.01).any() and not passes)
    verdict = "PROMOTE" if passes else ("WEAK" if weak else "DROP")

    lines = [
        "Section 2 summary - Extreme Regime Subset Regression",
        "",
        f"Q95 = {q95:.4f}  n_extreme = {n_ext} ({100*n_ext/len(y):.2f}%)",
        f"baseline (mega33) MAE full    = {mae_mega33_full:.5f}",
        f"baseline (mega33) MAE extreme = {mae_mega33_ext:.5f}",
        f"extreme_only model MAE extreme = {mae_ext_model_on_ext:.5f}",
        f"  delta vs mega33 on extreme  = {mae_ext_model_on_ext - mae_mega33_ext:+.5f}",
        "",
        "Best blend configurations:",
    ]
    top5 = blend_df.sort_values("oof_mae").head(8)
    for _, r in top5.iterrows():
        extra = f" gate={r.get('gate_value', float('nan'))}" if r["type"] != "global" else ""
        lines.append(
            f"  {r['type']:12s} w={r['weight']:.2f}  mae={r['oof_mae']:.5f}  "
            f"delta={r['delta_vs_baseline']:+.5f}{extra}"
        )
    lines += [
        "",
        f"DECISION (threshold delta<=-0.05): {verdict}",
        f"  min delta achieved = {blend_df['delta_vs_baseline'].min():+.5f}",
    ]
    (OUT / "extreme_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines[-10:]:
        print(ln)

    mark_done(2, f"verdict={verdict} min_delta={blend_df['delta_vs_baseline'].min():+.5f}")
    print(f"[OK] section 2 complete - verdict: {verdict}")

    return dict(
        verdict=verdict,
        blend_df=blend_df,
        extreme_oof=extreme_oof,
        q95=q95,
        mae_mega33_ext=mae_mega33_ext,
        mae_ext_model_on_ext=mae_ext_model_on_ext,
    )


# ==================================================================
# Section 3: Test-only Drift Bucket Matrix
# ==================================================================
def _build_test_fe() -> pd.DataFrame:
    """FE for test (cached)."""
    cache = OUT / "v30_test_fe_cache.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    print("building v23 features for test (fresh) ...")
    layout = pd.read_csv(DATA_LAYOUT)
    test_raw = pd.read_csv(DATA_TEST)
    test_fe = engineer_features_v23(test_raw, layout)
    with open(cache, "wb") as f:
        pickle.dump(test_fe, f)
    return test_fe


def section_3(ctx: dict | None = None) -> dict:
    """Drift bucket matrix (layout_type x timeslot//5) for v23 top-30 feats."""
    print("=" * 60)
    print("Section 3: Test-only Drift Bucket Matrix")
    print("=" * 60)
    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]
    baseline_mae: float = ctx["baseline_mae"]

    # --- load v23 importance ranking ---
    with open(V23_PHASE1_PKL, "rb") as f:
        v23 = pickle.load(f)
    imp_df = v23["importance"].sort_values("importance", ascending=False).reset_index(drop=True)
    top_feats = imp_df.head(30)["feature"].tolist()
    print(f"using v23 top-30 features (by importance)")

    # --- FE for train & test ---
    train_fe, feat_cols = _load_fe_cached(ctx)
    test_fe = _build_test_fe()

    # filter to top_feats that exist in both
    usable = [c for c in top_feats if c in train_fe.columns and c in test_fe.columns]
    print(f"usable top feats in both train & test: {len(usable)}")

    # --- add layout_type and timeslot to both (from layout_info) ---
    layout = pd.read_csv(DATA_LAYOUT)[["layout_id", "layout_type"]]
    train_meta = train[["layout_id"]].merge(layout, on="layout_id", how="left")
    # timeslot already in train_fe from engineer_features_v23
    assert "timeslot" in train_fe.columns, "timeslot missing in train_fe"
    train_meta["timeslot"] = train_fe["timeslot"].values

    test_meta = pd.read_csv(DATA_TEST, usecols=["layout_id"]).merge(
        layout, on="layout_id", how="left"
    )
    # Make test_fe's row order align with raw test order
    raw_test = pd.read_csv(DATA_TEST, usecols=["ID"])
    # engineer_features_v23 sorts by [layout_id, scenario_id, timeslot]; re-sort test_meta accordingly
    test_meta_sorted = (
        pd.read_csv(DATA_TEST, usecols=["ID", "layout_id", "scenario_id"])
        .merge(layout, on="layout_id", how="left")
    )
    # Use test_fe (already sorted by engineer_features_v23) — rebuild meta with matching sort
    test_meta = test_meta_sorted.sort_values(
        ["layout_id", "scenario_id"]
    ).reset_index(drop=True)
    test_meta["timeslot"] = test_fe["timeslot"].values

    # --- bucket ids ---
    train_bucket_type = train_meta["layout_type"].values
    train_bucket_ts = (train_fe["timeslot"].values // 5).astype(np.int8)  # 0..4
    test_bucket_type = test_meta["layout_type"].values
    test_bucket_ts = (test_fe["timeslot"].values // 5).astype(np.int8)

    types = ["narrow", "grid", "hybrid", "hub_spoke"]
    ts_buckets = [0, 1, 2, 3, 4]

    # --- per-bucket counts ---
    bucket_rows = []
    overall_mae = baseline_mae
    total_train = len(train)
    total_test = len(test_fe)

    for t in types:
        for tb in ts_buckets:
            tr_m = (train_bucket_type == t) & (train_bucket_ts == tb)
            te_m = (test_bucket_type == t) & (test_bucket_ts == tb)
            n_tr = int(tr_m.sum())
            n_te = int(te_m.sum())
            if n_tr < 100:
                continue
            from sklearn.metrics import mean_absolute_error
            bmae = mean_absolute_error(y[tr_m], oof[tr_m])
            bucket_rows.append(
                {
                    "layout_type": t,
                    "ts_bucket": int(tb),
                    "n_train": n_tr,
                    "n_test": n_te,
                    "train_frac": n_tr / total_train,
                    "test_frac": n_te / total_test,
                    "oof_mae": float(bmae),
                    "delta_vs_overall": float(bmae - overall_mae),
                }
            )
    bucket_df = pd.DataFrame(bucket_rows).sort_values(
        "delta_vs_overall", ascending=False
    ).reset_index(drop=True)
    print(f"bucket_df rows: {len(bucket_df)}")

    # --- drift score per feature per bucket (mean shift) ---
    from scipy.stats import ks_2samp
    drift_rows = []
    drift_heat = np.zeros((len(usable), len(bucket_rows)), dtype=np.float64)
    col_labels = []
    for bi, r in enumerate(bucket_rows):
        t = r["layout_type"]
        tb = r["ts_bucket"]
        tr_m = (train_bucket_type == t) & (train_bucket_ts == tb)
        te_m = (test_bucket_type == t) & (test_bucket_ts == tb)
        col_labels.append(f"{t}-ts{tb}")
        for fi, feat in enumerate(usable):
            tr_vals = train_fe[feat].values[tr_m]
            te_vals = test_fe[feat].values[te_m]
            tr_vals = tr_vals[~np.isnan(tr_vals)]
            te_vals = te_vals[~np.isnan(te_vals)]
            if len(tr_vals) < 50 or len(te_vals) < 50:
                ks = np.nan
            else:
                # subsample for speed
                max_n = 5000
                if len(tr_vals) > max_n:
                    tr_vals = np.random.default_rng(0).choice(tr_vals, max_n, replace=False)
                if len(te_vals) > max_n:
                    te_vals = np.random.default_rng(0).choice(te_vals, max_n, replace=False)
                ks = float(ks_2samp(tr_vals, te_vals).statistic)
            drift_heat[fi, bi] = ks
            drift_rows.append(
                {
                    "layout_type": t,
                    "ts_bucket": int(tb),
                    "feature": feat,
                    "ks_stat": ks,
                }
            )

    # --- aggregate drift_score per bucket = mean KS across top feats ---
    bucket_drift_mean = np.nanmean(drift_heat, axis=0)
    bucket_drift_max = np.nanmax(drift_heat, axis=0)
    for bi, r in enumerate(bucket_rows):
        r["drift_score_mean"] = float(bucket_drift_mean[bi])
        r["drift_score_max"] = float(bucket_drift_max[bi])
    bucket_df2 = pd.DataFrame(bucket_rows).sort_values(
        "delta_vs_overall", ascending=False
    ).reset_index(drop=True)
    bucket_df2.to_csv(OUT / "drift_bucket_mae.csv", index=False)
    pd.DataFrame(drift_rows).to_csv(OUT / "drift_feat_bucket.csv", index=False)
    print("wrote drift_bucket_mae.csv, drift_feat_bucket.csv")

    # --- plot heatmap ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    im = axes[0].imshow(drift_heat, aspect="auto", cmap="viridis")
    axes[0].set_yticks(range(len(usable)))
    axes[0].set_yticklabels(usable, fontsize=7)
    axes[0].set_xticks(range(len(col_labels)))
    axes[0].set_xticklabels(col_labels, rotation=45, ha="right", fontsize=7)
    axes[0].set_title("KS(train vs test) per feature x bucket")
    fig.colorbar(im, ax=axes[0], fraction=0.04)

    # bucket MAE bar chart
    ord_buckets = list(range(len(bucket_rows)))
    maes = [bucket_rows[bi]["oof_mae"] for bi in ord_buckets]
    axes[1].barh(col_labels, maes, color="tab:gray")
    axes[1].axvline(overall_mae, color="r", ls="--", label=f"overall {overall_mae:.3f}")
    axes[1].axvline(overall_mae + 1.0, color="orange", ls="--", label="+1.0 thresh")
    axes[1].set_xlabel("OOF MAE (mega33)")
    axes[1].set_title("Per-bucket MAE")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "drift_heatmap.png", dpi=100)
    plt.close(fig)
    print("wrote drift_heatmap.png")

    # --- decision ---
    high = bucket_df2[bucket_df2["delta_vs_overall"] >= 1.0]
    covered_frac = float(high["train_frac"].sum())
    passes = bool(covered_frac >= 0.10)
    verdict = "PROMOTE" if passes else "DROP"

    lines = [
        "Section 3 summary - Test-only Drift Bucket Matrix",
        "",
        f"overall baseline MAE: {overall_mae:.5f}",
        f"buckets analyzed: {len(bucket_rows)}",
        f"usable top feats for KS: {len(usable)}",
        "",
        "Top-5 worst buckets (by delta_vs_overall):",
    ]
    for _, r in bucket_df2.head(5).iterrows():
        lines.append(
            f"  {r['layout_type']:10s} ts{int(r['ts_bucket'])}  n_tr={int(r['n_train']):5d} "
            f"mae={r['oof_mae']:.4f} delta={r['delta_vs_overall']:+.4f} "
            f"train_frac={r['train_frac']:.3f} test_frac={r['test_frac']:.3f} "
            f"ks_mean={r['drift_score_mean']:.4f}"
        )
    lines += [
        "",
        "Top-5 highest drift_score (mean KS) buckets:",
    ]
    bd_drift = bucket_df2.sort_values("drift_score_mean", ascending=False).head(5)
    for _, r in bd_drift.iterrows():
        lines.append(
            f"  {r['layout_type']:10s} ts{int(r['ts_bucket'])}  "
            f"ks_mean={r['drift_score_mean']:.4f} ks_max={r['drift_score_max']:.4f} "
            f"mae={r['oof_mae']:.4f} delta={r['delta_vs_overall']:+.4f}"
        )
    lines += [
        "",
        f"DECISION (plan: delta>=+1.0 bucket covering >=10% samples): {verdict}",
        f"  buckets with delta>=+1.0: {len(high)}",
        f"  sample fraction covered: {covered_frac:.4f}",
    ]
    (OUT / "drift_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    for ln in lines[-10:]:
        print(ln)

    mark_done(3, f"verdict={verdict} cov={covered_frac:.3f} n_high={len(high)}")
    print(f"[OK] section 3 complete - verdict: {verdict}")
    return dict(verdict=verdict, bucket_df=bucket_df2, covered_frac=covered_frac)


# ==================================================================
# Section 4: hub_spoke Subset Submodel
# ==================================================================
def section_4(ctx: dict | None = None) -> dict:
    """Train LGB-Huber on hub_spoke subset only, compare vs mega33."""
    print("=" * 60)
    print("Section 4: hub_spoke Subset Submodel")
    print("=" * 60)
    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]
    fold_ids: np.ndarray = ctx["fold_ids"]

    # --- rebuild FE (cached) ---
    train_fe, feat_cols = _load_fe_cached(ctx)
    X = train_fe[feat_cols].to_numpy(dtype=np.float32)

    # --- layout_type / timeslot alignment ---
    layout = pd.read_csv(DATA_LAYOUT)[["layout_id", "layout_type"]]
    meta = train[["layout_id"]].merge(layout, on="layout_id", how="left")
    layout_type = meta["layout_type"].values
    timeslot = train_fe["timeslot"].values

    hs_mask = layout_type == "hub_spoke"
    n_hs = int(hs_mask.sum())
    print(f"hub_spoke rows: {n_hs} ({100*n_hs/len(y):.2f}%)")
    assert n_hs > 30000, "hub_spoke subset too small"

    # --- baseline on hub_spoke ---
    from sklearn.metrics import mean_absolute_error
    mae_mega33_hs = mean_absolute_error(y[hs_mask], oof[hs_mask])
    print(f"mega33 MAE on hub_spoke = {mae_mega33_hs:.5f}")
    print(f"mega33 MAE overall       = {ctx['baseline_mae']:.5f}")

    # --- per-fold LGB-Huber on hs subset ---
    try:
        from lightgbm import LGBMRegressor, early_stopping, log_evaluation
    except ImportError as e:
        raise SystemExit(f"lightgbm not available: {e}")

    hs_oof = np.full(len(y), np.nan, dtype=np.float64)
    fold_metrics = []
    imp_accum = np.zeros(len(feat_cols), dtype=np.float64)
    for f in range(5):
        tr_m = hs_mask & (fold_ids != f)
        val_m = hs_mask & (fold_ids == f)
        tr_idx = np.where(tr_m)[0]
        val_idx = np.where(val_m)[0]
        n_tr = len(tr_idx)
        n_val = len(val_idx)
        if n_val == 0:
            print(f"  fold {f}: no hub_spoke val samples, skip")
            continue

        # held-out for early stopping
        rng = np.random.default_rng(200 + f)
        tr_shuf = tr_idx.copy()
        rng.shuffle(tr_shuf)
        cut = int(0.9 * len(tr_shuf))
        fit_idx = tr_shuf[:cut]
        es_idx = tr_shuf[cut:]

        model = LGBMRegressor(
            n_estimators=3000,
            learning_rate=0.03,
            num_leaves=63,
            max_depth=-1,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            objective="huber",
            alpha=0.9,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(
            X[fit_idx], y[fit_idx],
            eval_set=[(X[es_idx], y[es_idx])],
            callbacks=[early_stopping(150, verbose=False), log_evaluation(0)],
        )
        pred = model.predict(X[val_idx])
        hs_oof[val_idx] = pred
        mae_fold = mean_absolute_error(y[val_idx], pred)
        mae_mega_fold = mean_absolute_error(y[val_idx], oof[val_idx])

        imp_accum += np.asarray(model.feature_importances_, dtype=np.float64)

        fold_metrics.append(
            {
                "fold": f,
                "n_train": n_tr,
                "n_val": n_val,
                "best_iter": int(model.best_iteration_ or 0),
                "mae_submodel": float(mae_fold),
                "mae_mega33": float(mae_mega_fold),
                "delta": float(mae_fold - mae_mega_fold),
            }
        )
        print(
            f"  fold {f}: n_tr={n_tr} n_val={n_val} best_it={model.best_iteration_} "
            f"mae_sub={mae_fold:.4f} mae_mega={mae_mega_fold:.4f} delta={mae_fold-mae_mega_fold:+.4f}"
        )

    pd.DataFrame(fold_metrics).to_csv(OUT / "hubspoke_fold_metrics.csv", index=False)
    np.save(OUT / "hubspoke_oof.npy", hs_oof)

    # --- overall hub_spoke MAE ---
    valid = ~np.isnan(hs_oof)
    mae_hs_sub = mean_absolute_error(y[valid], hs_oof[valid])
    delta_full_hs = mae_hs_sub - mae_mega33_hs
    print(f"submodel MAE on hub_spoke = {mae_hs_sub:.5f}")
    print(f"delta vs mega33 on hub_spoke = {delta_full_hs:+.5f}")

    # --- per ts_bucket breakdown ---
    ts_bucket = (timeslot // 5).astype(np.int8)
    bucket_rows = []
    for tb in range(5):
        bm = hs_mask & (ts_bucket == tb)
        if bm.sum() < 100:
            continue
        m_sub = mean_absolute_error(y[bm], hs_oof[bm])
        m_meg = mean_absolute_error(y[bm], oof[bm])
        bucket_rows.append(
            {
                "ts_bucket": int(tb),
                "n": int(bm.sum()),
                "mae_submodel": float(m_sub),
                "mae_mega33": float(m_meg),
                "delta": float(m_sub - m_meg),
            }
        )
    bdf = pd.DataFrame(bucket_rows)
    bdf.to_csv(OUT / "hubspoke_ts_breakdown.csv", index=False)
    print("per ts_bucket (hub_spoke only):")
    for _, r in bdf.iterrows():
        print(
            f"  ts{int(r['ts_bucket'])}: n={int(r['n']):5d} "
            f"sub={r['mae_submodel']:.4f} mega={r['mae_mega33']:.4f} "
            f"delta={r['delta']:+.4f}"
        )

    # --- feature importance diff vs v23 whole ---
    with open(V23_PHASE1_PKL, "rb") as f:
        v23 = pickle.load(f)
    whole_imp = v23["importance"].set_index("feature")["importance"]
    sub_imp_series = pd.Series(imp_accum / 5.0, index=feat_cols, name="imp_submodel")
    whole_aligned = whole_imp.reindex(sub_imp_series.index).fillna(0.0)
    imp_df = pd.DataFrame(
        {
            "feature": sub_imp_series.index,
            "imp_submodel": sub_imp_series.values,
            "imp_whole": whole_aligned.values,
        }
    )
    # normalize both to rank 0-1 for fair comparison
    imp_df["rank_submodel"] = imp_df["imp_submodel"].rank(ascending=False).astype(int)
    imp_df["rank_whole"] = imp_df["imp_whole"].rank(ascending=False).astype(int)
    imp_df["rank_change"] = imp_df["rank_whole"] - imp_df["rank_submodel"]  # +: more important in submodel
    imp_df = imp_df.sort_values("rank_change", ascending=False).reset_index(drop=True)
    imp_df.to_csv(OUT / "hubspoke_imp.csv", index=False)
    print("top 10 features MORE important in hub_spoke submodel (vs whole):")
    for _, r in imp_df.head(10).iterrows():
        print(
            f"  {r['feature']:40s} "
            f"rank_sub={int(r['rank_submodel']):3d} "
            f"rank_whole={int(r['rank_whole']):3d} "
            f"change={int(r['rank_change']):+4d}"
        )

    # --- global MAE effect if hs submodel replaces mega33 on hs subset ---
    blend = oof.copy()
    blend[valid] = hs_oof[valid]
    global_mae_swap = mean_absolute_error(y, blend)
    delta_global = global_mae_swap - ctx["baseline_mae"]
    print(f"global MAE if hs-only swap = {global_mae_swap:.5f} (delta {delta_global:+.5f})")

    # --- also try a simple blend on hs subset ---
    best_blend = None
    blend_rows = []
    for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        b = oof.copy()
        b[valid] = (1 - w) * oof[valid] + w * hs_oof[valid]
        g_mae = mean_absolute_error(y, b)
        blend_rows.append(
            {"weight": w, "global_mae": float(g_mae), "delta": float(g_mae - ctx["baseline_mae"])}
        )
        if best_blend is None or g_mae < best_blend[1]:
            best_blend = (w, g_mae)
    pd.DataFrame(blend_rows).to_csv(OUT / "hubspoke_blend.csv", index=False)

    # --- decision ---
    passes_plan = delta_full_hs <= -0.5  # plan threshold on hs subset
    passes_global = delta_global <= -0.05  # end-to-end threshold
    verdict = (
        "PROMOTE" if (passes_plan and passes_global)
        else ("WEAK" if passes_plan else "DROP")
    )

    lines = [
        "Section 4 summary - hub_spoke Subset Submodel",
        "",
        f"hub_spoke n = {n_hs}",
        f"mega33 MAE on hub_spoke = {mae_mega33_hs:.5f}",
        f"submodel MAE on hub_spoke = {mae_hs_sub:.5f}",
        f"delta on hub_spoke = {delta_full_hs:+.5f}  (plan threshold <= -0.5)",
        "",
        f"global MAE (full swap) = {global_mae_swap:.5f}",
        f"delta vs baseline (full)  = {delta_global:+.5f}  (plan threshold <= -0.05)",
        "",
        f"best blend weight = {best_blend[0]:.2f}  "
        f"global_mae = {best_blend[1]:.5f}  "
        f"delta = {best_blend[1] - ctx['baseline_mae']:+.5f}",
        "",
        "per ts_bucket (hub_spoke only):",
    ]
    for _, r in bdf.iterrows():
        lines.append(
            f"  ts{int(r['ts_bucket'])} n={int(r['n']):5d} "
            f"sub={r['mae_submodel']:.4f} mega={r['mae_mega33']:.4f} "
            f"delta={r['delta']:+.4f}"
        )
    lines += [
        "",
        f"DECISION: {verdict}",
        f"  hs subset delta achieved  = {delta_full_hs:+.5f}",
        f"  global delta achieved     = {delta_global:+.5f}",
    ]
    (OUT / "hubspoke_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    mark_done(
        4,
        f"verdict={verdict} hs_delta={delta_full_hs:+.5f} global_delta={delta_global:+.5f}",
    )
    print(f"[OK] section 4 complete - verdict: {verdict}")
    return dict(
        verdict=verdict,
        mae_mega33_hs=mae_mega33_hs,
        mae_hs_sub=mae_hs_sub,
        delta_hs=delta_full_hs,
        delta_global=delta_global,
    )


# ==================================================================
# Section 5: scenario_id / ID numeric decode
# ==================================================================
def _parse_int_suffix(s: pd.Series) -> np.ndarray:
    """'SC_07598' -> 7598, 'TRAIN_000123' -> 123"""
    return s.str.split("_").str[-1].astype(np.int64).values


def section_5(ctx: dict | None = None) -> dict:
    print("=" * 60)
    print("Section 5: scenario_id / ID numeric decode")
    print("=" * 60)
    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]
    baseline_mae = ctx["baseline_mae"]
    residual = y - oof

    test = pd.read_csv(DATA_TEST, usecols=["ID", "layout_id", "scenario_id"])
    layout = pd.read_csv(DATA_LAYOUT)[["layout_id", "layout_type"]]

    # parse numeric suffixes
    train_sc_num = _parse_int_suffix(train["scenario_id"])
    test_sc_num = _parse_int_suffix(test["scenario_id"])
    train_id_num = _parse_int_suffix(train["ID"])
    test_id_num = _parse_int_suffix(test["ID"])

    # --- range overlap ---
    sc_tr_range = (int(train_sc_num.min()), int(train_sc_num.max()))
    sc_te_range = (int(test_sc_num.min()), int(test_sc_num.max()))
    overlap = (
        max(sc_tr_range[0], sc_te_range[0]),
        min(sc_tr_range[1], sc_te_range[1]),
    )
    overlap_size = max(0, overlap[1] - overlap[0] + 1)

    tr_sc_unique = set(np.unique(train_sc_num).tolist())
    te_sc_unique = set(np.unique(test_sc_num).tolist())
    overlap_sc_set = tr_sc_unique & te_sc_unique

    print(f"train scenario_num range: {sc_tr_range}  n_unique={len(tr_sc_unique)}")
    print(f"test  scenario_num range: {sc_te_range}  n_unique={len(te_sc_unique)}")
    print(f"range overlap: {overlap}  size={overlap_size}")
    print(f"set intersection: {len(overlap_sc_set)} scenario_num values shared")

    id_tr_range = (int(train_id_num.min()), int(train_id_num.max()))
    id_te_range = (int(test_id_num.min()), int(test_id_num.max()))
    print(f"train ID num range: {id_tr_range}")
    print(f"test  ID num range: {id_te_range}")

    # --- correlations with target and residual ---
    from scipy.stats import spearmanr
    rho_sc_y = spearmanr(train_sc_num, y).statistic
    rho_sc_res = spearmanr(train_sc_num, residual).statistic
    rho_id_y = spearmanr(train_id_num, y).statistic
    rho_id_res = spearmanr(train_id_num, residual).statistic
    print(f"Spearman(scenario_num, y)        = {rho_sc_y:+.5f}")
    print(f"Spearman(scenario_num, residual) = {rho_sc_res:+.5f}")
    print(f"Spearman(ID_num, y)              = {rho_id_y:+.5f}")
    print(f"Spearman(ID_num, residual)       = {rho_id_res:+.5f}")

    # --- scenario_num vs layout_type distribution ---
    meta = train[["layout_id", "scenario_id"]].merge(layout, on="layout_id", how="left")
    meta["sc_num"] = train_sc_num
    type_ranges = []
    for t in ["narrow", "grid", "hybrid", "hub_spoke"]:
        sub = meta[meta["layout_type"] == t]["sc_num"].values
        if len(sub) == 0:
            continue
        type_ranges.append(
            {
                "layout_type": t,
                "n": int(len(sub)),
                "min": int(sub.min()),
                "max": int(sub.max()),
                "mean": float(sub.mean()),
                "median": float(np.median(sub)),
            }
        )
    type_df = pd.DataFrame(type_ranges)
    print("\nscenario_num distribution by layout_type:")
    print(type_df.to_string(index=False))

    # --- scenario_num binned residual pattern ---
    bins = pd.cut(train_sc_num, bins=20)
    bin_stats = pd.DataFrame(
        {
            "sc_num_bin": bins,
            "residual": residual,
            "y": y,
        }
    ).groupby("sc_num_bin", observed=True).agg(
        n=("y", "count"),
        residual_mean=("residual", "mean"),
        residual_abs_mean=("residual", lambda x: np.abs(x).mean()),
        y_mean=("y", "mean"),
    ).reset_index()
    bin_stats.to_csv(OUT / "scenario_num_bins.csv", index=False)

    # --- decision ---
    passes = (
        max(abs(rho_sc_y), abs(rho_sc_res)) > 0.02
        or max(abs(rho_id_y), abs(rho_id_res)) > 0.02
    )
    verdict = "PROMOTE" if passes else "DROP"

    lines = [
        "Section 5 summary - scenario_id / ID numeric decode",
        "",
        f"train scenario_num range = {sc_tr_range}, n_unique={len(tr_sc_unique)}",
        f"test  scenario_num range = {sc_te_range}, n_unique={len(te_sc_unique)}",
        f"range numeric overlap size = {overlap_size}",
        f"set intersection (shared sc_num values) = {len(overlap_sc_set)}",
        f"train ID range = {id_tr_range}",
        f"test  ID range = {id_te_range}",
        "",
        f"Spearman(scenario_num, y)        = {rho_sc_y:+.5f}",
        f"Spearman(scenario_num, residual) = {rho_sc_res:+.5f}",
        f"Spearman(ID_num, y)              = {rho_id_y:+.5f}",
        f"Spearman(ID_num, residual)       = {rho_id_res:+.5f}",
        "",
        "scenario_num distribution by layout_type:",
        type_df.to_string(index=False),
        "",
        f"DECISION (threshold |rho|>0.02): {verdict}",
    ]
    (OUT / "scenario_id_decode.txt").write_text("\n".join(lines), encoding="utf-8")
    mark_done(
        5,
        f"verdict={verdict} rho_sc_y={rho_sc_y:+.5f} overlap={len(overlap_sc_set)}",
    )
    print(f"[OK] section 5 complete - verdict: {verdict}")
    return dict(verdict=verdict)


# ==================================================================
# Section 6: Timeslot trajectory
# ==================================================================
def section_6(ctx: dict | None = None) -> dict:
    print("=" * 60)
    print("Section 6: Timeslot trajectory")
    print("=" * 60)
    if ctx is None:
        ctx = section_0()

    train: pd.DataFrame = ctx["train"]
    y: np.ndarray = ctx["y"]
    oof: np.ndarray = ctx["oof"]
    baseline_mae = ctx["baseline_mae"]

    train_fe, feat_cols = _load_fe_cached(ctx)
    timeslot = train_fe["timeslot"].values.astype(np.int16)
    print(f"timeslot range: [{timeslot.min()}, {timeslot.max()}]  unique={len(np.unique(timeslot))}")

    # merge layout_type
    layout = pd.read_csv(DATA_LAYOUT)[["layout_id", "layout_type"]]
    meta = train[["layout_id"]].merge(layout, on="layout_id", how="left")
    layout_type = meta["layout_type"].values

    # --- target trajectory: y mean by timeslot globally and by layout_type ---
    curve_rows = []
    for ts in range(int(timeslot.max()) + 1):
        m = timeslot == ts
        if m.sum() == 0:
            continue
        row = {"timeslot": ts, "n": int(m.sum()), "y_mean": float(y[m].mean()),
               "y_median": float(np.median(y[m])), "y_std": float(y[m].std())}
        for t in ["narrow", "grid", "hybrid", "hub_spoke"]:
            mt = m & (layout_type == t)
            if mt.sum() > 0:
                row[f"y_mean_{t}"] = float(y[mt].mean())
        curve_rows.append(row)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(OUT / "trajectory_curve.csv", index=False)
    print(f"target curve written ({len(curve_df)} timeslots)")
    print(curve_df[["timeslot", "n", "y_mean", "y_median", "y_std"]].head(25).to_string(index=False))

    # --- monotonicity: fraction of scenarios where y is increasing within scenario ---
    df = pd.DataFrame(
        {
            "layout_id": train["layout_id"].values,
            "scenario_id": train["scenario_id"].values,
            "timeslot": timeslot,
            "y": y,
            "layout_type": layout_type,
            "order_inflow_15m": train_fe["order_inflow_15m"].values,
        }
    )
    df = df.sort_values(["layout_id", "scenario_id", "timeslot"]).reset_index(drop=True)
    g = df.groupby(["layout_id", "scenario_id"], sort=False)
    # compute per-scenario summary
    def _traj_stats(sub: pd.DataFrame) -> pd.Series:
        ys = sub["y"].values
        if len(ys) < 3:
            return pd.Series({"argmax_ts": -1, "argmin_ts": -1, "range": 0.0,
                              "spearman_y_ts": 0.0, "y_first": ys[0], "y_last": ys[-1]})
        ts_ = sub["timeslot"].values
        # spearman(y, timeslot) — monotonic increase detector
        from scipy.stats import spearmanr
        rho = spearmanr(ts_, ys).statistic
        return pd.Series(
            {
                "argmax_ts": int(ts_[np.argmax(ys)]),
                "argmin_ts": int(ts_[np.argmin(ys)]),
                "range": float(ys.max() - ys.min()),
                "spearman_y_ts": float(rho if not np.isnan(rho) else 0.0),
                "y_first": float(ys[0]),
                "y_last": float(ys[-1]),
            }
        )
    print("computing per-scenario trajectory stats (may take ~30s)...")
    traj = g.apply(_traj_stats, include_groups=False).reset_index()
    # merge layout_type
    traj = traj.merge(
        df[["layout_id", "scenario_id", "layout_type"]].drop_duplicates(),
        on=["layout_id", "scenario_id"],
        how="left",
    )
    traj.to_csv(OUT / "traj_stats.csv", index=False)

    monotonic_frac = float((traj["spearman_y_ts"] > 0.5).mean())
    strong_monotonic_frac = float((traj["spearman_y_ts"] > 0.8).mean())
    neg_monotonic_frac = float((traj["spearman_y_ts"] < -0.5).mean())
    print(f"fraction scenarios with rho(y,ts)>0.5:  {monotonic_frac:.4f}")
    print(f"fraction scenarios with rho(y,ts)>0.8:  {strong_monotonic_frac:.4f}")
    print(f"fraction scenarios with rho(y,ts)<-0.5: {neg_monotonic_frac:.4f}")
    rho_mean_by_type = traj.groupby("layout_type")["spearman_y_ts"].mean()
    print("mean rho(y,ts) by layout_type:")
    print(rho_mean_by_type.to_string())

    # argmax distribution
    print(f"argmax timeslot distribution (which ts holds the max y per scenario):")
    argmax_counts = traj["argmax_ts"].value_counts().sort_index()
    print(argmax_counts.to_string())

    # --- cumsum correlation vs raw (NaN-safe: fillna mean before cumsum) ---
    print("computing cumulative feature correlations (NaN-safe)...")
    from scipy.stats import spearmanr

    def _safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() < 100:
            return float("nan")
        return float(spearmanr(a[mask], b[mask]).statistic)

    # fill order_inflow NaN with column mean (for cumsum stability)
    oi = df["order_inflow_15m"].values.astype(np.float64)
    oi_mean = np.nanmean(oi)
    oi_filled = np.where(np.isnan(oi), oi_mean, oi)
    df["order_inflow_filled"] = oi_filled
    df["order_cumsum"] = (
        df.groupby(["layout_id", "scenario_id"], sort=False)["order_inflow_filled"]
        .cumsum()
        .values
    )

    rho_raw = _safe_spearman(oi, df["y"].values)
    rho_cum = _safe_spearman(df["order_cumsum"].values, df["y"].values)
    print(f"Spearman(order_inflow_15m, y) = {rho_raw:+.5f}")
    print(f"Spearman(order_cumsum, y)     = {rho_cum:+.5f}")
    print(f"  delta = {rho_cum - rho_raw:+.5f}")

    extra_cum = {}
    for col in [
        "congestion_score", "fault_count_15m", "blocked_path_15m",
        "charge_queue_length", "near_collision_15m",
    ]:
        if col not in train_fe.columns:
            continue
        vals = train_fe[col].values.astype(np.float64)
        col_mean = np.nanmean(vals)
        filled = np.where(np.isnan(vals), col_mean, vals)
        tmp = pd.DataFrame(
            {
                "layout_id": train["layout_id"].values,
                "scenario_id": train["scenario_id"].values,
                "ts": timeslot,
                "val": filled,
            }
        ).sort_values(["layout_id", "scenario_id", "ts"]).reset_index(drop=True)
        cum = tmp.groupby(["layout_id", "scenario_id"], sort=False)["val"].cumsum().values
        # reorder cum to match df (which is already sorted same way)
        rho_c = _safe_spearman(cum, tmp.merge(
            df[["layout_id", "scenario_id", "timeslot", "y"]],
            left_on=["layout_id", "scenario_id", "ts"],
            right_on=["layout_id", "scenario_id", "timeslot"],
            how="left",
        )["y"].values)
        rho_r = _safe_spearman(vals, y)
        extra_cum[col] = {"raw": rho_r, "cum": rho_c, "delta": rho_c - rho_r}
        print(f"  {col:25s} raw={rho_r:+.4f} cum={rho_c:+.4f} delta={rho_c - rho_r:+.4f}")

    # --- plot ---
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(curve_df["timeslot"], curve_df["y_mean"], marker="o",
                 label="overall", color="k", lw=2)
    for t in ["narrow", "grid", "hybrid", "hub_spoke"]:
        if f"y_mean_{t}" in curve_df.columns:
            axes[0].plot(curve_df["timeslot"], curve_df[f"y_mean_{t}"],
                         marker="o", label=t, lw=1.2, alpha=0.8)
    axes[0].set_xlabel("timeslot")
    axes[0].set_ylabel("y mean")
    axes[0].set_title("Target mean trajectory within scenario")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    axes[1].hist(traj["spearman_y_ts"].values, bins=40, color="tab:blue", alpha=0.8)
    axes[1].axvline(0, color="k", lw=0.8)
    axes[1].axvline(0.5, color="r", lw=0.8, ls="--", label="+0.5")
    axes[1].axvline(-0.5, color="r", lw=0.8, ls="--")
    axes[1].set_xlabel("Spearman(y, timeslot) per scenario")
    axes[1].set_ylabel("count")
    axes[1].set_title(f"monotonic scenarios: >0.5={monotonic_frac:.2%}")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "trajectory_curve.png", dpi=100)
    plt.close(fig)

    # --- decision ---
    best_cum_delta = max(
        [extra_cum[c]["delta"] for c in extra_cum] + [rho_cum - rho_raw]
    )
    passes_cum = best_cum_delta >= 0.05
    passes_monotonic = strong_monotonic_frac >= 0.30
    verdict = (
        "PROMOTE" if (passes_cum or passes_monotonic) else "WEAK"
    )

    lines = [
        "Section 6 summary - Timeslot trajectory",
        "",
        f"per-scenario rho(y, timeslot):",
        f"  fraction >+0.5: {monotonic_frac:.4f}",
        f"  fraction >+0.8: {strong_monotonic_frac:.4f}",
        f"  fraction <-0.5: {neg_monotonic_frac:.4f}",
        "",
        "mean rho(y,ts) by layout_type:",
    ]
    for t, v in rho_mean_by_type.items():
        lines.append(f"  {t:10s}: {v:+.4f}")

    lines += [
        "",
        f"Spearman(order_inflow_15m, y) = {rho_raw:+.5f}",
        f"Spearman(order_cumsum, y)     = {rho_cum:+.5f}",
        f"  delta = {rho_cum - rho_raw:+.5f}",
        "",
        "other cumulative feature correlations:",
    ]
    for c, info in extra_cum.items():
        lines.append(
            f"  {c:25s} raw={info['raw']:+.4f} cum={info['cum']:+.4f} "
            f"delta={info['delta']:+.4f}"
        )
    lines += [
        "",
        "argmax timeslot distribution (count per ts):",
        argmax_counts.to_string(),
        "",
        f"DECISION: {verdict}",
        f"  best_cum_delta  = {best_cum_delta:+.4f}  (threshold +0.05)",
        f"  strong_monotonic_frac = {strong_monotonic_frac:.4f}  (threshold 0.30)",
    ]
    (OUT / "traj_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    mark_done(
        6,
        f"verdict={verdict} best_cum={best_cum_delta:+.4f} mono>0.8={strong_monotonic_frac:.4f}",
    )
    print(f"[OK] section 6 complete - verdict: {verdict}")
    return dict(
        verdict=verdict,
        strong_monotonic_frac=strong_monotonic_frac,
        best_cum_delta=best_cum_delta,
    )


# ==================================================================
# entry point
# ==================================================================
if __name__ == "__main__":
    section = sys.argv[1] if len(sys.argv) > 1 else "0"

    if section == "0":
        section_0()
    elif section == "1":
        section_1()
    elif section == "2":
        section_2()
    elif section == "3":
        section_3()
    elif section == "4":
        section_4()
    elif section == "5":
        section_5()
    elif section == "6":
        section_6()
    else:
        raise SystemExit(f"section {section} not implemented yet")
