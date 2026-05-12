"""
Framework G: 4-dimensional paradigm shift.

1. Layout clustering (KMeans on layout_info, K=8)
2. Per-cluster specialist models
3. Hurdle-Magnitude-Extreme target decomposition:
   h = P(y > 5 | x)  [hurdle]
   m = E[y | y > 5, x]  [magnitude, on filtered subset]
   e = P(y > 50 | x)  [extreme flag]

   final pred options:
     A. hurdle * magnitude (mean-like)
     B. LGB meta over [h, m, e, raw x] → MAE
4. Test cluster assignment via layout_info proximity

Compare:
  - Per-cluster Framework G OOF vs mega33 8.39895
  - Blend G + mega33
  - Check corr axis
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "framework_g")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Framework G: Layout-Cluster × Hurdle-Magnitude × Test-Aware", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    test = (
        pd.read_csv(os.path.join(ROOT, "test.csv"))
        .sort_values(["layout_id", "scenario_id"])
        .reset_index(drop=True)
    )
    y = train["avg_delay_minutes_next_30m"].values.astype(np.float64)
    y_log = np.log1p(np.clip(y, 0, None))
    fold_idx = np.load(os.path.join(ROOT, "results", "eda_v30", "fold_idx.npy"))

    with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
        d = pickle.load(f)
    train_fe = d["train_fe"].reset_index(drop=True)
    feat_cols = list(d["feat_cols"])
    with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
        test_fe = pickle.load(f).reset_index(drop=True)

    X_train = train_fe[feat_cols].values
    X_test = test_fe[feat_cols].values

    # ─── Step 1: Layout clustering ───
    print("\n[Step 1] Layout clustering on layout_info (K=8)", flush=True)
    layout_info = pd.read_csv(os.path.join(ROOT, "layout_info.csv"))
    # layout_type one-hot
    lt_dummies = pd.get_dummies(layout_info["layout_type"], prefix="lt")
    layout_numeric_cols = [c for c in layout_info.columns if c not in ["layout_id", "layout_type"]]
    li_feat = pd.concat([layout_info[["layout_id"] + layout_numeric_cols], lt_dummies], axis=1)

    # Fill NaN with median
    for c in layout_numeric_cols:
        li_feat[c] = li_feat[c].fillna(li_feat[c].median())

    cluster_feat_cols = layout_numeric_cols + list(lt_dummies.columns)
    li_X = li_feat[cluster_feat_cols].values.astype(np.float64)
    scaler_li = StandardScaler().fit(li_X)
    li_X_std = scaler_li.transform(li_X)

    K = 8
    km = KMeans(n_clusters=K, random_state=42, n_init=10).fit(li_X_std)
    li_feat["cluster"] = km.labels_
    layout_cluster = dict(zip(li_feat["layout_id"], li_feat["cluster"]))
    # Assign each train/test row to cluster
    train["cluster"] = train["layout_id"].map(layout_cluster)
    test["cluster"] = test["layout_id"].map(layout_cluster)

    # Cluster balance
    print(f"  Cluster sizes (# layouts):", flush=True)
    for k in range(K):
        n_layouts = int((li_feat["cluster"] == k).sum())
        n_train_layouts = int(((li_feat["cluster"] == k) & li_feat["layout_id"].isin(train["layout_id"].unique())).sum())
        n_test_layouts = int(((li_feat["cluster"] == k) & li_feat["layout_id"].isin(test["layout_id"].unique())).sum())
        n_rows_tr = int((train["cluster"] == k).sum())
        n_rows_te = int((test["cluster"] == k).sum())
        print(f"    cluster {k}: {n_layouts} layouts ({n_train_layouts} tr, {n_test_layouts} te), "
              f"{n_rows_tr} tr rows, {n_rows_te} te rows", flush=True)

    # Sanity check: all test clusters have training data
    test_clusters_without_tr = []
    for k in range(K):
        n_tr = int((train["cluster"] == k).sum())
        n_te = int((test["cluster"] == k).sum())
        if n_te > 0 and n_tr == 0:
            test_clusters_without_tr.append(k)
    if test_clusters_without_tr:
        print(f"  ⚠️ Clusters with test but no train: {test_clusters_without_tr}", flush=True)
        return

    # ─── Step 2: Hurdle threshold analysis ───
    HURDLE_THRESH = 5.0  # y > 5 as "non-low" regime
    EXTREME_THRESH = 50.0  # y > 50 as "extreme"
    hurdle_label = (y > HURDLE_THRESH).astype(int)
    extreme_label = (y > EXTREME_THRESH).astype(int)
    print(f"\n[Step 2] Target decomposition", flush=True)
    print(f"  Hurdle (y > {HURDLE_THRESH}): {hurdle_label.mean()*100:.1f}% positive", flush=True)
    print(f"  Extreme (y > {EXTREME_THRESH}): {extreme_label.mean()*100:.1f}% positive", flush=True)

    # ─── Step 3: Per-cluster × per-fold training ───
    print("\n[Step 3] Per-cluster models (hurdle + magnitude + extreme)", flush=True)
    hurdle_oof = np.zeros(len(y), dtype=np.float64)
    magn_oof = np.zeros(len(y), dtype=np.float64)
    extreme_oof = np.zeros(len(y), dtype=np.float64)
    hurdle_test = np.zeros(len(test), dtype=np.float64)
    magn_test = np.zeros(len(test), dtype=np.float64)
    extreme_test = np.zeros(len(test), dtype=np.float64)

    for k in range(K):
        tr_cluster_mask = train["cluster"].values == k
        te_cluster_mask = test["cluster"].values == k
        if tr_cluster_mask.sum() < 1000:
            print(f"\n  Cluster {k}: too few train rows ({tr_cluster_mask.sum()}). Fallback to global.", flush=True)
            # Use global v23 predictions (approximate fallback)
            tr_cluster_mask = np.ones(len(y), dtype=bool)  # use all
        print(f"\n  Cluster {k}: train {int(tr_cluster_mask.sum())}, test {int(te_cluster_mask.sum())}", flush=True)

        # Per-fold training within cluster
        for f in range(5):
            val_mask_global = fold_idx == f
            tr_mask_global = ~val_mask_global

            # Train set: cluster AND non-val fold
            mask_fit = tr_cluster_mask & tr_mask_global
            mask_val = tr_cluster_mask & val_mask_global

            if mask_fit.sum() < 500 or mask_val.sum() == 0:
                # Not enough training data in cluster+fold. Use global fallback.
                mask_fit = tr_mask_global
                mask_val_full = val_mask_global & te_cluster_mask
                if mask_val_full.sum() == 0:
                    mask_val = val_mask_global  # all val
                else:
                    mask_val = val_mask_global

            X_fit = X_train[mask_fit]
            X_val = X_train[val_mask_global]

            # Hurdle LGB
            hurd_clf = lgb.LGBMClassifier(
                objective="binary", n_estimators=500, learning_rate=0.05,
                num_leaves=31, max_depth=6, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            hurd_clf.fit(X_fit, hurdle_label[mask_fit],
                         eval_set=[(X_val, hurdle_label[val_mask_global])],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
            # Only update OOF for rows in this cluster
            cluster_val_idx = np.where(val_mask_global & (train["cluster"].values == k))[0]
            if len(cluster_val_idx) > 0:
                hurdle_oof[cluster_val_idx] = hurd_clf.predict_proba(X_train[cluster_val_idx])[:, 1]

            # Extreme LGB
            ext_clf = lgb.LGBMClassifier(
                objective="binary", n_estimators=500, learning_rate=0.05,
                num_leaves=31, max_depth=6, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                scale_pos_weight=10,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            ext_clf.fit(X_fit, extreme_label[mask_fit],
                        eval_set=[(X_val, extreme_label[val_mask_global])],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
            if len(cluster_val_idx) > 0:
                extreme_oof[cluster_val_idx] = ext_clf.predict_proba(X_train[cluster_val_idx])[:, 1]

            # Magnitude LGB (conditional on y > HURDLE_THRESH during training, predict on all)
            mag_mask = mask_fit & (y > HURDLE_THRESH)
            if mag_mask.sum() < 100:
                mag_mask = mask_fit  # fallback: all rows
            mag_reg = lgb.LGBMRegressor(
                objective="huber", n_estimators=1000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            mag_reg.fit(X_train[mag_mask], y_log[mag_mask],
                        callbacks=[lgb.log_evaluation(0)])
            if len(cluster_val_idx) > 0:
                magn_oof[cluster_val_idx] = np.expm1(mag_reg.predict(X_train[cluster_val_idx]))

            # Test predictions (cluster-specific test rows)
            cluster_te_idx = np.where(te_cluster_mask)[0]
            if len(cluster_te_idx) > 0:
                hurdle_test[cluster_te_idx] = hurdle_test[cluster_te_idx] + hurd_clf.predict_proba(X_test[cluster_te_idx])[:, 1] / 5
                extreme_test[cluster_te_idx] = extreme_test[cluster_te_idx] + ext_clf.predict_proba(X_test[cluster_te_idx])[:, 1] / 5
                magn_test[cluster_te_idx] = magn_test[cluster_te_idx] + np.expm1(mag_reg.predict(X_test[cluster_te_idx])) / 5

            print(f"    fold {f}: fit_n={mask_fit.sum()}, val_n={cluster_val_idx.size}", flush=True)

    hurdle_oof = np.clip(hurdle_oof, 0, 1)
    extreme_oof = np.clip(extreme_oof, 0, 1)
    magn_oof = np.clip(magn_oof, 0, None)

    # Composition A: hurdle * magn (mean-like)
    pred_A = hurdle_oof * magn_oof
    pred_A_test = np.clip(hurdle_test, 0, 1) * np.clip(magn_test, 0, None)

    mae_A = float(mean_absolute_error(y, pred_A))
    print(f"\n[Result A] hurdle*magn OOF MAE: {mae_A:.5f}", flush=True)

    # Composition B: meta LGB over [h, e, magn, X...]
    print("\n[Step 4] Meta LGB over [h, e, magn] + v23", flush=True)
    meta_train = np.column_stack([hurdle_oof, extreme_oof, magn_oof, X_train])
    meta_test = np.column_stack([np.clip(hurdle_test,0,1), np.clip(extreme_test,0,1), np.clip(magn_test,0,None), X_test])
    meta_oof = np.zeros(len(y))
    meta_test_pred = np.zeros(len(test))
    for f in range(5):
        val_mask = fold_idx == f
        tr_mask = ~val_mask
        m = lgb.LGBMRegressor(
            objective="mae", n_estimators=3000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
        m.fit(meta_train[tr_mask], y_log[tr_mask],
              eval_set=[(meta_train[val_mask], y_log[val_mask])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        meta_oof[val_mask] = np.expm1(m.predict(meta_train[val_mask]))
        meta_test_pred += np.expm1(m.predict(meta_test)) / 5
        fold_mae = np.mean(np.abs(meta_oof[val_mask] - y[val_mask]))
        print(f"  meta fold {f}: best_iter={m.best_iteration_} val MAE={fold_mae:.4f}", flush=True)

    meta_oof = np.clip(meta_oof, 0, None)
    meta_test_pred = np.clip(meta_test_pred, 0, None)
    mae_B = float(mean_absolute_error(y, meta_oof))

    # ─── Compare ───
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))

    corr_A = float(np.corrcoef(y - mega_oof, y - pred_A)[0, 1])
    corr_B = float(np.corrcoef(y - mega_oof, y - meta_oof)[0, 1])

    print("\n" + "=" * 60, flush=True)
    print("Framework G Final Results", flush=True)
    print("=" * 60, flush=True)
    print(f"  mega33 baseline:            {mega_mae:.5f}", flush=True)
    print(f"  Composition A (h*m):        {mae_A:.5f}  delta {mae_A - mega_mae:+.5f}  corr {corr_A:.4f}", flush=True)
    print(f"  Composition B (meta LGB):   {mae_B:.5f}  delta {mae_B - mega_mae:+.5f}  corr {corr_B:.4f}", flush=True)

    # Blend tests
    for name, oof, test_p in [("A_h*m", pred_A, pred_A_test), ("B_meta", meta_oof, meta_test_pred)]:
        best_w = 0; best_m = mega_mae
        for w in np.linspace(0, 0.5, 51):
            pred = (1 - w) * mega_oof + w * oof
            m = float(np.mean(np.abs(y - pred)))
            if m < best_m: best_m = m; best_w = w
        print(f"  Blend ({name}): w={best_w:.3f} MAE={best_m:.5f} delta={best_m-mega_mae:+.5f}", flush=True)

    # Save
    np.save(os.path.join(OUT, "hurdle_oof.npy"), hurdle_oof)
    np.save(os.path.join(OUT, "magn_oof.npy"), magn_oof)
    np.save(os.path.join(OUT, "extreme_oof.npy"), extreme_oof)
    np.save(os.path.join(OUT, "pred_A_oof.npy"), pred_A)
    np.save(os.path.join(OUT, "pred_A_test.npy"), pred_A_test)
    np.save(os.path.join(OUT, "meta_B_oof.npy"), meta_oof)
    np.save(os.path.join(OUT, "meta_B_test.npy"), meta_test_pred)

    summary = dict(
        K=K, hurdle_thresh=HURDLE_THRESH, extreme_thresh=EXTREME_THRESH,
        mega_mae=mega_mae,
        mae_A=mae_A, corr_A=corr_A,
        mae_B=mae_B, corr_B=corr_B,
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {OUT}", flush=True)


if __name__ == "__main__":
    main()
