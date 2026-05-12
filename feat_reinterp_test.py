"""
Feature reinterpretation test:
  Hypothesis: lag/rolling/cummean features are noise (ts_idx not temporal)
  Option A: Strip lag/rolling/cummean → LGB with scenario-aggregates only
  Option B: Re-sort rows by shift_hour within scenario, recompute lag/rolling

Compare single v23 LGB_Huber OOF MAE:
  - Original (8.61185)
  - Option A (no temporal features)
  - Option B (temporal-sorted features)
"""
import os
import sys
import json
import pickle
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

ROOT = r"c:/Users/user/Desktop/데이콘 4월"
OUT = os.path.join(ROOT, "results", "feat_reinterp")
os.makedirs(OUT, exist_ok=True)


def main():
    print("=" * 60, flush=True)
    print("Feature Reinterpretation Test", flush=True)
    print("=" * 60, flush=True)

    train = (
        pd.read_csv(os.path.join(ROOT, "train.csv"))
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

    # ─── Reference: v23s42_LGB_Huber ───
    with open(os.path.join(ROOT, "results", "v23_seed42.pkl"), "rb") as f:
        v23 = pickle.load(f)
    old_oof = v23["oofs"]["LGB_Huber"]
    old_mae = float(mean_absolute_error(y, old_oof))
    print(f"Reference v23s42_LGB_Huber OOF: {old_mae:.5f}", flush=True)

    # ─── Identify temporal features ───
    temporal_suffixes = ['_lag1', '_lag2', '_rmean3', '_rstd3', '_rmean5',
                         '_cummean', '_lead1', '_lead2', '_diff1', '_diff_lead1']
    temporal_feats = [c for c in feat_cols if any(c.endswith(s) for s in temporal_suffixes)]
    non_temporal_feats = [c for c in feat_cols if c not in temporal_feats]
    print(f"\nv23 total features: {len(feat_cols)}", flush=True)
    print(f"  Temporal (lag/rolling/cummean/lead/diff): {len(temporal_feats)}", flush=True)
    print(f"  Non-temporal (sc_*, interactions, etc.): {len(non_temporal_feats)}", flush=True)
    print(f"  Temporal samples: {temporal_feats[:5]}", flush=True)

    # ─── Train LGB_Huber per config ───
    def train_lgb(X_train, X_test, name):
        oof = np.zeros(len(y), dtype=np.float64)
        test_preds = []
        for f in range(5):
            val_mask = fold_idx == f
            tr_mask = ~val_mask
            m = lgb.LGBMRegressor(
                objective="huber", n_estimators=5000, learning_rate=0.03,
                num_leaves=63, max_depth=8, min_child_samples=50,
                subsample=0.7, colsample_bytree=0.7,
                reg_alpha=1.0, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1,
            )
            m.fit(X_train[tr_mask], y_log[tr_mask],
                  eval_set=[(X_train[val_mask], y_log[val_mask])],
                  callbacks=[lgb.early_stopping(200, verbose=False)])
            oof[val_mask] = np.expm1(m.predict(X_train[val_mask]))
            test_preds.append(np.expm1(m.predict(X_test)))
            fold_mae = np.mean(np.abs(oof[val_mask] - y[val_mask]))
            print(f"  [{name}] fold {f}: best_iter={m.best_iteration_} val MAE={fold_mae:.4f}", flush=True)
        test = np.mean(np.stack(test_preds, axis=0), axis=0)
        mae = float(mean_absolute_error(y, oof))
        return oof, test, mae

    # ─── Option A: non-temporal only ───
    print(f"\n=== Option A: Non-temporal features only ({len(non_temporal_feats)} features) ===", flush=True)
    X_A = train_fe[non_temporal_feats].values
    X_A_test = test_fe[non_temporal_feats].values
    oof_A, test_A, mae_A = train_lgb(X_A, X_A_test, "A_no_temporal")

    # Compare
    corr_A = float(np.corrcoef(y - old_oof, y - oof_A)[0, 1])
    print(f"\nOption A OOF MAE: {mae_A:.5f}  (vs v23 old {old_mae:.5f}, delta {mae_A - old_mae:+.5f})", flush=True)
    print(f"Option A corr with v23 old: {corr_A:.4f}", flush=True)

    # ─── Option B: re-sort by shift_hour within scenario, recompute lag/rolling ───
    print(f"\n=== Option B: Temporal-sorted lag/rolling ===", flush=True)
    # Build new temporal features using shift_hour-sorted order
    train_sorted = train.copy()
    train_sorted['_orig_idx'] = train_sorted.index
    # Sort: layout, scenario, then shift_hour (NaN last)
    train_sorted['_shift_sort'] = train_sorted['shift_hour'].fillna(99)
    train_sorted = train_sorted.sort_values(['layout_id', 'scenario_id', '_shift_sort']).reset_index(drop=True)

    key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'battery_mean', 'fault_count_15m', 'blocked_path_15m',
                'pack_utilization', 'charge_queue_length']
    new_feats = {}
    for col in key_cols:
        if col not in train_sorted.columns:
            continue
        g = train_sorted.groupby(['layout_id', 'scenario_id'])[col]
        new_feats[f'{col}_tlag1'] = g.shift(1)
        new_feats[f'{col}_tlag2'] = g.shift(2)
        new_feats[f'{col}_tdiff1'] = train_sorted[col] - new_feats[f'{col}_tlag1']
        new_feats[f'{col}_trmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        new_feats[f'{col}_trstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        new_feats[f'{col}_trmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        new_feats[f'{col}_tcummean'] = g.transform(lambda x: x.expanding().mean())
        new_feats[f'{col}_tlead1'] = g.shift(-1)
        new_feats[f'{col}_tlead2'] = g.shift(-2)

    # Restore original order (by _orig_idx)
    for k, v in new_feats.items():
        train_sorted[k] = v
    train_back = train_sorted.sort_values('_orig_idx').reset_index(drop=True)

    # Same for test
    test_copy = pd.read_csv(os.path.join(ROOT, "test.csv")).sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
    test_copy['_orig_idx'] = test_copy.index
    test_copy['_shift_sort'] = test_copy['shift_hour'].fillna(99)
    test_sorted = test_copy.sort_values(['layout_id', 'scenario_id', '_shift_sort']).reset_index(drop=True)

    new_test_feats = {}
    for col in key_cols:
        if col not in test_sorted.columns:
            continue
        g = test_sorted.groupby(['layout_id', 'scenario_id'])[col]
        new_test_feats[f'{col}_tlag1'] = g.shift(1)
        new_test_feats[f'{col}_tlag2'] = g.shift(2)
        new_test_feats[f'{col}_tdiff1'] = test_sorted[col] - new_test_feats[f'{col}_tlag1']
        new_test_feats[f'{col}_trmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        new_test_feats[f'{col}_trstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        new_test_feats[f'{col}_trmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        new_test_feats[f'{col}_tcummean'] = g.transform(lambda x: x.expanding().mean())
        new_test_feats[f'{col}_tlead1'] = g.shift(-1)
        new_test_feats[f'{col}_tlead2'] = g.shift(-2)

    for k, v in new_test_feats.items():
        test_sorted[k] = v
    test_back = test_sorted.sort_values('_orig_idx').reset_index(drop=True)

    new_t_cols = list(new_feats.keys())
    print(f"  New temporal-sorted features: {len(new_t_cols)}", flush=True)

    # Option B input: non-temporal v23 + new temporal-sorted
    X_B = np.hstack([
        train_fe[non_temporal_feats].values,
        train_back[new_t_cols].values,
    ])
    X_B_test = np.hstack([
        test_fe[non_temporal_feats].values,
        test_back[new_t_cols].values,
    ])
    # NaN fill (shouldn't be needed but just in case)
    X_B = np.where(np.isnan(X_B), 0, X_B)
    X_B_test = np.where(np.isnan(X_B_test), 0, X_B_test)

    print(f"  Combined X: {X_B.shape}", flush=True)
    oof_B, test_B, mae_B = train_lgb(X_B, X_B_test, "B_time_sorted")

    corr_B = float(np.corrcoef(y - old_oof, y - oof_B)[0, 1])
    print(f"\nOption B OOF MAE: {mae_B:.5f}  (vs v23 old {old_mae:.5f}, delta {mae_B - old_mae:+.5f})", flush=True)
    print(f"Option B corr with v23 old: {corr_B:.4f}", flush=True)

    # ─── Summary ───
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        mega = pickle.load(f)
    mega_oof = mega["meta_avg_oof"]
    mega_mae = float(mean_absolute_error(y, mega_oof))

    print(f"\n{'='*60}", flush=True)
    print(f"Summary:", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  v23 original (149 feat):      {old_mae:.5f}", flush=True)
    print(f"  Option A (non-temporal 57+):  {mae_A:.5f}  delta {mae_A - old_mae:+.5f}", flush=True)
    print(f"  Option B (time-sorted 57+72): {mae_B:.5f}  delta {mae_B - old_mae:+.5f}", flush=True)
    print(f"  mega33 baseline:              {mega_mae:.5f}", flush=True)

    # Blend with mega33
    for name, oof in [("A", oof_A), ("B", oof_B)]:
        corr_m = float(np.corrcoef(y - mega_oof, y - oof)[0, 1])
        best_w = 0; best_m = mega_mae
        for w in np.linspace(0, 0.3, 31):
            pred = (1-w) * mega_oof + w * oof
            m = mean_absolute_error(y, pred)
            if m < best_m: best_m = m; best_w = w
        print(f"  Option {name} + mega33 blend: w={best_w:.3f} MAE={best_m:.5f} delta={best_m-mega_mae:+.5f} corr(mega)={corr_m:.4f}", flush=True)

    np.save(os.path.join(OUT, "option_A_oof.npy"), oof_A)
    np.save(os.path.join(OUT, "option_A_test.npy"), test_A)
    np.save(os.path.join(OUT, "option_B_oof.npy"), oof_B)
    np.save(os.path.join(OUT, "option_B_test.npy"), test_B)

    summary = dict(
        v23_old_mae=old_mae,
        option_A_mae=mae_A, option_A_corr_v23=corr_A,
        option_B_mae=mae_B, option_B_corr_v23=corr_B,
        mega_mae=mega_mae,
        n_temporal_orig=len(temporal_feats),
        n_non_temporal=len(non_temporal_feats),
    )
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
