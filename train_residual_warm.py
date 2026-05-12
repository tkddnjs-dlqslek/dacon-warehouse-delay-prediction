"""
Residual Sequential v2: Layout-aware warm start.

Core fix: instead of initializing r_0 = 0 (cold start → error accumulates),
we use the layout-specific r_0 MEAN from training fold as initialization.
At position 0 training: also replace true r_0 with layout_mean (matching test dist).

Why this matters:
  - residual autocorr (lag1) = 0.7646 → strong signal
  - r_true mean=3.13, std=20.9 → cold start error = ~20.9 (huge)
  - Layout-specific mean: if std(layout_means) >> 0, warm start helps
  - Training distribution (layout_mean as r_0) matches test distribution
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

print("Loading FE...", flush=True)
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

print("Loading mega33...", flush=True)
with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# ── Residuals ──
r_true = y_true - mega_oof_id
train_raw['r_true'] = r_true
train_raw['layout_id_int'] = train_raw['layout_id']

# Global r_0 stats
train_pos0_mask = train_raw['row_in_sc'] == 0
r0_vals = r_true[train_pos0_mask]
global_r0_mean = r0_vals.mean()
print(f"Global r_0 mean={global_r0_mean:.4f}, std={r0_vals.std():.4f}", flush=True)

# Layout-specific r_0 means (ALL training, for test init)
layout_r0_map = train_raw[train_pos0_mask].groupby('layout_id')['r_true'].mean().to_dict()
layout_std = np.std(list(layout_r0_map.values()))
print(f"Layout r_0 std across layouts={layout_std:.4f}  "
      f"(useful if >> 0 → warm start helps)", flush=True)

# Lag features using TRUE residuals
train_raw['r_lag1'] = (train_raw
    .groupby(['layout_id','scenario_id'])['r_true']
    .shift(1))
train_raw['r_lag2'] = (train_raw
    .groupby(['layout_id','scenario_id'])['r_true']
    .shift(2))

X_train_base = fe_train_df[feat_cols].values
r_lag1_arr   = train_raw['r_lag1'].values
r_lag2_arr   = train_raw['r_lag2'].values
row_sc_arr   = train_raw['row_in_sc'].values
layout_arr   = train_raw['layout_id'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base   = X_test_base[feat_cols].values
test_layout   = test_raw['layout_id'].values
test_row_sc   = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_r_preds = np.full(len(train_raw), np.nan)
test_r_list = []

print("Training residual warm-start model...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    # Layout-specific r_0 mean from TRAINING FOLD only
    tr_pos0_mask = (row_sc_arr[tr_idx] == 0)
    tr_layout    = layout_arr[tr_idx]
    tr_r0        = r_true[tr_idx]

    fold_layout_r0 = {}
    for lid in np.unique(tr_layout[tr_pos0_mask]):
        mask = tr_layout[tr_pos0_mask] == lid
        fold_layout_r0[lid] = tr_r0[tr_pos0_mask][mask].mean()

    # Build training X: for pos=0, use layout_mean instead of NaN/0
    lag1_tr = r_lag1_arr[tr_idx].copy()
    lag2_tr = r_lag2_arr[tr_idx].copy()

    # Fill pos=0: lag1 → layout_mean_r0, lag2 → layout_mean_r0
    for i, orig_i in enumerate(tr_idx):
        if row_sc_arr[orig_i] == 0:
            lid = layout_arr[orig_i]
            lm  = fold_layout_r0.get(lid, global_r0_mean)
            lag1_tr[i] = lm
            lag2_tr[i] = lm
        elif row_sc_arr[orig_i] == 1:
            # lag2 is NaN (no t-2), fill with layout_mean
            if np.isnan(lag2_tr[i]):
                lid = layout_arr[orig_i]
                lag2_tr[i] = fold_layout_r0.get(lid, global_r0_mean)

    X_tr  = make_X(X_train_base[tr_idx], lag1_tr, lag2_tr, row_sc_arr[tr_idx])
    y_tr  = r_true[tr_idx]

    # Dummy val (true lag, for early stopping)
    lag1_val = r_lag1_arr[val_idx].copy()
    lag2_val = r_lag2_arr[val_idx].copy()
    val_lids = layout_arr[val_idx]
    for i, orig_i in enumerate(val_idx):
        rs = row_sc_arr[orig_i]
        lid = val_lids[i]
        lm = fold_layout_r0.get(lid, global_r0_mean)
        if rs == 0:
            lag1_val[i] = lm
            lag2_val[i] = lm
        elif rs == 1 and np.isnan(lag2_val[i]):
            lag2_val[i] = lm

    X_val_dummy = make_X(X_train_base[val_idx], lag1_val, lag2_val, row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_dummy, r_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)])

    # ── Sequential OOF ──
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    layout_vals    = train_raw['layout_id'].values[val_sorted]

    fold_oof_r = np.zeros(len(val_sorted))

    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        pos_layouts = layout_vals[pos_mask]

        if pos == 0:
            # Warm start: layout-specific r_0 mean
            lag1_fill = np.array([fold_layout_r0.get(lid, global_r0_mean)
                                  for lid in pos_layouts])
            lag2_fill = lag1_fill.copy()
        else:
            lag1_fill = fold_oof_r[row_in_sc_vals == (pos-1)]
            if pos >= 2:
                lag2_fill = fold_oof_r[row_in_sc_vals == (pos-2)]
            else:
                # pos=1: lag2 = layout_mean
                lag2_fill = np.array([fold_layout_r0.get(lid, global_r0_mean)
                                      for lid in pos_layouts])

        X_pos = make_X(X_train_base[pos_idx], lag1_fill, lag2_fill,
                       np.full(n_pos, pos))
        fold_oof_r[pos_mask] = model.predict(X_pos)

    oof_r_preds[val_sorted] = fold_oof_r
    oof_y_fold = np.maximum(0, mega_oof_id[val_sorted] + fold_oof_r)
    fold_mae_y = np.mean(np.abs(oof_y_fold - y_true[val_sorted]))
    fold_mae_r = np.mean(np.abs(fold_oof_r - r_true[val_sorted]))
    elapsed    = time.time() - t0
    print(f"Fold {fold_i+1}: r_MAE={fold_mae_r:.4f}  y_MAE={fold_mae_y:.4f} ({elapsed:.0f}s)", flush=True)

    # ── Sequential Test ──
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    test_layout_s = test_raw['layout_id'].values[test_sorted]

    test_r_sorted = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        pos_lids = test_layout_s[pos_mask]

        if pos == 0:
            l1 = np.array([layout_r0_map.get(lid, global_r0_mean) for lid in pos_lids])
            l2 = l1.copy()
        else:
            l1 = test_r_sorted[test_sorted[test_rsc_vals == (pos-1)]]
            if pos >= 2:
                l2 = test_r_sorted[test_sorted[test_rsc_vals == (pos-2)]]
            else:
                l2 = np.array([layout_r0_map.get(lid, global_r0_mean) for lid in pos_lids])

        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(n_pos, pos))
        test_r_sorted[pos_idx] = model.predict(X_pos)

    test_r_list.append(test_r_sorted)

# ── Final results ──
oof_y_preds = np.maximum(0, mega_oof_id + oof_r_preds)
oof_mae_y   = np.mean(np.abs(oof_y_preds - y_true))
mega_mae    = np.mean(np.abs(mega_oof_id - y_true))
res_corr    = np.corrcoef(oof_y_preds - y_true, mega_oof_id - y_true)[0,1]
threshold   = mega_mae + 1.25 * (1 - res_corr)

print(f"\n=== RESULTS ===", flush=True)
print(f"mega33 OOF MAE         = {mega_mae:.4f}", flush=True)
print(f"residual_warm OOF MAE  = {oof_mae_y:.4f}  (delta={oof_mae_y-mega_mae:.4f})", flush=True)
print(f"residual_corr          = {res_corr:.4f}", flush=True)
print(f"blend threshold        = {threshold:.4f}  → {'✓ MEETS' if oof_mae_y < threshold else '✗ FAILS'}", flush=True)
print(f"layout_r0 std          = {layout_std:.4f}", flush=True)

best_mae, best_w = 9999, 0
for w in np.arange(0, 1.01, 0.02):
    m = np.mean(np.abs(w*oof_y_preds + (1-w)*mega_oof_id - y_true))
    if m < best_mae:
        best_mae, best_w = m, w
print(f"Best blend w={best_w:.2f}: MAE={best_mae:.4f}  delta={best_mae-mega_mae:.4f}", flush=True)

os.makedirs('results/residual_warm', exist_ok=True)
np.save('results/residual_warm/oof_y.npy', oof_y_preds)
np.save('results/residual_warm/oof_r.npy', oof_r_preds)
test_r_avg = np.mean(test_r_list, axis=0)
test_y_avg = np.maximum(0, mega_test_id + test_r_avg)
np.save('results/residual_warm/test_r.npy', test_r_avg)
np.save('results/residual_warm/test_y.npy', test_y_avg)

with open('results/residual_warm/summary.json','w',encoding='utf-8') as f:
    json.dump({
        'mega33_mae': float(mega_mae),
        'warm_mae': float(oof_mae_y),
        'delta': float(oof_mae_y - mega_mae),
        'residual_corr': float(res_corr),
        'threshold': float(threshold),
        'meets_threshold': bool(oof_mae_y < threshold),
        'best_blend_w': float(best_w),
        'best_blend_mae': float(best_mae),
        'blend_delta': float(best_mae - mega_mae),
        'layout_r0_std': float(layout_std),
    }, f, indent=2)
print("Saved to results/residual_warm/", flush=True)
