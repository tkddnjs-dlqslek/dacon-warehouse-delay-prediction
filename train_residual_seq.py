"""
Residual Sequential Model: predict r_t = y_t - mega33_oof_t sequentially.

Key insight: by construction, residual_corr(this, mega33) is LOW — we predict
what mega33 GOT WRONG, not y_t itself.

- Training target: r_t = y_t - mega33_oof_t
- lag features: r_{t-1}, r_{t-2} (true residuals in train, sequential in OOF/test)
- OOF/Test: sequential starting from r_0 = 0 (global_mean residual ≈ 0)
- Final prediction: mega33 + pred_residuals

Math: residual_corr(pred_r - r_true, -r_true) → 0 as pred_r → r_true
Even noisy pred_r gives LOW residual_corr vs mega33.
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

# ── Residual target ──
r_true = y_true - mega_oof_id   # what mega33 got wrong
print(f"Residual stats: mean={r_true.mean():.4f} std={r_true.std():.4f} "
      f"autocorr_lag1={pd.Series(r_true[:25]).autocorr(1):.4f}", flush=True)

# compute within-scenario autocorrelation of residuals
train_raw['r_true'] = r_true
train_raw['r_lag1'] = (train_raw
    .groupby(['layout_id','scenario_id'])['r_true']
    .shift(1).fillna(0.0))   # initialize at 0
train_raw['r_lag2'] = (train_raw
    .groupby(['layout_id','scenario_id'])['r_true']
    .shift(2).fillna(0.0))

# measure autocorrelation on scenarios with pos>=1
sc_corr = train_raw[train_raw['row_in_sc']>=1][['r_true','r_lag1']].corr().iloc[0,1]
print(f"Within-scenario residual autocorr (lag1): {sc_corr:.4f}", flush=True)

X_train_base = fe_train_df[feat_cols].values
r_lag1_arr   = train_raw['r_lag1'].values
r_lag2_arr   = train_raw['r_lag2'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].values
test_row_sc = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

LAG_COLS = ['r_lag1', 'r_lag2', 'row_in_sc']

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_r_preds   = np.full(len(train_raw), np.nan)
test_r_list   = []

print("Training residual sequential model...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    # Training: use TRUE residual lag
    X_tr = make_X(X_train_base[tr_idx], r_lag1_arr[tr_idx],
                  r_lag2_arr[tr_idx], row_sc_arr[tr_idx])
    y_tr = r_true[tr_idx]   # TARGET IS RESIDUAL

    # Dummy val with true lag (for early stopping signal)
    X_val_dummy = make_X(X_train_base[val_idx], r_lag1_arr[val_idx],
                         r_lag2_arr[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_dummy, r_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)])

    # ── Sequential OOF on residuals ──
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values

    fold_oof_r = np.zeros(len(val_sorted))

    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()

        if pos == 0:
            lag1_fill = np.zeros(n_pos)   # r_0 init = 0
            lag2_fill = np.zeros(n_pos)
        else:
            lag1_fill = fold_oof_r[row_in_sc_vals == (pos-1)]
            lag2_fill = (fold_oof_r[row_in_sc_vals == (pos-2)]
                         if pos >= 2 else np.zeros(n_pos))

        X_pos = make_X(X_train_base[pos_idx], lag1_fill, lag2_fill,
                       np.full(n_pos, pos))
        fold_oof_r[pos_mask] = model.predict(X_pos)

    oof_r_preds[val_sorted] = fold_oof_r

    # OOF y predictions = mega33 + pred_residual
    oof_y_fold = mega_oof_id[val_sorted] + fold_oof_r
    oof_y_fold = np.maximum(0, oof_y_fold)
    fold_mae_y = np.mean(np.abs(oof_y_fold - y_true[val_sorted]))
    fold_mae_r = np.mean(np.abs(fold_oof_r - r_true[val_sorted]))
    elapsed    = time.time() - t0
    print(f"Fold {fold_i+1}: r_MAE={fold_mae_r:.4f}  y_MAE={fold_mae_y:.4f} ({elapsed:.0f}s)", flush=True)

    # ── Sequential Test (residual) ──
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values

    test_r_sorted = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()

        if pos == 0:
            l1 = np.zeros(n_pos)
            l2 = np.zeros(n_pos)
        else:
            l1 = test_r_sorted[test_sorted[test_rsc_vals == (pos-1)]]
            l2 = (test_r_sorted[test_sorted[test_rsc_vals == (pos-2)]]
                  if pos >= 2 else np.zeros(n_pos))

        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(n_pos, pos))
        test_r_sorted[pos_idx] = model.predict(X_pos)

    test_r_list.append(test_r_sorted)

# ── Final y predictions ──
oof_y_preds = np.maximum(0, mega_oof_id + oof_r_preds)

oof_mae_y  = np.mean(np.abs(oof_y_preds - y_true))
mega_mae   = np.mean(np.abs(mega_oof_id - y_true))
res_corr   = np.corrcoef(oof_y_preds - y_true, mega_oof_id - y_true)[0,1]
threshold  = mega_mae + 1.25 * (1 - res_corr)

print(f"\n=== RESULTS ===", flush=True)
print(f"mega33 OOF MAE         = {mega_mae:.4f}", flush=True)
print(f"residual_seq OOF MAE   = {oof_mae_y:.4f}  (delta={oof_mae_y-mega_mae:.4f})", flush=True)
print(f"residual_corr          = {res_corr:.4f}", flush=True)
print(f"blend threshold        = {threshold:.4f}  → {'✓ MEETS' if oof_mae_y < threshold else '✗ FAILS'}", flush=True)

# Blend grid search
best_mae, best_w = 9999, 0
for w in np.arange(0, 1.01, 0.02):
    m = np.mean(np.abs(w*oof_y_preds + (1-w)*mega_oof_id - y_true))
    if m < best_mae:
        best_mae, best_w = m, w
print(f"Best blend w={best_w:.2f}: MAE={best_mae:.4f}  delta={best_mae-mega_mae:.4f}", flush=True)

# Blend with current final blend
try:
    final_blend_oof = np.load('results/final_blend/final_oof.npy')
    final_mae = np.mean(np.abs(final_blend_oof - y_true))
    best_fb_mae, best_fb_w = 9999, 0
    for w in np.arange(0, 0.51, 0.02):
        m = np.mean(np.abs(w*oof_y_preds + (1-w)*final_blend_oof - y_true))
        if m < best_fb_mae:
            best_fb_mae, best_fb_w = m, w
    print(f"Final blend OOF MAE    = {final_mae:.4f}", flush=True)
    print(f"+ residual_seq: w={best_fb_w:.2f}: MAE={best_fb_mae:.4f}  "
          f"delta={best_fb_mae-final_mae:.4f}", flush=True)
except FileNotFoundError:
    print("(final_oof.npy not found — skipping final blend check)", flush=True)

os.makedirs('results/residual_seq', exist_ok=True)
np.save('results/residual_seq/oof_y.npy', oof_y_preds)
np.save('results/residual_seq/oof_r.npy', oof_r_preds)
test_r_avg  = np.mean(test_r_list, axis=0)
test_y_avg  = np.maximum(0, mega_test_id + test_r_avg)
np.save('results/residual_seq/test_r.npy', test_r_avg)
np.save('results/residual_seq/test_y.npy', test_y_avg)

with open('results/residual_seq/summary.json','w',encoding='utf-8') as f:
    json.dump({
        'mega33_mae': float(mega_mae),
        'residual_seq_mae': float(oof_mae_y),
        'delta': float(oof_mae_y - mega_mae),
        'residual_corr': float(res_corr),
        'threshold': float(threshold),
        'meets_threshold': bool(oof_mae_y < threshold),
        'best_blend_w': float(best_w),
        'best_blend_mae': float(best_mae),
        'best_blend_delta': float(best_mae - mega_mae),
        'sc_residual_autocorr': float(sc_corr),
    }, f, indent=2)
print("Saved to results/residual_seq/", flush=True)
