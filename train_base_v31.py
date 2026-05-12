"""
Train LGB + XGB + CB on v31 features (335 feats) for mega33 stack upgrade.
Output: results/base_v31/lgb_v31_{oof,test}.npy  (same for xgb, cb)
Fold order matches existing mega33 (layout_id GroupKFold, ls-sorted).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT = 'results/base_v31'
os.makedirs(OUT, exist_ok=True)
TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5

print("Loading v31 FE cache...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
feat_cols = fe_v31['feat_cols']
print(f"  v31 feat_cols: {len(feat_cols)}", flush=True)

train_fe = fe_v31['train_fe']  # ls-sorted
test_fe  = fe_v31['test_fe']   # ls-sorted

y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
del fe_v31; gc.collect()

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

# Save fold assignments for later use
fold_ids = np.zeros(len(y), dtype=int)
for f_i, (_, val_idx) in enumerate(folds):
    fold_ids[val_idx] = f_i
np.save(f'{OUT}/fold_ids.npy', fold_ids)

# ─── LGB ───────────────────────────────────────────────────────
print("\n[1] LGB_v31 (MAE)...", flush=True)
LGB_PARAMS = dict(objective='mae', n_estimators=3000, learning_rate=0.03,
                  num_leaves=63, max_depth=6, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=0.5,
                  random_state=42, verbose=-1, n_jobs=-1)
oof_lgb = np.zeros(len(y)); test_lgb = np.zeros(len(X_te))
for f_i, (tr_idx, val_idx) in enumerate(folds):
    t0 = time.time()
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X_tr[tr_idx], y_log[tr_idx],
          eval_set=[(X_tr[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_lgb[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
    test_lgb += np.expm1(m.predict(X_te)) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_lgb[val_idx])
    print(f"  fold{f_i+1}: mae={mae:.4f} it={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
oof_lgb = np.clip(oof_lgb, 0, None); test_lgb = np.clip(test_lgb, 0, None)
np.save(f'{OUT}/lgb_v31_oof.npy', oof_lgb); np.save(f'{OUT}/lgb_v31_test.npy', test_lgb)
print(f"  LGB_v31 OOF MAE: {mean_absolute_error(y, oof_lgb):.5f}", flush=True)

# ─── XGB ───────────────────────────────────────────────────────
print("\n[2] XGB_v31 (MAE)...", flush=True)
XGB_PARAMS = dict(objective='reg:absoluteerror', n_estimators=3000, learning_rate=0.03,
                  max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=0.5, tree_method='hist',
                  random_state=42, verbosity=0, n_jobs=4, early_stopping_rounds=100)
oof_xgb = np.zeros(len(y)); test_xgb = np.zeros(len(X_te))
for f_i, (tr_idx, val_idx) in enumerate(folds):
    t0 = time.time()
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_tr[tr_idx], y_log[tr_idx],
          eval_set=[(X_tr[val_idx], y_log[val_idx])], verbose=False)
    oof_xgb[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
    test_xgb += np.expm1(m.predict(X_te)) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_xgb[val_idx])
    print(f"  fold{f_i+1}: mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
oof_xgb = np.clip(oof_xgb, 0, None); test_xgb = np.clip(test_xgb, 0, None)
np.save(f'{OUT}/xgb_v31_oof.npy', oof_xgb); np.save(f'{OUT}/xgb_v31_test.npy', test_xgb)
print(f"  XGB_v31 OOF MAE: {mean_absolute_error(y, oof_xgb):.5f}", flush=True)

# ─── CatBoost ─────────────────────────────────────────────────
print("\n[3] CB_v31 (MAE)...", flush=True)
oof_cb = np.zeros(len(y)); test_cb = np.zeros(len(X_te))
for f_i, (tr_idx, val_idx) in enumerate(folds):
    t0 = time.time()
    m = CatBoostRegressor(loss_function='MAE', iterations=3000, learning_rate=0.03,
                          depth=6, min_data_in_leaf=50, subsample=0.8, rsm=0.8,
                          l2_leaf_reg=3.0, random_seed=42, verbose=0,
                          early_stopping_rounds=100, task_type='CPU', thread_count=4)
    m.fit(X_tr[tr_idx], y_log[tr_idx],
          eval_set=(X_tr[val_idx], y_log[val_idx]), use_best_model=True)
    oof_cb[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
    test_cb += np.expm1(m.predict(X_te)) / N_SPLITS
    mae = mean_absolute_error(y[val_idx], oof_cb[val_idx])
    print(f"  fold{f_i+1}: mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
oof_cb = np.clip(oof_cb, 0, None); test_cb = np.clip(test_cb, 0, None)
np.save(f'{OUT}/cb_v31_oof.npy', oof_cb); np.save(f'{OUT}/cb_v31_test.npy', test_cb)
print(f"  CB_v31 OOF MAE: {mean_absolute_error(y, oof_cb):.5f}", flush=True)

# ─── Summary ──────────────────────────────────────────────────
print("\n=== v31 Base Models Summary ===")
with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)
mega33 = d['meta_avg_oof']
print(f"  mega33 OOF: {mean_absolute_error(y, mega33):.5f}")
print(f"  LGB_v31:   {mean_absolute_error(y, oof_lgb):.5f}  corr(mega33)={np.corrcoef(mega33, oof_lgb)[0,1]:.4f}")
print(f"  XGB_v31:   {mean_absolute_error(y, oof_xgb):.5f}  corr(mega33)={np.corrcoef(mega33, oof_xgb)[0,1]:.4f}")
print(f"  CB_v31:    {mean_absolute_error(y, oof_cb):.5f}  corr(mega33)={np.corrcoef(mega33, oof_cb)[0,1]:.4f}")
print("Done.")
