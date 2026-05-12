"""
mega34 base: retrain all 13 tree model variants on v31 features (335 feats).
Replaces v23/v24/v26 tree models with v31-powered equivalents.
Output: results/mega34_trees.pkl
Models:
  G1. LGB_Huber × seeds [42, 123, 2024]
  G2. XGB       × seeds [42, 123, 2024]
  G3. CatBoost  × seeds [42, 123, 2024]
  G4. LGB_Tuned (v26 Optuna params, Huber)
  G5. LGB_DART
  G6. LGB_sqrt  (sqrt target)
  G7. LGB_pow   (y^0.25 target)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, warnings, gc
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT = 'results/mega34'
os.makedirs(OUT, exist_ok=True)
TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5

print("Loading v31 FE cache...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
feat_cols = fe_v31['feat_cols']
print(f"  feat_cols: {len(feat_cols)}", flush=True)

train_fe = fe_v31['train_fe']  # ls-sorted
test_fe  = fe_v31['test_fe']

y = train_fe[TARGET].values.astype(np.float64)
y_log  = np.log1p(y)
y_sqrt = np.sqrt(y)
y_pow  = np.power(y, 0.25)

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
del fe_v31; gc.collect()

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

oofs  = {}
tests = {}

# ─── Helper ──────────────────────────────────────────────────────
def run_lgb(params, y_tr, inv_fn, name):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y_tr[tr_idx],
              eval_set=[(X_tr[val_idx], y_tr[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = inv_fn(m.predict(X_tr[val_idx]))
        te += inv_fn(m.predict(X_te)) / N_SPLITS
        mae = mean_absolute_error(y[val_idx], np.clip(oof[val_idx], 0, None))
        print(f"  fold{i+1}: mae={mae:.4f} it={m.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    oofs[name]  = oof
    tests[name] = te
    print(f"  [{name}] OOF MAE: {mean_absolute_error(y, oof):.5f}", flush=True)

def run_lgb_dart(params, name):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y_log[tr_idx])  # DART: no early stopping
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        te += np.expm1(m.predict(X_te)) / N_SPLITS
        mae = mean_absolute_error(y[val_idx], np.clip(oof[val_idx], 0, None))
        print(f"  fold{i+1}: mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    oofs[name]  = oof
    tests[name] = te
    print(f"  [{name}] OOF MAE: {mean_absolute_error(y, oof):.5f}", flush=True)

def run_xgb(seed, name):
    params = dict(objective='reg:absoluteerror', n_estimators=3000, learning_rate=0.03,
                  max_depth=6, min_child_weight=50, subsample=0.8, colsample_bytree=0.8,
                  reg_alpha=0.5, reg_lambda=0.5, tree_method='hist',
                  random_state=seed, verbosity=0, n_jobs=4, early_stopping_rounds=100)
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        m = xgb.XGBRegressor(**params)
        m.fit(X_tr[tr_idx], y_log[tr_idx],
              eval_set=[(X_tr[val_idx], y_log[val_idx])], verbose=False)
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        te += np.expm1(m.predict(X_te)) / N_SPLITS
        mae = mean_absolute_error(y[val_idx], np.clip(oof[val_idx], 0, None))
        print(f"  fold{i+1}: mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    oofs[name]  = oof
    tests[name] = te
    print(f"  [{name}] OOF MAE: {mean_absolute_error(y, oof):.5f}", flush=True)

def run_cb(seed, name):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        t0 = time.time()
        m = CatBoostRegressor(loss_function='MAE', iterations=3000, learning_rate=0.03,
                              depth=6, min_data_in_leaf=50, subsample=0.8, rsm=0.8,
                              l2_leaf_reg=3.0, random_seed=seed, verbose=0,
                              early_stopping_rounds=100, task_type='CPU', thread_count=4)
        m.fit(X_tr[tr_idx], y_log[tr_idx],
              eval_set=(X_tr[val_idx], y_log[val_idx]), use_best_model=True)
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        te += np.expm1(m.predict(X_te)) / N_SPLITS
        mae = mean_absolute_error(y[val_idx], np.clip(oof[val_idx], 0, None))
        print(f"  fold{i+1}: mae={mae:.4f} ({time.time()-t0:.0f}s)", flush=True)
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    oofs[name]  = oof
    tests[name] = te
    print(f"  [{name}] OOF MAE: {mean_absolute_error(y, oof):.5f}", flush=True)

# ─── G1: LGB_Huber × 3 seeds ─────────────────────────────────────
for seed in [42, 123, 2024]:
    print(f"\n[G1] LGB_Huber seed={seed}...", flush=True)
    params = dict(objective='huber', alpha=0.9, n_estimators=3000, learning_rate=0.03,
                  num_leaves=63, max_depth=6, min_child_samples=50,
                  subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
                  random_state=seed, verbose=-1, n_jobs=-1)
    run_lgb(params, y_log, np.expm1, f'lgb_huber_s{seed}')

# ─── G2: XGB × 3 seeds ───────────────────────────────────────────
for seed in [42, 123, 2024]:
    print(f"\n[G2] XGB seed={seed}...", flush=True)
    run_xgb(seed, f'xgb_s{seed}')

# ─── G3: CatBoost × 3 seeds ──────────────────────────────────────
for seed in [42, 123, 2024]:
    print(f"\n[G3] CatBoost seed={seed}...", flush=True)
    run_cb(seed, f'cb_s{seed}')

# ─── G4: LGB_Tuned (v26 Optuna params, Huber) ────────────────────
print("\n[G4] LGB_Tuned (v26 Optuna params)...", flush=True)
tuned = dict(objective='huber', alpha=0.9, n_estimators=3000,
             learning_rate=0.013234784333291012, num_leaves=45, max_depth=9,
             min_child_samples=55, subsample=0.8914384900186465,
             colsample_bytree=0.4003537678772968, reg_alpha=3.7116839920604323,
             reg_lambda=0.18329098910085154, random_state=42, verbose=-1, n_jobs=-1)
run_lgb(tuned, y_log, np.expm1, 'lgb_tuned')

# ─── G5: LGB_DART ────────────────────────────────────────────────
print("\n[G5] LGB_DART (n=1000)...", flush=True)
dart_p = dict(objective='huber', alpha=0.9, boosting_type='dart',
              n_estimators=1000, learning_rate=0.05, num_leaves=63, max_depth=6,
              min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
              reg_alpha=0.5, reg_lambda=0.5, random_state=42, verbose=-1, n_jobs=-1)
run_lgb_dart(dart_p, 'lgb_dart')

# ─── G6: LGB_sqrt ────────────────────────────────────────────────
print("\n[G6] LGB_sqrt (sqrt target)...", flush=True)
sqrt_p = dict(objective='regression_l1', n_estimators=3000, learning_rate=0.03,
              num_leaves=63, max_depth=6, min_child_samples=50,
              subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
              random_state=42, verbose=-1, n_jobs=-1)
run_lgb(sqrt_p, y_sqrt, lambda x: np.clip(x, 0, None)**2, 'lgb_sqrt')

# ─── G7: LGB_pow ─────────────────────────────────────────────────
print("\n[G7] LGB_pow (y^0.25 target)...", flush=True)
pow_p = dict(objective='regression_l1', n_estimators=3000, learning_rate=0.03,
             num_leaves=63, max_depth=6, min_child_samples=50,
             subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
             random_state=42, verbose=-1, n_jobs=-1)
run_lgb(pow_p, y_pow, lambda x: np.clip(x, 0, None)**4, 'lgb_pow')

# ─── Save ─────────────────────────────────────────────────────────
print(f"\n=== Summary: {len(oofs)} models ===", flush=True)
mega33 = pickle.load(open('results/mega33_final.pkl','rb'))['meta_avg_oof']
for name, oof in sorted(oofs.items()):
    print(f"  {name}: OOF={mean_absolute_error(y, oof):.5f}  corr(mega33)={np.corrcoef(mega33, oof)[0,1]:.4f}")

with open(f'{OUT}/mega34_trees.pkl', 'wb') as f:
    pickle.dump({'oofs': oofs, 'tests': tests, 'y': y, 'groups': groups}, f)
print(f"\nSaved: {OUT}/mega34_trees.pkl")
print("Done.")
