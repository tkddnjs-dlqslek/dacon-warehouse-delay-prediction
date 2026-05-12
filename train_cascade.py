"""
2-Stage Cascade Model
Stage1: LGB classifier (y > threshold)
Stage2: High-delay specialist (sample-weighted LGB/XGB/CB on v31 features)
Blends with current best: P_high × specialist + (1-P_high) × current_best
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

# ─── Setup ────────────────────────────────────────────────────────
TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
THRESHOLD = 80.0   # y > 80 → high-delay regime

print(f"Loading data... (threshold={THRESHOLD})", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe = pickle.load(f)
feat_cols = fe['feat_cols']
train_fe = fe['train_fe']  # ls-sorted
test_fe  = fe['test_fe']
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
del fe

y_high = (y > THRESHOLD).astype(int)
print(f"  High delay (y>{THRESHOLD}): {y_high.sum()} / {len(y)} = {y_high.mean():.3%}", flush=True)
print(f"  High delay y stats: mean={y[y_high==1].mean():.1f}  max={y[y_high==1].max():.1f}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

# ─── Stage 1: Classifier ──────────────────────────────────────────
print("\n[Stage1] Training high-delay classifier...", flush=True)
clf_oof  = np.zeros(len(y))
clf_test = np.zeros(len(X_te))

for i, (tr_idx, val_idx) in enumerate(folds):
    clf = lgb.LGBMClassifier(
        objective='binary', n_estimators=2000, learning_rate=0.03,
        num_leaves=63, max_depth=6, min_child_samples=50,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
        random_state=SEED, verbose=-1, n_jobs=-1
    )
    clf.fit(X_tr[tr_idx], y_high[tr_idx],
            eval_set=[(X_tr[val_idx], y_high[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    clf_oof[val_idx] = clf.predict_proba(X_tr[val_idx])[:, 1]
    clf_test += clf.predict_proba(X_te)[:, 1] / N_SPLITS

auc = roc_auc_score(y_high, clf_oof)
print(f"  Classifier OOF AUC: {auc:.4f}", flush=True)
print(f"  Mean P(high) train: {clf_oof.mean():.4f}  test: {clf_test.mean():.4f}", flush=True)

# ─── Stage 2: High-delay Specialist ──────────────────────────────
print("\n[Stage2] Training high-delay specialist models...", flush=True)
# Sample weights: y>80 gets 30x weight, others get 1x
sample_weights = np.where(y_high == 1, 30.0, 1.0)

spec_oofs_all  = {}
spec_tests_all = {}

# Specialist objective: MAE on raw y (no log transform) to preserve high-value range
# BUT: use log(y) for stability, just with heavy weights

# Method A: log1p target but 30x weight on high rows
def train_lgb_weighted(params, name):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y_log[tr_idx],
              sample_weight=sample_weights[tr_idx],
              eval_set=[(X_tr[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
        te += np.expm1(m.predict(X_te)) / N_SPLITS
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    spec_oofs_all[name] = oof; spec_tests_all[name] = te
    mae_all  = mean_absolute_error(y, oof)
    mae_high = mean_absolute_error(y[y_high==1], oof[y_high==1])
    print(f"  [{name}] OOF MAE all={mae_all:.4f}  high={mae_high:.4f}", flush=True)

print("  LGB (weighted, Huber)...", flush=True)
train_lgb_weighted(dict(
    objective='huber', alpha=0.9, n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'lgb_w30_huber')

print("  LGB (weighted, MAE)...", flush=True)
train_lgb_weighted(dict(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'lgb_w30_mae')

# CatBoost weighted (faster than XGB absoluteerror on large datasets)
print("  CatBoost (weighted)...", flush=True)
oof = np.zeros(len(y)); te = np.zeros(len(X_te))
for i, (tr_idx, val_idx) in enumerate(folds):
    m = CatBoostRegressor(
        loss_function='MAE', iterations=3000, learning_rate=0.03,
        depth=8, min_data_in_leaf=20, subsample=0.8, rsm=0.7,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type='CPU'
    )
    m.fit(X_tr[tr_idx], y_log[tr_idx],
          sample_weight=sample_weights[tr_idx],
          eval_set=(X_tr[val_idx], y_log[val_idx]))
    oof[val_idx] = np.expm1(m.predict(X_tr[val_idx]))
    te += np.expm1(m.predict(X_te)) / N_SPLITS
oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
spec_oofs_all['cb_w30'] = oof; spec_tests_all['cb_w30'] = te
print(f"  [cb_w30] OOF MAE all={mean_absolute_error(y,oof):.4f}  high={mean_absolute_error(y[y_high==1],oof[y_high==1]):.4f}", flush=True)

# Average specialist predictions
spec_avg_oof  = np.mean(list(spec_oofs_all.values()), axis=0)
spec_avg_test = np.mean(list(spec_tests_all.values()), axis=0)
print(f"\n  Specialist avg OOF MAE all={mean_absolute_error(y,spec_avg_oof):.4f}  high={mean_absolute_error(y[y_high==1],spec_avg_oof[y_high==1]):.4f}", flush=True)

# ─── Save ─────────────────────────────────────────────────────────
os.makedirs('results/cascade', exist_ok=True)
np.save('results/cascade/clf_oof.npy',  clf_oof)
np.save('results/cascade/clf_test.npy', clf_test)
for name, arr in spec_oofs_all.items():
    np.save(f'results/cascade/spec_{name}_oof.npy', arr)
for name, arr in spec_tests_all.items():
    np.save(f'results/cascade/spec_{name}_test.npy', arr)
np.save('results/cascade/spec_avg_oof.npy',  spec_avg_oof)
np.save('results/cascade/spec_avg_test.npy', spec_avg_test)

print(f"\nSaved to results/cascade/")
print("Done.")
