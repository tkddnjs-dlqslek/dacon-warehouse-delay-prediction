"""
Cascade v2: specialist trained on RAW y (no log1p) + 100x weight
Key fix: log1p compresses high-value range → specialist underestimates y>80
Using raw MAE target forces specialist to directly predict the true scale.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
THRESHOLD = 80.0

print(f"Loading data... (threshold={THRESHOLD}, raw-y specialist, 100x weight)", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe = pickle.load(f)
feat_cols = fe['feat_cols']
train_fe = fe['train_fe']
test_fe  = fe['test_fe']
y = train_fe[TARGET].values.astype(np.float64)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
del fe

y_high = (y > THRESHOLD).astype(int)
print(f"  High delay: {y_high.sum()} / {len(y)} = {y_high.mean():.3%}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

# Load classifier from v1 (already trained)
clf_oof  = np.load('results/cascade/clf_oof.npy')
clf_test = np.load('results/cascade/clf_test.npy')
print(f"  Loaded classifier OOF AUC: {roc_auc_score(y_high, clf_oof):.4f}", flush=True)

# 100x weight on high-delay rows, raw y target
sample_weights = np.where(y_high == 1, 100.0, 1.0)

spec_oofs_all  = {}
spec_tests_all = {}

def train_lgb_raw(params, name):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_tr[tr_idx], y[tr_idx],
              sample_weight=sample_weights[tr_idx],
              eval_set=[(X_tr[val_idx], y[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = m.predict(X_tr[val_idx])
        te += m.predict(X_te) / N_SPLITS
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    spec_oofs_all[name] = oof; spec_tests_all[name] = te
    mae_all  = mean_absolute_error(y, oof)
    mae_high = mean_absolute_error(y[y_high==1], oof[y_high==1])
    print(f"  [{name}] OOF MAE all={mae_all:.4f}  high={mae_high:.4f}", flush=True)

print("\n[Stage2-v2] Raw-y specialist (100x weight)...", flush=True)

print("  LGB (raw-y, Huber)...", flush=True)
train_lgb_raw(dict(
    objective='huber', alpha=0.9, n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'lgb_raw_huber')

print("  LGB (raw-y, MAE)...", flush=True)
train_lgb_raw(dict(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'lgb_raw_mae')

print("  CatBoost (raw-y, MAE)...", flush=True)
oof = np.zeros(len(y)); te = np.zeros(len(X_te))
for i, (tr_idx, val_idx) in enumerate(folds):
    m = CatBoostRegressor(
        loss_function='MAE', iterations=3000, learning_rate=0.03,
        depth=8, min_data_in_leaf=20, subsample=0.8, rsm=0.7,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type='CPU'
    )
    m.fit(X_tr[tr_idx], y[tr_idx],
          sample_weight=sample_weights[tr_idx],
          eval_set=(X_tr[val_idx], y[val_idx]))
    oof[val_idx] = np.clip(m.predict(X_tr[val_idx]), 0, None)
    te += np.clip(m.predict(X_te), 0, None) / N_SPLITS
spec_oofs_all['cb_raw'] = oof; spec_tests_all['cb_raw'] = te
print(f"  [cb_raw] OOF MAE all={mean_absolute_error(y,oof):.4f}  high={mean_absolute_error(y[y_high==1],oof[y_high==1]):.4f}", flush=True)

spec_avg_oof  = np.mean(list(spec_oofs_all.values()), axis=0)
spec_avg_test = np.mean(list(spec_tests_all.values()), axis=0)
print(f"\n  Spec-v2 avg OOF MAE all={mean_absolute_error(y,spec_avg_oof):.4f}  high={mean_absolute_error(y[y_high==1],spec_avg_oof[y_high==1]):.4f}", flush=True)

os.makedirs('results/cascade', exist_ok=True)
for name, arr in spec_oofs_all.items():
    np.save(f'results/cascade/spec_{name}_oof.npy', arr)
for name, arr in spec_tests_all.items():
    np.save(f'results/cascade/spec_{name}_test.npy', arr)
np.save('results/cascade/spec_v2_avg_oof.npy',  spec_avg_oof)
np.save('results/cascade/spec_v2_avg_test.npy', spec_avg_test)
print("Saved to results/cascade/  Done.")
