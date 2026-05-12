"""
Pack-station bottleneck specialist.
pack_station_count<=2 layouts → OOF MAE 24.65 (3x average), 14 layouts.
Train a dedicated model for these layouts, blend into main prediction.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

print("Loading data...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe = pickle.load(f)
feat_cols = fe['feat_cols']
train_fe = fe['train_fe']
test_fe  = fe['test_fe']
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
layout_info = pd.read_csv('layout_info.csv')[['layout_id','pack_station_count']]
pack_tr = train_fe[['layout_id']].merge(layout_info, on='layout_id', how='left')['pack_station_count'].values
pack_te = test_fe[['layout_id']].merge(layout_info, on='layout_id', how='left')['pack_station_count'].values
del fe

# Bottleneck mask
is_bottleneck_tr = (pack_tr <= 2).astype(int)
is_bottleneck_te = (pack_te <= 2).astype(int)
print(f"  Train bottleneck rows: {is_bottleneck_tr.sum()} ({is_bottleneck_tr.mean():.3%})", flush=True)
print(f"  Test  bottleneck rows: {is_bottleneck_te.sum()} ({is_bottleneck_te.mean():.3%})", flush=True)
print(f"  Bottleneck y stats: mean={y[is_bottleneck_tr==1].mean():.2f}  max={y[is_bottleneck_tr==1].max():.2f}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

spec_oofs_all  = {}
spec_tests_all = {}

def train_lgb(params, name, use_raw=False):
    oof = np.zeros(len(y)); te = np.zeros(len(X_te))
    target = y if use_raw else y_log
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(**params)
        # 10x weight on bottleneck rows
        w = np.where(is_bottleneck_tr[tr_idx] == 1, 10.0, 1.0)
        m.fit(X_tr[tr_idx], target[tr_idx],
              sample_weight=w,
              eval_set=[(X_tr[val_idx], target[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        pred = m.predict(X_tr[val_idx])
        oof[val_idx] = pred if use_raw else np.expm1(pred)
        pred_te = m.predict(X_te)
        te += (pred_te if use_raw else np.expm1(pred_te)) / N_SPLITS
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    spec_oofs_all[name] = oof; spec_tests_all[name] = te
    mae_all  = mean_absolute_error(y, oof)
    mae_bot  = mean_absolute_error(y[is_bottleneck_tr==1], oof[is_bottleneck_tr==1])
    mae_high = mean_absolute_error(y[y>80], oof[y>80])
    print(f"  [{name}] MAE all={mae_all:.4f}  bottleneck={mae_bot:.4f}  y>80={mae_high:.4f}", flush=True)

print("\n[Pack Specialist] LGB Huber (log-y, 10x bottleneck)...", flush=True)
train_lgb(dict(
    objective='huber', alpha=0.9, n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'pack_lgb_huber', use_raw=False)

print("\n[Pack Specialist] LGB MAE (raw-y, 10x bottleneck)...", flush=True)
train_lgb(dict(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=127, max_depth=8, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
    random_state=SEED, verbose=-1, n_jobs=-1
), 'pack_lgb_mae', use_raw=True)

print("\n[Pack Specialist] CatBoost (raw-y, 10x bottleneck)...", flush=True)
oof = np.zeros(len(y)); te = np.zeros(len(X_te))
for i, (tr_idx, val_idx) in enumerate(folds):
    w = np.where(is_bottleneck_tr[tr_idx] == 1, 10.0, 1.0)
    m = CatBoostRegressor(
        loss_function='MAE', iterations=3000, learning_rate=0.03,
        depth=8, min_data_in_leaf=20, subsample=0.8, rsm=0.7,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
        early_stopping_rounds=100, task_type='CPU'
    )
    m.fit(X_tr[tr_idx], y[tr_idx], sample_weight=w,
          eval_set=(X_tr[val_idx], y[val_idx]))
    oof[val_idx] = np.clip(m.predict(X_tr[val_idx]), 0, None)
    te += np.clip(m.predict(X_te), 0, None) / N_SPLITS
spec_oofs_all['pack_cb'] = oof; spec_tests_all['pack_cb'] = te
mae_all = mean_absolute_error(y, oof)
mae_bot = mean_absolute_error(y[is_bottleneck_tr==1], oof[is_bottleneck_tr==1])
mae_high = mean_absolute_error(y[y>80], oof[y>80])
print(f"  [pack_cb] MAE all={mae_all:.4f}  bottleneck={mae_bot:.4f}  y>80={mae_high:.4f}", flush=True)

pack_avg_oof  = np.mean(list(spec_oofs_all.values()), axis=0)
pack_avg_test = np.mean(list(spec_tests_all.values()), axis=0)
print(f"\n  Pack avg MAE all={mean_absolute_error(y,pack_avg_oof):.4f}  bottleneck={mean_absolute_error(y[is_bottleneck_tr==1],pack_avg_oof[is_bottleneck_tr==1]):.4f}", flush=True)

os.makedirs('results/pack_spec', exist_ok=True)
for name, arr in spec_oofs_all.items():
    np.save(f'results/pack_spec/{name}_oof.npy', arr)
for name, arr in spec_tests_all.items():
    np.save(f'results/pack_spec/{name}_test.npy', arr)
np.save('results/pack_spec/pack_avg_oof.npy',  pack_avg_oof)
np.save('results/pack_spec/pack_avg_test.npy', pack_avg_test)
np.save('results/pack_spec/is_bottleneck_tr.npy', is_bottleneck_tr)
np.save('results/pack_spec/is_bottleneck_te.npy', is_bottleneck_te)
print("Saved to results/pack_spec/  Done.")
