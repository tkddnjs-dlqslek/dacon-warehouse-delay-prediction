"""
Retrain mega33 stacking layer with v31 base models added.
Keeps all existing model pkl files (neural networks, v23/v24/v26/domain etc).
Adds LGB_v31, XGB_v31, CB_v31 OOF predictions to the stack.
Saves: results/mega33_v31_final.pkl
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

RESULT_DIR = './results'
TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

print("Loading v31 FE for y and groups...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
train_fe = fe_v31['train_fe']
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
groups = train_fe['layout_id'].values
del fe_v31

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

print("Loading existing mega33 component predictions...", flush=True)
selected_oofs = {}
selected_tests = {}

# v23 seed models
for seed in [42, 123, 2024]:
    try:
        s = pickle.load(open(f'{RESULT_DIR}/v23_seed{seed}.pkl', 'rb'))
        for name in ['LGB_Huber', 'XGB', 'CatBoost']:
            k = f'v23s{seed}_{name}'
            selected_oofs[k] = s['oofs'][name]
            selected_tests[k] = s['tests'][name]
    except: pass

# v24
try:
    v24 = pickle.load(open(f'{RESULT_DIR}/v24_final.pkl', 'rb'))
    for name, oof in v24['oofs'].items():
        selected_oofs[f'v24_{name}'] = oof
        selected_tests[f'v24_{name}'] = v24['tests'][name]
except: pass

# v26
try:
    v26 = pickle.load(open(f'{RESULT_DIR}/v26_final.pkl', 'rb'))
    for name in ['Tuned_Huber', 'Tuned_sqrt', 'Tuned_pow', 'DART']:
        if name in v26['oofs']:
            selected_oofs[f'v26_{name}'] = v26['oofs'][name]
            selected_tests[f'v26_{name}'] = v26['tests'][name]
except: pass

# Neural & domain models
for pname, key_map in [
    ('mlp_final', {'mlp_oof': 'mlp1', 'mlp_test': 'mlp1'}),
    ('mlp2_final', {'mlp2_oof': 'mlp2', 'mlp2_test': 'mlp2'}),
    ('cnn_final', {'cnn_oof': 'cnn', 'cnn_test': 'cnn'}),
    ('mlp_aug_final', {'mlp_aug_oof': 'mlp_aug', 'mlp_aug_test': 'mlp_aug'}),
]:
    try:
        p = pickle.load(open(f'{RESULT_DIR}/{pname}.pkl', 'rb'))
        for k, v in key_map.items():
            if '_oof' in k:
                selected_oofs[v] = p[k]
            else:
                selected_tests[v] = p[k]
    except: pass

try:
    domain = pickle.load(open(f'{RESULT_DIR}/domain_phase2.pkl', 'rb'))
    for n in domain['oofs']:
        selected_oofs[f'domain_{n}'] = domain['oofs'][n]
        selected_tests[f'domain_{n}'] = domain['tests'][n]
except: pass

try:
    offset_p3 = pickle.load(open(f'{RESULT_DIR}/offset_phase3.pkl', 'rb'))
    for n, data in offset_p3.items():
        selected_oofs[f'offset_{n}'] = data['oof']
        selected_tests[f'offset_{n}'] = data['test']
except: pass

try:
    na = pickle.load(open(f'{RESULT_DIR}/neural_army.pkl', 'rb'))
    for name, data in na.items():
        selected_oofs[f'na_{name}'] = data['oof']
        selected_tests[f'na_{name}'] = data['test']
except: pass

# ─── NEW: v31 base models ─────────────────────────────────────
print("Loading v31 base models...", flush=True)
for mname in ['lgb', 'xgb', 'cb']:
    oof_path  = f'{RESULT_DIR}/base_v31/{mname}_v31_oof.npy'
    test_path = f'{RESULT_DIR}/base_v31/{mname}_v31_test.npy'
    if os.path.exists(oof_path) and os.path.exists(test_path):
        selected_oofs[f'v31_{mname}']  = np.load(oof_path)
        selected_tests[f'v31_{mname}'] = np.load(test_path)
        print(f"  v31_{mname}: OOF MAE={mean_absolute_error(y, selected_oofs[f'v31_{mname}']):.4f}")
    else:
        print(f"  WARNING: {oof_path} not found, skipping v31_{mname}")

print(f"\nTotal stack models: {len(selected_oofs)}", flush=True)

# Build stack matrices (log1p transform)
n_train = len(y)
n_test = len(selected_tests[list(selected_tests.keys())[0]])
stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test  = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])
print(f"Stack shape: train={stack_train.shape}  test={stack_test.shape}", flush=True)

# ─── 3-meta stacking ─────────────────────────────────────────
meta_oofs_all  = {}
meta_tests_all = {}

# LGB meta
print("\n[Meta-LGB]...", flush=True)
oof = np.zeros(n_train); tpred = np.zeros(n_test)
for tr_idx, val_idx in folds:
    m = lgb.LGBMRegressor(objective='mae', n_estimators=500, learning_rate=0.05,
                           num_leaves=15, max_depth=4, min_child_samples=100,
                           random_state=SEED, verbose=-1, n_jobs=-1)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])],
          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['lgb'] = oof; meta_tests_all['lgb'] = tpred
print(f"  LGB meta OOF: {mean_absolute_error(y, oof):.5f}", flush=True)

# XGB meta
print("[Meta-XGB]...", flush=True)
oof = np.zeros(n_train); tpred = np.zeros(n_test)
for tr_idx, val_idx in folds:
    m = xgb.XGBRegressor(objective='reg:absoluteerror', n_estimators=500, learning_rate=0.05,
                          max_depth=4, min_child_weight=100, subsample=0.8, colsample_bytree=0.8,
                          tree_method='hist', random_state=SEED, verbosity=0, n_jobs=4,
                          early_stopping_rounds=50)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=[(stack_train[val_idx], y_log[val_idx])], verbose=False)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['xgb'] = oof; meta_tests_all['xgb'] = tpred
print(f"  XGB meta OOF: {mean_absolute_error(y, oof):.5f}", flush=True)

# CatBoost meta
print("[Meta-CB]...", flush=True)
oof = np.zeros(n_train); tpred = np.zeros(n_test)
for tr_idx, val_idx in folds:
    m = CatBoostRegressor(loss_function='MAE', iterations=500, learning_rate=0.05,
                          depth=4, min_data_in_leaf=100, random_seed=SEED, verbose=0,
                          early_stopping_rounds=50, task_type='CPU', thread_count=4)
    m.fit(stack_train[tr_idx], y_log[tr_idx],
          eval_set=(stack_train[val_idx], y_log[val_idx]), use_best_model=True)
    oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
oof = np.clip(oof, 0, None); tpred = np.clip(tpred, 0, None)
meta_oofs_all['cb'] = oof; meta_tests_all['cb'] = tpred
print(f"  CB meta OOF: {mean_absolute_error(y, oof):.5f}", flush=True)

# Average meta predictions
meta_avg_oof  = np.mean([meta_oofs_all[k]  for k in meta_oofs_all], axis=0)
meta_avg_test = np.mean([meta_tests_all[k] for k in meta_tests_all], axis=0)
print(f"\nmega33_v31 avg OOF: {mean_absolute_error(y, meta_avg_oof):.5f}", flush=True)

# Compare with original mega33
with open('results/mega33_final.pkl','rb') as f: d_old = pickle.load(f)
print(f"mega33_orig avg OOF: {mean_absolute_error(y, d_old['meta_avg_oof']):.5f}")
print(f"corr(old,new) = {np.corrcoef(d_old['meta_avg_oof'], meta_avg_oof)[0,1]:.4f}")

# Save
out = {
    'meta_oofs':     meta_oofs_all,
    'meta_tests':    meta_tests_all,
    'meta_avg_oof':  meta_avg_oof,
    'meta_avg_test': meta_avg_test,
}
with open('results/mega33_v31_final.pkl', 'wb') as f:
    pickle.dump(out, f)
print("Saved: results/mega33_v31_final.pkl")
print("Done.")
