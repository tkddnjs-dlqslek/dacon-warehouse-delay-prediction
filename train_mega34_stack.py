"""
mega34 stacking: mega34 trees + mega33 neural networks → mega34_final.pkl
Loads:
  - results/mega34/mega34_trees.pkl  (13 tree models on v31)
  - results/base_v31/lgb_v31_{oof,test}.npy + xgb/cb  (already done)
  - existing neural networks from mega33 component pkls
3-meta stacking (LGB+XGB+CB) → mega34_final.pkl
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
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

selected_oofs  = {}
selected_tests = {}

# ─── mega34 tree models ───────────────────────────────────────────
print("Loading mega34 tree models...", flush=True)
trees = pickle.load(open('results/mega34/mega34_trees.pkl', 'rb'))
for name, oof in trees['oofs'].items():
    selected_oofs[f'm34_{name}']  = oof
    selected_tests[f'm34_{name}'] = trees['tests'][name]
print(f"  mega34 trees: {len(trees['oofs'])} models", flush=True)

# ─── v31 base models (already trained) ───────────────────────────
print("Loading v31 base models...", flush=True)
for mn in ['lgb', 'xgb', 'cb']:
    op = f'{RESULT_DIR}/base_v31/{mn}_v31_oof.npy'
    tp = f'{RESULT_DIR}/base_v31/{mn}_v31_test.npy'
    if os.path.exists(op) and os.path.exists(tp):
        selected_oofs[f'v31_{mn}']  = np.load(op)
        selected_tests[f'v31_{mn}'] = np.load(tp)

# ─── Existing neural networks from mega33 ────────────────────────
print("Loading mega33 neural networks...", flush=True)
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
                selected_oofs[v]  = p[k]
            else:
                selected_tests[v] = p[k]
    except: pass

try:
    na = pickle.load(open(f'{RESULT_DIR}/neural_army.pkl', 'rb'))
    for name, data in na.items():
        selected_oofs[f'na_{name}']  = data['oof']
        selected_tests[f'na_{name}'] = data['test']
except: pass

# domain + offset (tree models but with special FE — keep for diversity)
try:
    domain = pickle.load(open(f'{RESULT_DIR}/domain_phase2.pkl', 'rb'))
    for n in domain['oofs']:
        selected_oofs[f'domain_{n}']  = domain['oofs'][n]
        selected_tests[f'domain_{n}'] = domain['tests'][n]
except: pass

try:
    offset_p3 = pickle.load(open(f'{RESULT_DIR}/offset_phase3.pkl', 'rb'))
    for n, data in offset_p3.items():
        selected_oofs[f'offset_{n}']  = data['oof']
        selected_tests[f'offset_{n}'] = data['test']
except: pass

print(f"\nTotal stack models: {len(selected_oofs)}", flush=True)

# ─── Build stack matrices ─────────────────────────────────────────
n_train = len(y)
n_test  = len(selected_tests[list(selected_tests.keys())[0]])
stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test  = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])
print(f"Stack shape: train={stack_train.shape}  test={stack_test.shape}", flush=True)

meta_oofs_all  = {}
meta_tests_all = {}

# Meta-LGB
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

# Meta-XGB
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

# Meta-CB
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

meta_avg_oof  = np.mean([meta_oofs_all[k]  for k in meta_oofs_all], axis=0)
meta_avg_test = np.mean([meta_tests_all[k] for k in meta_tests_all], axis=0)
print(f"\nmega34 avg OOF: {mean_absolute_error(y, meta_avg_oof):.5f}", flush=True)

mega33 = pickle.load(open('results/mega33_final.pkl','rb'))
print(f"mega33 avg OOF: {mean_absolute_error(y, mega33['meta_avg_oof']):.5f}")
print(f"corr(mega33, mega34) = {np.corrcoef(mega33['meta_avg_oof'], meta_avg_oof)[0,1]:.4f}")

out = {
    'meta_oofs':     meta_oofs_all,
    'meta_tests':    meta_tests_all,
    'meta_avg_oof':  meta_avg_oof,
    'meta_avg_test': meta_avg_test,
}
with open('results/mega34_final.pkl', 'wb') as f:
    pickle.dump(out, f)
print("Saved: results/mega34_final.pkl")
print("Done.")
