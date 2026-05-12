"""
Mega-Stacking: 20개 모델 OOF로 stacking
  - v23 seed42/123/2024 (12개)
  - v24 (4개)
  - v26 Tuned/transforms (4개)
  - MLP1, MLP2, CNN (3개)
  → Total 20개 (선별)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
import time
import pickle
import os

warnings.filterwarnings('ignore')

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
RESULT_DIR = './results'

t0 = time.time()
print("=== Mega-Stacking: 20 모델 ===", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 0: 모든 OOF/test 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 0] 모델 OOF/test 로드...", flush=True)

train = pd.read_csv('./train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test = pd.read_csv('./test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
y = train[TARGET].values
y_log = np.log1p(y)
groups = train['layout_id']

selected_oofs = {}
selected_tests = {}

# v23 multi-seed (LGB_MAE 제외 — 가중치 0%였음)
for seed in [42, 123, 2024]:
    s = pickle.load(open(f'{RESULT_DIR}/v23_seed{seed}.pkl', 'rb'))
    for name in ['LGB_Huber', 'XGB', 'CatBoost']:
        selected_oofs[f'v23s{seed}_{name}'] = s['oofs'][name]
        selected_tests[f'v23s{seed}_{name}'] = s['tests'][name]

# v24 (다른 피처셋)
v24 = pickle.load(open(f'{RESULT_DIR}/v24_final.pkl', 'rb'))
for name, oof in v24['oofs'].items():
    selected_oofs[f'v24_{name}'] = oof
    selected_tests[f'v24_{name}'] = v24['tests'][name]

# v26 (HP tuned + transforms)
v26 = pickle.load(open(f'{RESULT_DIR}/v26_final.pkl', 'rb'))
for name in ['Tuned_Huber', 'Tuned_sqrt', 'Tuned_pow', 'DART']:
    if name in v26['oofs']:
        selected_oofs[f'v26_{name}'] = v26['oofs'][name]
        selected_tests[f'v26_{name}'] = v26['tests'][name]

# MLP, CNN
mlp1 = pickle.load(open(f'{RESULT_DIR}/mlp_final.pkl', 'rb'))
mlp2 = pickle.load(open(f'{RESULT_DIR}/mlp2_final.pkl', 'rb'))
cnn = pickle.load(open(f'{RESULT_DIR}/cnn_final.pkl', 'rb'))
selected_oofs['mlp1'] = mlp1['mlp_oof']
selected_tests['mlp1'] = mlp1['mlp_test']
selected_oofs['mlp2'] = mlp2['mlp2_oof']
selected_tests['mlp2'] = mlp2['mlp2_test']
selected_oofs['cnn'] = cnn['cnn_oof']
selected_tests['cnn'] = cnn['cnn_test']

print(f"  총 {len(selected_oofs)}개 모델 OOF 로드")
for name, oof in selected_oofs.items():
    print(f"    {name:30s}: {mean_absolute_error(y, oof):.4f}")

# Stacking 피처 (log space)
stack_train = np.column_stack([np.log1p(np.clip(o, 0, None)) for o in selected_oofs.values()])
stack_test = np.column_stack([np.log1p(np.clip(t, 0, None)) for t in selected_tests.values()])
print(f"\n  Stacking 피처 shape: train {stack_train.shape}, test {stack_test.shape}")

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(stack_train, y, groups=groups))

pickle.dump({
    'model_names': list(selected_oofs.keys()),
    'individual_maes': {n: mean_absolute_error(y, o) for n, o in selected_oofs.items()},
}, open(f'{RESULT_DIR}/mega_stacking_phase0.pkl', 'wb'))
print(f"  Phase 0 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: Meta-Learner 학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 1] Meta-Learner 학습...", flush=True)

results = {}

# --- Ridge ---
print("\n  [Ridge]", flush=True)
best_ridge_mae = 999
best_alpha = 1.0
for alpha in [1.0, 10.0, 50.0, 100.0]:
    oof = np.zeros(len(y))
    for tr_idx, val_idx in folds:
        m = Ridge(alpha=alpha)
        m.fit(stack_train[tr_idx], y_log[tr_idx])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    oof = np.clip(oof, 0, None)
    mae = mean_absolute_error(y, oof)
    if mae < best_ridge_mae:
        best_ridge_mae = mae
        best_alpha = alpha
    print(f"    alpha={alpha}: OOF={mae:.4f}", flush=True)

ridge_oof = np.zeros(len(y))
ridge_test = np.zeros(len(stack_test))
for tr_idx, val_idx in folds:
    m = Ridge(alpha=best_alpha)
    m.fit(stack_train[tr_idx], y_log[tr_idx])
    ridge_oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
    ridge_test += np.expm1(m.predict(stack_test)) / N_SPLITS
ridge_oof = np.clip(ridge_oof, 0, None)
ridge_test = np.clip(ridge_test, 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: ridge_test}).to_csv(
    './submission_megastack_ridge.csv', index=False)
results['ridge'] = {'oof': ridge_oof, 'test': ridge_test, 'mae': best_ridge_mae}
print(f"  -> ridge alpha={best_alpha}, OOF={best_ridge_mae:.4f}, csv 저장", flush=True)

# --- LGB Meta ---
lgb_configs = [
    ('lgb_15_4', {'num_leaves': 15, 'max_depth': 4, 'min_child_samples': 100}),
    ('lgb_7_3', {'num_leaves': 7, 'max_depth': 3, 'min_child_samples': 200}),
    ('lgb_31_5', {'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 50}),
]

for config_name, params in lgb_configs:
    print(f"\n  [{config_name}]", flush=True)
    oof = np.zeros(len(y))
    tpred = np.zeros(len(stack_test))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=500, learning_rate=0.05,
            **params, random_state=SEED, verbose=-1, n_jobs=4)
        m.fit(stack_train[tr_idx], y_log[tr_idx],
              eval_set=[(stack_train[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = np.expm1(m.predict(stack_train[val_idx]))
        tpred += np.expm1(m.predict(stack_test)) / N_SPLITS
    oof = np.clip(oof, 0, None)
    tpred = np.clip(tpred, 0, None)
    mae = mean_absolute_error(y, oof)
    pd.DataFrame({'ID': test['ID'], TARGET: tpred}).to_csv(
        f'./submission_megastack_{config_name}.csv', index=False)
    results[config_name] = {'oof': oof, 'test': tpred, 'mae': mae}
    print(f"    OOF={mae:.4f}, csv 저장", flush=True)

pickle.dump({'results': {k: {'mae': v['mae']} for k, v in results.items()},
             'model_names': list(selected_oofs.keys())},
            open(f'{RESULT_DIR}/mega_stacking_phase1.pkl', 'wb'))
print(f"\n  Phase 1 완료 ({time.time()-t0:.0f}s)", flush=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: 블렌딩
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n[Phase 2] 블렌딩...", flush=True)

# 기존 best LB (oofonly_5way_70 = 9.897)
prev_best = pd.read_csv('./submission_oofonly_5way_70.csv')[TARGET].values

# 5-way blend
v22_test = pd.read_csv('submission_v22_pre.csv')[TARGET].values
v24_test_csv = pd.read_csv('submission_v24.csv')[TARGET].values
s42 = pickle.load(open(f'{RESULT_DIR}/v23_seed42.pkl', 'rb'))
v23_test = np.clip(sum(wi*p for wi,p in zip(s42['weights'], [s42['tests'][n] for n in s42['oofs'].keys()])), 0, None)
best5_test = 0.50*v23_test + 0.15*v22_test + 0.10*v24_test_csv + 0.125*mlp1['mlp_test'] + 0.125*mlp2['mlp2_test']

# best mega-stacking 선택
best_name = min(results, key=lambda k: results[k]['mae'])
best = results[best_name]
print(f"  Best mega-stacking: {best_name} (OOF={best['mae']:.4f})")

from scipy.stats import pearsonr
r_prev, _ = pearsonr(best['test'], prev_best)
r_5way, _ = pearsonr(best['test'], best5_test)
print(f"  Mega vs prev_best(9.897) 상관: {r_prev:.4f}")
print(f"  Mega vs 5-way 상관: {r_5way:.4f}")

# Mega + prev_best 블렌딩
print("\n  Mega + oofonly_5way_70 블렌딩:")
for ratio in [0.3, 0.5, 0.6, 0.7, 0.8]:
    blend = ratio * best['test'] + (1 - ratio) * prev_best
    pd.DataFrame({'ID': test['ID'], TARGET: blend}).to_csv(
        f'./submission_megastack_prev_{int(ratio*100)}.csv', index=False)
    print(f"    mega {int(ratio*100)}% + prev {int((1-ratio)*100)}% 저장")

# Mega + 5-way 블렌딩
print("\n  Mega + 5-way 블렌딩:")
for ratio in [0.5, 0.7]:
    blend = ratio * best['test'] + (1 - ratio) * best5_test
    pd.DataFrame({'ID': test['ID'], TARGET: blend}).to_csv(
        f'./submission_megastack_5way_{int(ratio*100)}.csv', index=False)
    print(f"    mega {int(ratio*100)}% + 5-way 저장")

# 모든 mega 결과 + prev 50% 블렌딩 (보수적)
for name, data in results.items():
    blend = 0.6 * data['test'] + 0.4 * prev_best
    pd.DataFrame({'ID': test['ID'], TARGET: blend}).to_csv(
        f'./submission_{name}_prev_60.csv', index=False)

print(f"\n{'='*60}")
print(f"Mega-Stacking 완료!")
for name, data in results.items():
    print(f"  {name}: OOF={data['mae']:.4f}")
print(f"  Best: {best_name} (OOF={best['mae']:.4f})")
print(f"  v23 단독 대비: {8.5787 - best['mae']:+.4f}")
print(f"  prev best (9.897) 대비 OOF 차이: -{8.485 - best['mae']:.4f}")
print(f"  submission_megastack_*.csv 저장")
print(f"  총 소요시간: {time.time()-t0:.0f}s")
print(f"{'='*60}", flush=True)
