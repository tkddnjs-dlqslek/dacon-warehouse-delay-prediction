"""
Step 4: 피처 엔지니어링 적용 + LGB/XGB/CatBoost 3종 학습 (GroupKFold)
"""
import sys
import os
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings
import pickle
warnings.filterwarnings('ignore')

sys.stdout.reconfigure(encoding='utf-8')

from pipeline import load_data, build_features, get_feature_cols

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

# ── 데이터 로드 & 피처 엔지니어링 ──
print("데이터 로드 중...")
train, test, layout = load_data('.')
print("피처 엔지니어링 중...")
train, test = build_features(train, test, layout)
feature_cols = get_feature_cols(train)
print(f"피처 수: {len(feature_cols)}")

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]
groups = train['scenario_id']

# ── GroupKFold 설정 ──
gkf = GroupKFold(n_splits=N_SPLITS)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1) LightGBM (MAE objective)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("LightGBM (MAE) 학습 시작")
print("="*60)

lgb_oof = np.zeros(len(train))
lgb_test = np.zeros(len(test))
lgb_best_iters = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n── LGB Fold {fold+1} ──")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = lgb.LGBMRegressor(
        objective='mae',
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=127,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=SEED,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200), lgb.log_evaluation(200)],
    )

    lgb_best_iters.append(model.best_iteration_)
    lgb_oof[val_idx] = model.predict(X_val)
    lgb_test += model.predict(X_test) / N_SPLITS

lgb_mae = mean_absolute_error(y, lgb_oof)
print(f"\n[LightGBM MAE] OOF MAE: {lgb_mae:.4f}")
print(f"  Best iterations: {lgb_best_iters}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2) XGBoost (MAE objective)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("XGBoost (MAE) 학습 시작")
print("="*60)

xgb_oof = np.zeros(len(train))
xgb_test = np.zeros(len(test))
xgb_best_iters = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n── XGB Fold {fold+1} ──")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(
        objective='reg:absoluteerror',
        n_estimators=5000,
        learning_rate=0.05,
        max_depth=8,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        tree_method='hist',
        random_state=SEED,
        verbosity=0,
        n_jobs=-1,
        early_stopping_rounds=200,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=200,
    )

    xgb_best_iters.append(model.best_iteration)
    xgb_oof[val_idx] = model.predict(X_val)
    xgb_test += model.predict(X_test) / N_SPLITS

xgb_mae = mean_absolute_error(y, xgb_oof)
print(f"\n[XGBoost MAE] OOF MAE: {xgb_mae:.4f}")
print(f"  Best iterations: {xgb_best_iters}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3) CatBoost (MAE objective)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("CatBoost (MAE) 학습 시작")
print("="*60)

cb_oof = np.zeros(len(train))
cb_test = np.zeros(len(test))
cb_best_iters = []

for fold, (tr_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
    print(f"\n── CatBoost Fold {fold+1} ──")
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    model = CatBoostRegressor(
        loss_function='MAE',
        eval_metric='MAE',
        iterations=5000,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=3.0,
        random_strength=1.0,
        bagging_temperature=1.0,
        random_seed=SEED,
        verbose=500,
        early_stopping_rounds=200,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=(X_val, y_val),
        verbose=500,
    )

    cb_best_iters.append(model.best_iteration_)
    cb_oof[val_idx] = model.predict(X_val)
    cb_test += model.predict(X_test) / N_SPLITS

cb_mae = mean_absolute_error(y, cb_oof)
print(f"\n[CatBoost MAE] OOF MAE: {cb_mae:.4f}")
print(f"  Best iterations: {cb_best_iters}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 결과 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("모델별 OOF MAE 요약")
print("="*60)
print(f"  LightGBM : {lgb_mae:.4f}")
print(f"  XGBoost  : {xgb_mae:.4f}")
print(f"  CatBoost : {cb_mae:.4f}")

# ── OOF 및 test predictions 저장 ──
np.save('./results/lgb_oof.npy', lgb_oof)
np.save('./results/xgb_oof.npy', xgb_oof)
np.save('./results/cb_oof.npy', cb_oof)
np.save('./results/lgb_test.npy', lgb_test)
np.save('./results/xgb_test.npy', xgb_test)
np.save('./results/cb_test.npy', cb_test)
np.save('./results/y_true.npy', y.values)

print("\nOOF/test predictions 저장 완료 (results/)")

# ── 단순 평균 앙상블 제출 (1차) ──
ens_test = (lgb_test + xgb_test + cb_test) / 3
ens_oof = (lgb_oof + xgb_oof + cb_oof) / 3
ens_mae = mean_absolute_error(y, ens_oof)
print(f"\n[Simple Avg Ensemble] OOF MAE: {ens_mae:.4f}")

submission = pd.DataFrame({
    'ID': test['ID'],
    TARGET: np.clip(ens_test, 0, None),
})
submission.to_csv('./submission_v2_ensemble_avg.csv', index=False)
print("submission_v2_ensemble_avg.csv 저장 완료.")
