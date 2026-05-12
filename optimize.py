"""
Step 5: Optuna 하이퍼파라미터 튜닝 (LightGBM, XGBoost, CatBoost)
train_models.py 결과를 보고 가장 유망한 모델부터 최적화
"""
import sys
import pandas as pd
import numpy as np
import optuna
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
import warnings
import json
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.stdout.reconfigure(encoding='utf-8')

from pipeline import load_data, build_features, get_feature_cols

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42

# ── 데이터 준비 ──
print("데이터 로드 + 피처 엔지니어링...")
train, test, layout = load_data('.')
train, test = build_features(train, test, layout)
feature_cols = get_feature_cols(train)

X = train[feature_cols]
y = train[TARGET]
X_test = test[feature_cols]
groups = train['scenario_id']
gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(X, y, groups))

print(f"피처 수: {len(feature_cols)}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# LightGBM Optuna
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def lgb_objective(trial):
    params = {
        'objective': 'mae',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }

    oof = np.zeros(len(train))
    for tr_idx, val_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
    return mean_absolute_error(y, oof)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# XGBoost Optuna
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def xgb_objective(trial):
    params = {
        'objective': 'reg:absoluteerror',
        'n_estimators': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'random_state': SEED,
        'verbosity': 0,
        'n_jobs': -1,
        'early_stopping_rounds': 100,
    }

    oof = np.zeros(len(train))
    for tr_idx, val_idx in folds:
        model = xgb.XGBRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
    return mean_absolute_error(y, oof)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# CatBoost Optuna
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
def cb_objective(trial):
    params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'iterations': 5000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
        'random_strength': trial.suggest_float('random_strength', 0.1, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.1, 10.0, log=True),
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 100,
    }

    oof = np.zeros(len(train))
    for tr_idx, val_idx in folds:
        model = CatBoostRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
    return mean_absolute_error(y, oof)


if __name__ == '__main__':
    n_trials = 30  # 모델당 30 trials (시간 고려)

    results = {}

    # LightGBM
    print("\n" + "="*60)
    print(f"LightGBM Optuna ({n_trials} trials)")
    print("="*60)
    study_lgb = optuna.create_study(direction='minimize', study_name='lgb')
    study_lgb.optimize(lgb_objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best MAE: {study_lgb.best_value:.4f}")
    print(f"  Best params: {study_lgb.best_params}")
    results['lgb'] = {'mae': study_lgb.best_value, 'params': study_lgb.best_params}

    # XGBoost
    print("\n" + "="*60)
    print(f"XGBoost Optuna ({n_trials} trials)")
    print("="*60)
    study_xgb = optuna.create_study(direction='minimize', study_name='xgb')
    study_xgb.optimize(xgb_objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best MAE: {study_xgb.best_value:.4f}")
    print(f"  Best params: {study_xgb.best_params}")
    results['xgb'] = {'mae': study_xgb.best_value, 'params': study_xgb.best_params}

    # CatBoost
    print("\n" + "="*60)
    print(f"CatBoost Optuna ({n_trials} trials)")
    print("="*60)
    study_cb = optuna.create_study(direction='minimize', study_name='cb')
    study_cb.optimize(cb_objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best MAE: {study_cb.best_value:.4f}")
    print(f"  Best params: {study_cb.best_params}")
    results['cb'] = {'mae': study_cb.best_value, 'params': study_cb.best_params}

    # 결과 저장
    # Convert numpy types to native Python types for JSON serialization
    def convert_types(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        return obj

    with open('./results/optuna_results.json', 'w') as f:
        json.dump(convert_types(results), f, indent=2)

    print("\n결과 저장: results/optuna_results.json")
