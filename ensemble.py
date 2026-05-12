"""
Step 6: 최적 파라미터로 재학습 + 가중 앙상블 + Feature Importance + 최종 제출
- Optuna 결과(results/optuna_results.json) 사용
- 가중치 최적화 (scipy.optimize)
- Feature Importance 분석 및 저장
"""
import sys
import os
import json
import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

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

# ── Optuna 최적 파라미터 로드 ──
optuna_path = './results/optuna_results.json'
if os.path.exists(optuna_path):
    with open(optuna_path) as f:
        optuna_results = json.load(f)
    print("Optuna 결과 로드 완료")
    for name, res in optuna_results.items():
        print(f"  {name}: MAE={res['mae']:.4f}")
else:
    print("Optuna 결과 없음 — 기본 파라미터 사용")
    optuna_results = None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최적 파라미터로 재학습
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def train_lgb(params_override=None):
    params = {
        'objective': 'mae',
        'n_estimators': 5000,
        'learning_rate': 0.05,
        'num_leaves': 127,
        'max_depth': -1,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': SEED,
        'verbose': -1,
        'n_jobs': -1,
    }
    if params_override:
        params.update(params_override)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances = np.zeros(len(feature_cols))
    models = []

    for tr_idx, val_idx in folds:
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)],
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_pred += model.predict(X_test) / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS
        models.append(model)

    return oof, test_pred, importances, models


def train_xgb(params_override=None):
    params = {
        'objective': 'reg:absoluteerror',
        'n_estimators': 5000,
        'learning_rate': 0.05,
        'max_depth': 8,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'tree_method': 'hist',
        'random_state': SEED,
        'verbosity': 0,
        'n_jobs': -1,
        'early_stopping_rounds': 200,
    }
    if params_override:
        params.update(params_override)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances = np.zeros(len(feature_cols))

    for tr_idx, val_idx in folds:
        model = xgb.XGBRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_pred += model.predict(X_test) / N_SPLITS
        importances += model.feature_importances_ / N_SPLITS

    return oof, test_pred, importances


def train_cb(params_override=None):
    params = {
        'loss_function': 'MAE',
        'eval_metric': 'MAE',
        'iterations': 5000,
        'learning_rate': 0.05,
        'depth': 8,
        'l2_leaf_reg': 3.0,
        'random_strength': 1.0,
        'bagging_temperature': 1.0,
        'random_seed': SEED,
        'verbose': 0,
        'early_stopping_rounds': 200,
    }
    if params_override:
        params.update(params_override)

    oof = np.zeros(len(train))
    test_pred = np.zeros(len(test))
    importances = np.zeros(len(feature_cols))

    for tr_idx, val_idx in folds:
        model = CatBoostRegressor(**params)
        model.fit(
            X.iloc[tr_idx], y.iloc[tr_idx],
            eval_set=(X.iloc[val_idx], y.iloc[val_idx]),
            verbose=0,
        )
        oof[val_idx] = model.predict(X.iloc[val_idx])
        test_pred += model.predict(X_test) / N_SPLITS
        fi = model.get_feature_importance()
        importances += fi / N_SPLITS

    return oof, test_pred, importances


# ── 학습 실행 ──
print("\n" + "="*60)
print("최적 파라미터로 LightGBM 학습")
print("="*60)
lgb_params = optuna_results['lgb']['params'] if optuna_results else None
lgb_oof, lgb_test, lgb_fi, lgb_models = train_lgb(lgb_params)
lgb_mae = mean_absolute_error(y, lgb_oof)
print(f"LightGBM OOF MAE: {lgb_mae:.4f}")

print("\n" + "="*60)
print("최적 파라미터로 XGBoost 학습")
print("="*60)
xgb_params = optuna_results['xgb']['params'] if optuna_results else None
xgb_oof, xgb_test, xgb_fi = train_xgb(xgb_params)
xgb_mae = mean_absolute_error(y, xgb_oof)
print(f"XGBoost OOF MAE: {xgb_mae:.4f}")

print("\n" + "="*60)
print("최적 파라미터로 CatBoost 학습")
print("="*60)
cb_params = optuna_results['cb']['params'] if optuna_results else None
cb_oof, cb_test, cb_fi = train_cb(cb_params)
cb_mae = mean_absolute_error(y, cb_oof)
print(f"CatBoost OOF MAE: {cb_mae:.4f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 가중 앙상블 최적화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("가중 앙상블 최적화")
print("="*60)

oof_preds = [lgb_oof, xgb_oof, cb_oof]
test_preds = [lgb_test, xgb_test, cb_test]


def ensemble_mae(weights):
    w = np.array(weights)
    w = w / w.sum()  # normalize
    pred = sum(wi * p for wi, p in zip(w, oof_preds))
    return mean_absolute_error(y, pred)


# 최적 가중치 탐색
result = minimize(
    ensemble_mae,
    x0=[1/3, 1/3, 1/3],
    method='Nelder-Mead',
    options={'maxiter': 10000},
)

best_weights = np.array(result.x)
best_weights = best_weights / best_weights.sum()
best_mae = result.fun

print(f"최적 가중치: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CB={best_weights[2]:.3f}")
print(f"가중 앙상블 OOF MAE: {best_mae:.4f}")

# 단순 평균도 비교
simple_avg_mae = mean_absolute_error(y, (lgb_oof + xgb_oof + cb_oof) / 3)
print(f"단순 평균 OOF MAE: {simple_avg_mae:.4f}")

# 최종 test 예측
final_test_pred = sum(w * p for w, p in zip(best_weights, test_preds))
final_test_pred = np.clip(final_test_pred, 0, None)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Feature Importance 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("Feature Importance 분석")
print("="*60)

fi_df = pd.DataFrame({
    'feature': feature_cols,
    'lgb_importance': lgb_fi,
    'xgb_importance': xgb_fi,
    'cb_importance': cb_fi,
})

# 정규화
for col in ['lgb_importance', 'xgb_importance', 'cb_importance']:
    fi_df[col] = fi_df[col] / fi_df[col].sum()

fi_df['avg_importance'] = fi_df[['lgb_importance', 'xgb_importance', 'cb_importance']].mean(axis=1)
fi_df = fi_df.sort_values('avg_importance', ascending=False).reset_index(drop=True)

# 상위 30개 출력
print("\n상위 30개 피처:")
for i, row in fi_df.head(30).iterrows():
    print(f"  {i+1:2d}. {row['feature']:40s} avg={row['avg_importance']:.4f}")

fi_df.to_csv('./results/feature_importance.csv', index=False)

# 시각화
fig, ax = plt.subplots(figsize=(12, 10))
top30 = fi_df.head(30).iloc[::-1]
y_pos = range(len(top30))
ax.barh(y_pos, top30['avg_importance'], color='steelblue')
ax.set_yticks(y_pos)
ax.set_yticklabels(top30['feature'], fontsize=8)
ax.set_xlabel('Average Importance (Normalized)')
ax.set_title('Top 30 Feature Importance (LGB + XGB + CatBoost Average)')
plt.tight_layout()
plt.savefig('./results/fi_plot.png', dpi=150)
print("Feature Importance 저장 완료: results/feature_importance.csv, results/fi_plot.png")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 제출 파일 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("\n" + "="*60)
print("제출 파일 생성")
print("="*60)

submission = pd.DataFrame({
    'ID': test['ID'],
    TARGET: final_test_pred,
})
submission.to_csv('./submission_v3_optimized.csv', index=False)
print(f"submission_v3_optimized.csv 저장 완료 (shape: {submission.shape})")
print(f"예측값 통계: mean={final_test_pred.mean():.2f}, median={np.median(final_test_pred):.2f}, "
      f"min={final_test_pred.min():.2f}, max={final_test_pred.max():.2f}")

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 실험 로그 업데이트
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
log_data = {
    'experiment': ['v1_baseline', 'v2_ensemble_avg', 'v3_optimized'],
    'oof_mae': [9.2405, 8.7868, best_mae],
    'public_lb': ['', '', ''],
    'description': [
        '원본 90피처 LightGBM KFold',
        'FE(212피처) + LGB/XGB/CB 단순평균',
        f'Optuna 튜닝 + 가중앙상블 (w={best_weights.round(3).tolist()})',
    ],
}
log_df = pd.DataFrame(log_data)
log_df.to_csv('./results/experiment_log.csv', index=False)
print("\n실험 로그 저장: results/experiment_log.csv")

print("\n" + "="*60)
print("전체 파이프라인 완료!")
print(f"  Baseline MAE:    9.2405")
print(f"  단순 앙상블 MAE: {simple_avg_mae:.4f}")
print(f"  최종 MAE:        {best_mae:.4f}")
print("="*60)
