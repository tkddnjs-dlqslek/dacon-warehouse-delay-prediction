"""
기존 OOF predictions로 가중 앙상블 최적화 + 제출 생성
(Optuna 완료 대기 없이 즉시 실행)
"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

sys.stdout.reconfigure(encoding='utf-8')

TARGET = 'avg_delay_minutes_next_30m'

# ── 저장된 predictions 로드 ──
lgb_oof = np.load('./results/lgb_oof.npy')
xgb_oof = np.load('./results/xgb_oof.npy')
cb_oof = np.load('./results/cb_oof.npy')
lgb_test = np.load('./results/lgb_test.npy')
xgb_test = np.load('./results/xgb_test.npy')
cb_test = np.load('./results/cb_test.npy')
y = np.load('./results/y_true.npy')

test = pd.read_csv('./test.csv')

print("모델별 OOF MAE:")
print(f"  LightGBM : {mean_absolute_error(y, lgb_oof):.4f}")
print(f"  XGBoost  : {mean_absolute_error(y, xgb_oof):.4f}")
print(f"  CatBoost : {mean_absolute_error(y, cb_oof):.4f}")

# ── 가중 앙상블 최적화 ──
oof_preds = [lgb_oof, xgb_oof, cb_oof]
test_preds = [lgb_test, xgb_test, cb_test]


def ensemble_mae(weights):
    w = np.array(weights)
    w = w / w.sum()
    pred = sum(wi * p for wi, p in zip(w, oof_preds))
    return mean_absolute_error(y, pred)


result = minimize(
    ensemble_mae,
    x0=[1/3, 1/3, 1/3],
    method='Nelder-Mead',
    options={'maxiter': 10000},
)

best_weights = np.array(result.x)
best_weights = best_weights / best_weights.sum()
best_mae = result.fun

print(f"\n최적 가중치: LGB={best_weights[0]:.3f}, XGB={best_weights[1]:.3f}, CB={best_weights[2]:.3f}")
print(f"가중 앙상블 OOF MAE: {best_mae:.4f}")
print(f"단순 평균 OOF MAE:  {mean_absolute_error(y, (lgb_oof + xgb_oof + cb_oof) / 3):.4f}")

# ── 제출 파일 ──
final_pred = sum(w * p for w, p in zip(best_weights, test_preds))
final_pred = np.clip(final_pred, 0, None)

submission = pd.DataFrame({'ID': test['ID'], TARGET: final_pred})
submission.to_csv('./submission_v2_weighted.csv', index=False)
print(f"\nsubmission_v2_weighted.csv 저장 완료")
print(f"예측값: mean={final_pred.mean():.2f}, median={np.median(final_pred):.2f}")
