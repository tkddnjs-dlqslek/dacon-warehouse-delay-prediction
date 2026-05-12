"""
Step 3: 베이스라인 모델 — 원본 피처 LightGBM 5-Fold
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# ── 데이터 로드 ──
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')

TARGET = 'avg_delay_minutes_next_30m'
ID_COLS = ['ID', 'layout_id', 'scenario_id']
feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET]]

print(f"피처 수: {len(feature_cols)}")
print(f"Train: {train.shape}, Test: {test.shape}")

# ── 5-Fold CV ──
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(train))
test_preds = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f"\n── Fold {fold + 1} ──")
    X_tr = train.loc[tr_idx, feature_cols]
    y_tr = train.loc[tr_idx, TARGET]
    X_val = train.loc[val_idx, feature_cols]
    y_val = train.loc[val_idx, TARGET]

    model = LGBMRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=7,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1, random_state=42, verbose=-1,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    oof_preds[val_idx] = model.predict(X_val)
    test_preds += model.predict(test[feature_cols]) / 5

# ── 결과 ──
oof_mae = mean_absolute_error(train[TARGET], oof_preds)
print(f"\n{'='*50}")
print(f"[Baseline] OOF MAE: {oof_mae:.4f}")
print(f"{'='*50}")

# ── 베이스라인 제출 ──
submission = pd.DataFrame({'ID': test['ID'], TARGET: np.clip(test_preds, 0, None)})
submission.to_csv('./submission_v1_baseline.csv', index=False)
print("submission_v1_baseline.csv 저장 완료.")
