"""
누수 없는 깨끗한 모델 — layout merge + 안전한 피처만 사용
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb

print("데이터 로드...")
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# layout merge
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

# layout_type encoding
le = LabelEncoder()
combined = pd.concat([train['layout_type'], test['layout_type']])
le.fit(combined)
train['layout_type_enc'] = le.transform(train['layout_type'])
test['layout_type_enc'] = le.transform(test['layout_type'])

TARGET = 'avg_delay_minutes_next_30m'
exclude = ['ID', 'layout_id', 'scenario_id', TARGET, 'layout_type']
feature_cols = [c for c in train.columns if c not in exclude]
print(f'피처 수: {len(feature_cols)}')

# 5-fold CV
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
test_pred = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(kf.split(train)):
    print(f'Fold {fold+1}...', flush=True)
    model = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.05,
        num_leaves=127, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(
        train.loc[tr_idx, feature_cols], train.loc[tr_idx, TARGET],
        eval_set=[(train.loc[val_idx, feature_cols], train.loc[val_idx, TARGET])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
    )
    oof[val_idx] = model.predict(train.loc[val_idx, feature_cols])
    test_pred += model.predict(test[feature_cols]) / 5

mae = mean_absolute_error(train[TARGET], oof)
print(f'\nClean OOF MAE: {mae:.4f}')

sub = pd.DataFrame({'ID': test['ID'], TARGET: np.clip(test_pred, 0, None)})
sub.to_csv('./submission_v4_clean.csv', index=False)
print(f'submission_v4_clean.csv 저장 완료')
print(f'pred: mean={test_pred.mean():.2f}, median={np.median(test_pred):.2f}')
