"""
Adversarial Validation: train vs test 피처 분포 차이 분석
- train/test를 구분하는 분류기를 학습하여 어떤 피처가 가장 다른지 파악
- CV-LB 갭 원인 진단
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

print("=== Adversarial Validation ===", flush=True)

# 데이터 로드
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
layout = pd.read_csv('./layout_info.csv')

# layout 병합
train = train.merge(layout, on='layout_id', how='left')
test = test.merge(layout, on='layout_id', how='left')

# 공통 피처만
exclude = ['ID', 'layout_id', 'scenario_id', 'avg_delay_minutes_next_30m', 'layout_type']
feature_cols = [c for c in train.columns if c not in exclude and c in test.columns]

# train=0, test=1 라벨
train_av = train[feature_cols].copy()
test_av = test[feature_cols].copy()
train_av['is_test'] = 0
test_av['is_test'] = 1

combined = pd.concat([train_av, test_av], axis=0).reset_index(drop=True)
X = combined[feature_cols]
y = combined['is_test']

# 5-fold CV로 AUC 측정
print("\n1. Train vs Test 구분 가능성 (AUC)...", flush=True)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_proba = np.zeros(len(combined))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    model = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    model.fit(X.iloc[tr_idx], y.iloc[tr_idx],
              eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
    oof_proba[val_idx] = model.predict_proba(X.iloc[val_idx])[:, 1]

auc = roc_auc_score(y, oof_proba)
print(f"  AUC: {auc:.4f}", flush=True)
if auc > 0.7:
    print(f"  → train/test 분포 차이가 큼! 피처 보정 필요", flush=True)
elif auc > 0.55:
    print(f"  → 약간의 분포 차이 존재", flush=True)
else:
    print(f"  → train/test 분포가 비슷함 (좋음)", flush=True)

# 피처 중요도 (어떤 피처가 train/test를 가장 잘 구분하는지)
print("\n2. Train/Test 구분에 기여하는 피처 (상위 30개)...", flush=True)

model_full = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=31,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, verbose=-1, n_jobs=-1,
)
model_full.fit(X, y)

fi = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_full.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n  {'피처':<40} {'중요도':>10}")
print(f"  {'-'*50}")
for _, row in fi.head(30).iterrows():
    print(f"  {row['feature']:<40} {row['importance']:>10}", flush=True)

# 상위 분포 차이 피처 상세 분석
print("\n3. 상위 피처 분포 비교 (train vs test)...", flush=True)
for _, row in fi.head(10).iterrows():
    col = row['feature']
    tr_vals = train[col].dropna()
    te_vals = test[col].dropna()
    print(f"\n  {col}:", flush=True)
    print(f"    Train: mean={tr_vals.mean():.3f}, std={tr_vals.std():.3f}, "
          f"median={tr_vals.median():.3f}, [min={tr_vals.min():.3f}, max={tr_vals.max():.3f}]", flush=True)
    print(f"    Test:  mean={te_vals.mean():.3f}, std={te_vals.std():.3f}, "
          f"median={te_vals.median():.3f}, [min={te_vals.min():.3f}, max={te_vals.max():.3f}]", flush=True)

# layout 분석
print("\n4. Layout 분포 분석...", flush=True)
train_layouts = set(train['layout_id'].unique())
test_layouts = set(test['layout_id'].unique())
overlap = train_layouts & test_layouts
unseen = test_layouts - train_layouts
print(f"  Train layouts: {len(train_layouts)}")
print(f"  Test layouts: {len(test_layouts)}")
print(f"  겹침: {len(overlap)}, Unseen: {len(unseen)}")

# layout_type 분포 비교
print("\n  Layout type 분포:")
tr_type = train.groupby('layout_type')['layout_id'].nunique()
te_type = test.groupby('layout_type')['layout_id'].nunique()
for lt in sorted(train['layout_type'].unique()):
    tr_n = tr_type.get(lt, 0)
    te_n = te_type.get(lt, 0)
    print(f"    {lt}: train={tr_n}, test={te_n}", flush=True)

# 결과 저장
fi.to_csv('./results/adversarial_fi.csv', index=False)

print(f"\n{'='*60}")
print(f"Adversarial Validation 완료!")
print(f"  AUC: {auc:.4f}")
print(f"  결과: results/adversarial_fi.csv")
print(f"{'='*60}", flush=True)
