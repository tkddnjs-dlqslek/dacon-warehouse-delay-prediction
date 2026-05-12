"""
Recall-boosted classifier v2.
scale_pos_weight=200 (vs baseline ~39) → more recall at high P thresholds.
Goal: at p>0.5, get precision 70%+ → can use w=0.2-0.3 specialist blend.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
THRESHOLD = 80.0

print("Loading data...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe = pickle.load(f)
feat_cols = fe['feat_cols']
train_fe = fe['train_fe']
test_fe  = fe['test_fe']
y = train_fe[TARGET].values.astype(np.float64)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
groups = train_fe['layout_id'].values
del fe

y_high = (y > THRESHOLD).astype(int)
n_neg = (y_high == 0).sum(); n_pos = (y_high == 1).sum()
natural_spw = n_neg / n_pos
print(f"  y>80: {n_pos}/{len(y)} = {y_high.mean():.3%}  natural_spw={natural_spw:.1f}", flush=True)

gkf = GroupKFold(n_splits=N_SPLITS)
folds = list(gkf.split(np.arange(len(y)), groups=groups))

clf_oof  = np.zeros(len(y))
clf_test = np.zeros(len(X_te))

# Train multiple spw values to find best recall/precision tradeoff
for spw in [100, 200, 500]:
    oof_spw  = np.zeros(len(y))
    test_spw = np.zeros(len(X_te))
    for i, (tr_idx, val_idx) in enumerate(folds):
        clf = lgb.LGBMClassifier(
            objective='binary', n_estimators=2000, learning_rate=0.03,
            num_leaves=63, max_depth=6, min_child_samples=50,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5,
            scale_pos_weight=spw, random_state=SEED, verbose=-1, n_jobs=-1
        )
        clf.fit(X_tr[tr_idx], y_high[tr_idx],
                eval_set=[(X_tr[val_idx], y_high[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof_spw[val_idx] = clf.predict_proba(X_tr[val_idx])[:, 1]
        test_spw += clf.predict_proba(X_te)[:, 1] / N_SPLITS

    auc = roc_auc_score(y_high, oof_spw)
    print(f"\n  [spw={spw}] AUC={auc:.4f}  P(high)_mean={oof_spw.mean():.4f}  P(high|y>80)={oof_spw[y_high==1].mean():.4f}", flush=True)
    for pct in [90, 95, 97, 99]:
        thresh = np.percentile(oof_spw, pct)
        n_above = (oof_spw > thresh).sum()
        recall = ((oof_spw > thresh) & (y_high == 1)).sum() / n_pos
        precision = ((oof_spw > thresh) & (y_high == 1)).sum() / max(n_above, 1)
        print(f"    top {100-pct}% (p>{thresh:.4f}): n={n_above}  recall={recall:.3f}  precision={precision:.3f}", flush=True)

    # Save best (spw=200 is default target)
    if spw == 200:
        clf_oof  = oof_spw
        clf_test = test_spw

os.makedirs('results/cascade', exist_ok=True)
np.save('results/cascade/clf_v2_oof.npy',  clf_oof)
np.save('results/cascade/clf_v2_test.npy', clf_test)
print(f"\nSaved clf_v2 (spw=200). Done.")
