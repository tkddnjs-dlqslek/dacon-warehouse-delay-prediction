"""
Quick validation: v31 FE vs v30 FE using LGB 5-fold OOF.
If v31 OOF < v30 OOF by > 0.005, integrate into mega33/oracle pipeline.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = r"C:\Users\user\Desktop\데이콘 4월"

print("Loading v31 cache...", flush=True)
with open(os.path.join(ROOT, "results", "eda_v31", "v31_fe_cache.pkl"), "rb") as f:
    v31 = pickle.load(f)

train_fe  = v31['train_fe']
test_fe   = v31['test_fe']
feat_cols = v31['feat_cols']
fold_idx  = v31['fold_idx']

TARGET = "avg_delay_minutes_next_30m"
y = train_fe[TARGET].values
groups = train_fe["layout_id"].values

print(f"v31 feat_cols: {len(feat_cols)}", flush=True)
print(f"Train shape: {train_fe.shape}", flush=True)

LGB_PARAMS = dict(
    objective='mae',
    n_estimators=3000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=4,
    random_state=42,
    verbose=-1,
)

gkf = GroupKFold(n_splits=5)
oof = np.full(len(train_fe), np.nan)
test_preds = []

X = train_fe[feat_cols].values
X_test = test_fe[feat_cols].values

print("Training v31-LGB (5-fold)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
    )
    oof[val_idx] = np.maximum(0, model.predict(X[val_idx]))
    test_preds.append(np.maximum(0, model.predict(X_test)))
    mae = np.mean(np.abs(oof[val_idx] - y[val_idx]))
    print(f"Fold {fold_i+1}: v31-LGB={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)

oof_mae = np.mean(np.abs(oof - y))
test_avg = np.mean(test_preds, axis=0)
print(f"\nv31-LGB OOF MAE: {oof_mae:.4f}", flush=True)

# Compare with v30
v30_path = os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl")
if os.path.exists(v30_path):
    with open(v30_path, "rb") as f:
        v30 = pickle.load(f)
    # Quick v30 eval with same folds
    feat_cols_v30 = v30['feat_cols']
    train_v30 = v30['train_fe']
    test_v30  = v30['test_fe']
    X_v30 = train_v30[feat_cols_v30].values
    X_test_v30 = test_v30[feat_cols_v30].values
    oof_v30 = np.full(len(train_v30), np.nan)
    test_v30_preds = []
    print(f"\nTraining v30-LGB (5-fold) for comparison...", flush=True)
    for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X_v30, groups=groups)):
        t0 = time.time()
        model_v30 = lgb.LGBMRegressor(**LGB_PARAMS)
        model_v30.fit(
            X_v30[tr_idx], y[tr_idx],
            eval_set=[(X_v30[val_idx], y[val_idx])],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]
        )
        oof_v30[val_idx] = np.maximum(0, model_v30.predict(X_v30[val_idx]))
        test_v30_preds.append(np.maximum(0, model_v30.predict(X_test_v30)))
        mae = np.mean(np.abs(oof_v30[val_idx] - y[val_idx]))
        print(f"Fold {fold_i+1}: v30-LGB={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    oof_v30_mae = np.mean(np.abs(oof_v30 - y))
    print(f"\nv30-LGB OOF MAE: {oof_v30_mae:.4f}", flush=True)
    print(f"v31-LGB OOF MAE: {oof_mae:.4f}", flush=True)
    print(f"Delta (v31-v30): {oof_mae - oof_v30_mae:+.4f}", flush=True)

# Save results
os.makedirs(os.path.join(ROOT, "results", "eda_v31"), exist_ok=True)
np.save(os.path.join(ROOT, "results", "eda_v31", "v31_lgb_oof.npy"), oof)
np.save(os.path.join(ROOT, "results", "eda_v31", "v31_lgb_test.npy"), test_avg)
print(f"Saved v31_lgb_oof.npy, v31_lgb_test.npy", flush=True)

# Correlation with existing FIXED blend
try:
    train_raw = pd.read_csv(os.path.join(ROOT, "train.csv")).sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
    with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
        d = pickle.load(f)
    ls2 = {row['ID']:i for i,row in train_raw.iterrows()}
    id2 = [ls2[i] for i in train_fe['ID'].values]
    fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
              iter_r1=0.011855567572749024, iter_r2=0.03456830669223538, iter_r3=0.031038826035934514)
    mega2  = d['meta_avg_oof'][id2]
    rank2  = np.load(os.path.join(ROOT, "results", "ranking", "rank_adj_oof.npy"))[id2]
    it1    = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round1_oof.npy"))[id2]
    it2    = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round2_oof.npy"))[id2]
    it3    = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round3_oof.npy"))[id2]
    fixed  = fw['mega33']*mega2+fw['rank_adj']*rank2+fw['iter_r1']*it1+fw['iter_r2']*it2+fw['iter_r3']*it3
    y_true = train_raw["avg_delay_minutes_next_30m"].values[id2]
    fixed_mae = np.mean(np.abs(fixed - y_true))
    corr_with_fixed = np.corrcoef(fixed, oof)[0,1]
    print(f"\nFIXED MAE: {fixed_mae:.4f}", flush=True)
    print(f"v31-LGB corr with FIXED: {corr_with_fixed:.4f}", flush=True)
    # Blend search
    best_m = fixed_mae; best_w = 0
    for w in np.arange(0, 0.51, 0.02):
        bl = (1-w)*fixed + w*oof
        mm = np.mean(np.abs(bl - y_true))
        if mm < best_m: best_m = mm; best_w = w
    print(f"Best 1-way blend: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
except Exception as e:
    print(f"Blend eval skipped: {e}", flush=True)

print("Done.", flush=True)
