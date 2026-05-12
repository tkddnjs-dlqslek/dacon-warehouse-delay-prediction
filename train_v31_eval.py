"""v31 LGB OOF + blend eval (no v30 comparison)"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

ROOT = r"C:\Users\user\Desktop\데이콘 4월"

with open(os.path.join(ROOT, "results", "eda_v31", "v31_fe_cache.pkl"), "rb") as f:
    v31 = pickle.load(f)
train_fe  = v31['train_fe']
test_fe   = v31['test_fe']
feat_cols = v31['feat_cols']

TARGET = "avg_delay_minutes_next_30m"
y = train_fe[TARGET].values
groups = train_fe["layout_id"].values
print(f"v31 features: {len(feat_cols)}", flush=True)

LGB_PARAMS = dict(objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=127, min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=4, random_state=42, verbose=-1)

gkf = GroupKFold(n_splits=5)
oof = np.full(len(train_fe), np.nan)
test_preds = []
X = train_fe[feat_cols].values
X_test = test_fe[feat_cols].values

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X, groups=groups)):
    t0 = time.time()
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X[tr_idx], y[tr_idx],
        eval_set=[(X[val_idx], y[val_idx])],
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])
    oof[val_idx] = np.maximum(0, model.predict(X[val_idx]))
    test_preds.append(np.maximum(0, model.predict(X_test)))
    print(f"Fold {fold_i+1}: {np.mean(np.abs(oof[val_idx]-y[val_idx])):.4f}  ({time.time()-t0:.0f}s)", flush=True)

test_avg = np.mean(test_preds, axis=0)
np.save(os.path.join(ROOT, "results", "eda_v31", "v31_lgb_oof.npy"), oof)
np.save(os.path.join(ROOT, "results", "eda_v31", "v31_lgb_test.npy"), test_avg)
print(f"OOF MAE: {np.mean(np.abs(oof-y)):.4f}  — saved", flush=True)

# Blend eval
train_raw = pd.read_csv(os.path.join(ROOT, "train.csv")).sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
y_true = train_raw["avg_delay_minutes_next_30m"].values
with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
    d = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568306692235380, iter_r3=0.031038826035934514)
fixed = (fw['mega33']*d['meta_avg_oof']
       + fw['rank_adj']*np.load(os.path.join(ROOT,"results","ranking","rank_adj_oof.npy"))
       + fw['iter_r1']*np.load(os.path.join(ROOT,"results","iter_pseudo","round1_oof.npy"))
       + fw['iter_r2']*np.load(os.path.join(ROOT,"results","iter_pseudo","round2_oof.npy"))
       + fw['iter_r3']*np.load(os.path.join(ROOT,"results","iter_pseudo","round3_oof.npy")))
xgb_oof = np.load(os.path.join(ROOT,"results","oracle_seq","oof_seqC_xgb.npy"))
lv2_oof = np.load(os.path.join(ROOT,"results","oracle_seq","oof_seqC_log_v2.npy"))

# align v31 OOF to train_raw order
ls2 = {row['ID']:i for i,row in train_raw.iterrows()}
id2 = [ls2[i] for i in train_fe['ID'].values]
v31_oof_aligned = np.zeros(len(train_raw))
v31_oof_aligned[id2] = oof

fixed_mae = np.mean(np.abs(fixed - y_true))
base5 = (1-0.12-0.20)*fixed + 0.12*xgb_oof + 0.20*lv2_oof
base5_mae = np.mean(np.abs(base5 - y_true))
v31_mae = np.mean(np.abs(v31_oof_aligned - y_true))
print(f"\nFIXED MAE:  {fixed_mae:.4f}")
print(f"base5 MAE:  {base5_mae:.4f}  (current best)")
print(f"v31-LGB:    {v31_mae:.4f}")
print(f"corr FIXED: {np.corrcoef(fixed, v31_oof_aligned)[0,1]:.4f}")
print(f"corr XGB:   {np.corrcoef(xgb_oof, v31_oof_aligned)[0,1]:.4f}")
print(f"corr LV2:   {np.corrcoef(lv2_oof, v31_oof_aligned)[0,1]:.4f}")

best_m = fixed_mae; best_w = 0
for w in np.arange(0.02, 0.51, 0.02):
    mm = np.mean(np.abs((1-w)*fixed + w*v31_oof_aligned - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"\nFIXED+v31 best: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}")

best_m4 = base5_mae; best_w4 = 0
for w in np.arange(0.02, 0.21, 0.02):
    mm = np.mean(np.abs((1-w)*base5 + w*v31_oof_aligned - y_true))
    if mm < best_m4: best_m4 = mm; best_w4 = w
print(f"base5+v31 best:  w={best_w4:.2f}  MAE={best_m4:.4f}  delta={best_m4-base5_mae:+.4f}")
print("Done.")
