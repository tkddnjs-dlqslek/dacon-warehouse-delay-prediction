"""
Pseudo-lag model: use mega33 predictions as lag1 feature (no exposure bias).
- Training: lag1 = mega33_oof[t-1] (within same scenario)
- Test:     lag1 = mega33_test[t-1]
No sequential prediction needed → standard OOF loop.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

# ──── v30 FE 로드 ────
print("Loading FE...", flush=True)
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

# ──── mega33 OOF/Test 로드 ────
print("Loading mega33...", flush=True)
with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

# mega33는 layout,scenario 정렬 기준 → ID 순서로 재정렬
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]   # (250000,) ID 순서

# test도 ID 순서 확인 (sample_submission 기준)
sample_sub = pd.read_csv('sample_submission.csv')
mega_test_orig = d['meta_avg_test']  # (50000,) - 어떤 순서?
# test_raw ID 순서로 재배열 (sample_submission → layout,scenario 순 추정)
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = mega_test_orig[test_id_to_lspos]  # (50000,) ID 순서

# ──── Pseudo-lag features ────
print("Building pseudo-lag features...", flush=True)
# train: lag1 = mega33_oof[t-1] (같은 scenario 내)
train_raw['mega33_pred'] = mega_oof_id
train_raw['lag1_mega'] = (train_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(1).fillna(global_mean))
train_raw['lag2_mega'] = (train_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(2).fillna(global_mean))

# train: lag1 = y_true[t-1] (오라클, 성능 상한 확인용)
train_raw['lag1_true'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(1).fillna(global_mean))

# test
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(1).fillna(global_mean))
test_raw['lag2_mega'] = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(2).fillna(global_mean))

# ──── X 구성 ────
LAG_COLS = ['lag1_mega', 'lag2_mega', 'row_in_sc']
X_ALL = feat_cols + LAG_COLS

X_train = pd.concat([
    fe_train_df[feat_cols].reset_index(drop=True),
    train_raw[LAG_COLS].reset_index(drop=True)
], axis=1)
X_test = pd.concat([
    fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].reset_index(drop=True),
    test_raw[LAG_COLS].reset_index(drop=True)
], axis=1)
for c in feat_cols:
    if c not in X_test.columns:
        X_test[c] = 0.0
X_test = X_test[X_ALL]

# ──── 오라클 버전도 체크 ────
X_train_oracle = pd.concat([
    fe_train_df[feat_cols].reset_index(drop=True),
    train_raw[['lag1_true','lag2_mega','row_in_sc']].rename(
        columns={'lag1_true':'lag1_mega'}).reset_index(drop=True)
], axis=1)

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_pred  = np.zeros(len(train_raw))
oof_orac  = np.zeros(len(train_raw))
test_preds = []
fold_results = []

print("Training pseudo-lag model...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X_train, groups=groups)):
    t0 = time.time()

    # ── Pseudo-lag model ──
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train.iloc[tr_idx][X_ALL], y_true[tr_idx],
              eval_set=[(X_train.iloc[val_idx][X_ALL], y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)])

    pred_val = np.maximum(0, model.predict(X_train.iloc[val_idx][X_ALL]))
    oof_pred[val_idx] = pred_val

    # ── Oracle lag model (상한 확인) ──
    model_orac = lgb.LGBMRegressor(**LGB_PARAMS)
    model_orac.fit(X_train_oracle.iloc[tr_idx][X_ALL], y_true[tr_idx],
                   eval_set=[(X_train_oracle.iloc[val_idx][X_ALL], y_true[val_idx])],
                   callbacks=[lgb.early_stopping(100, verbose=False),
                              lgb.log_evaluation(9999)])

    pred_orac = np.maximum(0, model_orac.predict(X_train_oracle.iloc[val_idx][X_ALL]))
    oof_orac[val_idx] = pred_orac

    fold_mae = np.mean(np.abs(pred_val - y_true[val_idx]))
    fold_orac = np.mean(np.abs(pred_orac - y_true[val_idx]))
    elapsed = time.time() - t0

    print(f"Fold {fold_i+1}: pseudo={fold_mae:.4f}  oracle={fold_orac:.4f}  ({elapsed:.0f}s)", flush=True)
    fold_results.append({'fold': fold_i, 'pseudo': float(fold_mae), 'oracle': float(fold_orac)})

    test_preds.append(np.maximum(0, model.predict(X_test[X_ALL])))

# ──── 결과 ────
oof_mae   = np.mean(np.abs(oof_pred  - y_true))
oracle_mae = np.mean(np.abs(oof_orac - y_true))

# mega33과 residual corr
res_corr_pseudo = np.corrcoef(oof_pred - y_true, mega_oof_id - y_true)[0,1]
res_corr_oracle = np.corrcoef(oof_orac - y_true, mega_oof_id - y_true)[0,1]

print(f"\n=== RESULTS ===", flush=True)
print(f"mega33 OOF MAE       = {np.mean(np.abs(mega_oof_id-y_true)):.4f}", flush=True)
print(f"pseudo-lag OOF MAE   = {oof_mae:.4f}  (delta={oof_mae - np.mean(np.abs(mega_oof_id-y_true)):.4f})", flush=True)
print(f"oracle-lag OOF MAE   = {oracle_mae:.4f}  (상한)", flush=True)
print(f"residual_corr (pseudo, mega33) = {res_corr_pseudo:.4f}", flush=True)
print(f"residual_corr (oracle, mega33) = {res_corr_oracle:.4f}", flush=True)

# 블렌딩
for label, oof_x, corr_x in [('pseudo', oof_pred, res_corr_pseudo),
                               ('oracle', oof_orac, res_corr_oracle)]:
    best_mae, best_w = 9999, 0
    for w in np.arange(0, 1.01, 0.02):
        m = np.mean(np.abs(w*oof_x + (1-w)*mega_oof_id - y_true))
        if m < best_mae:
            best_mae, best_w = m, w
    print(f"Best blend [{label}] w={best_w:.2f}: MAE={best_mae:.4f} "
          f"delta={best_mae-np.mean(np.abs(mega_oof_id-y_true)):.4f}", flush=True)

# 저장
os.makedirs('results/pseudolag', exist_ok=True)
np.save('results/pseudolag/pseudo_oof.npy', oof_pred)
np.save('results/pseudolag/oracle_oof.npy', oof_orac)
test_avg = np.mean(test_preds, axis=0)
np.save('results/pseudolag/pseudo_test.npy', test_avg)

with open('results/pseudolag/summary.json','w',encoding='utf-8') as f:
    json.dump({
        'mega33_mae': float(np.mean(np.abs(mega_oof_id-y_true))),
        'pseudo_mae': float(oof_mae), 'oracle_mae': float(oracle_mae),
        'res_corr_pseudo': float(res_corr_pseudo),
        'res_corr_oracle': float(res_corr_oracle),
        'folds': fold_results
    }, f, indent=2)
print("Saved to results/pseudolag/", flush=True)
