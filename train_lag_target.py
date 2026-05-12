"""
Autoregressive model: target lag (y_{t-1}) as feature.
Key: each scenario's 25 rows are ordered by ID consecutively.
lag1_y -> MAE 3.94, corr=0.87, residual_corr(mega33)=0.38
Sequential prediction (vectorized by row_in_sc position for speed).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# ────────────────────────────
# 1. 데이터 로드 (ID 순서 정렬)
# ────────────────────────────
print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)

# row_in_sc: scenario 내 위치 (0~24)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

# ────────────────────────────
# 2. v30 FE 로드
# ────────────────────────────
print("Loading v30 FE...", flush=True)
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_train_cache = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_test_cache = pickle.load(f)

feat_cols = fe_train_cache['feat_cols']

# train FE: ID로 재정렬
fe_train_df = fe_train_cache['train_fe']  # layout,scenario 순서
fe_train_df = fe_train_df.set_index('ID').loc[train_raw['ID'].values].reset_index()

# test FE: dict of Series → DataFrame → ID 순서로 정렬
fe_test_df = pd.DataFrame(fe_test_cache)
fe_test_df = fe_test_df.set_index('ID').loc[test_raw['ID'].values].reset_index()

print(f"FE: train={fe_train_df.shape}, test={fe_test_df.shape}", flush=True)

# ────────────────────────────
# 3. Target lag features (y_{t-1}, y_{t-2})
# ────────────────────────────
LAG_COLS = ['lag1_y', 'lag2_y', 'row_in_sc']

# train: 실제 lag (첫 row NaN → global_mean)
train_raw['lag1_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(1).fillna(global_mean))
train_raw['lag2_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(2).fillna(global_mean))

# X 구성
X_all_cols = feat_cols + LAG_COLS

X_train_base = fe_train_df[feat_cols].values  # (250000, 149)
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
train_row_sc = train_raw['row_in_sc'].values

X_test_base  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].values  # (50000, 149)
test_row_sc  = test_raw['row_in_sc'].values

def make_X(base_arr, lag1_arr, lag2_arr, row_sc_arr):
    """base features + [lag1, lag2, row_in_sc] 결합"""
    extra = np.column_stack([lag1_arr, lag2_arr, row_sc_arr])
    return np.hstack([base_arr, extra])

# ────────────────────────────
# 4. LGB 파라미터
# ────────────────────────────
LGB_PARAMS = dict(
    objective='mae', n_estimators=2000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

# ────────────────────────────
# 5. GroupKFold (sequential OOF - 벡터화)
# ────────────────────────────
print("Training...", flush=True)
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_preds     = np.full(len(train_raw), np.nan)
test_pred_list = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx],
                  train_lag2[tr_idx], train_row_sc[tr_idx])
    y_tr = y_true[tr_idx]
    X_val_dummy = make_X(X_train_base[val_idx], train_lag1[val_idx],
                         train_lag2[val_idx], train_row_sc[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_dummy, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)])

    # ── Sequential OOF (벡터화: position별 batch) ──
    # val_idx를 [layout, scenario, row_in_sc] 순으로 정렬
    # → 정렬 후 같은 scenario의 pos=0,1,...,24가 연속으로 붙어 있음
    # → pos=p 마스크와 pos=p-1 마스크는 같은 scenario 집합을 같은 순서로 가리킴
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df_tmp['_orig'].values          # scenario+pos 순서의 원래 인덱스
    row_in_sc_vals = val_df_tmp['row_in_sc'].values  # 0~24 반복 (scenario 순)

    fold_oof = np.zeros(len(val_sorted))

    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]

        if pos == 0:
            lag1_fill = np.full(pos_mask.sum(), global_mean)
            lag2_fill = np.full(pos_mask.sum(), global_mean)
        else:
            # 같은 scenario 집합, pos-1 위치 → 같은 순서 → 그대로 읽으면 됨
            lag1_fill = fold_oof[row_in_sc_vals == (pos - 1)]
            lag2_fill = (fold_oof[row_in_sc_vals == (pos - 2)]
                         if pos >= 2 else np.full(pos_mask.sum(), global_mean))

        X_pos = make_X(X_train_base[pos_idx], lag1_fill, lag2_fill,
                       np.full(pos_mask.sum(), pos))
        fold_oof[pos_mask] = np.maximum(0, model.predict(X_pos))

    fold_mae = np.mean(np.abs(fold_oof - y_true[val_sorted]))

    # 원래 인덱스로 복원
    oof_preds[val_sorted] = fold_oof

    fold_mae = np.mean(np.abs(fold_oof - y_true[val_sorted]))
    elapsed  = time.time() - t0
    print(f"Fold {fold_i+1}: OOF_MAE={fold_mae:.4f} ({elapsed:.0f}s)", flush=True)

    # ── Sequential Test Prediction ──
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted    = test_df_tmp['_orig'].values
    test_rsc_vals  = test_df_tmp['row_in_sc'].values

    test_pred_sorted = np.zeros(len(test_raw))

    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]

        if pos == 0:
            l1 = np.full(pos_mask.sum(), float(y_tr.mean()))
            l2 = l1.copy()
        else:
            l1 = test_pred_sorted[test_sorted[test_rsc_vals == (pos-1)]]
            l2 = (test_pred_sorted[test_sorted[test_rsc_vals == (pos-2)]]
                  if pos >= 2 else np.full(pos_mask.sum(), float(y_tr.mean())))

        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(pos_mask.sum(), pos))
        preds = np.maximum(0, model.predict(X_pos))
        test_pred_sorted[pos_idx] = preds

    test_pred_list.append(test_pred_sorted)

# ────────────────────────────
# 6. 저장
# ────────────────────────────
os.makedirs('results/lag_target', exist_ok=True)

# OOF: ID 순서 → layout,scenario 순서로 변환 (mega33과 비교 위해)
lag_mae = np.mean(np.abs(oof_preds - y_true))
print(f"\n=== OOF MAE (lag_target, sequential) = {lag_mae:.4f} ===", flush=True)

# mega33 residual corr 계산
with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_id = {row['ID']:i for i,row in train_ls.iterrows()}
id_order_in_ls = [ls_to_id[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_order_in_ls]
mega_mae_id = np.mean(np.abs(mega_oof_id - y_true))

res_corr = np.corrcoef(oof_preds - y_true, mega_oof_id - y_true)[0,1]
print(f"mega33 OOF MAE (ID order) = {mega_mae_id:.4f}", flush=True)
print(f"residual_corr(lag, mega33) = {res_corr:.4f}", flush=True)

# blend 최적 가중치 계산
best_blend_mae = 9999; best_w = 0
for w in np.arange(0, 1.01, 0.05):
    b = w * oof_preds + (1-w) * mega_oof_id
    m = np.mean(np.abs(b - y_true))
    if m < best_blend_mae:
        best_blend_mae = m; best_w = w
print(f"Best blend (lag_w={best_w:.2f}): MAE={best_blend_mae:.4f} "
      f"(delta={best_blend_mae-mega_mae_id:.4f})", flush=True)

# ID 순서로 저장
np.save('results/lag_target/lag_oof.npy', oof_preds)

# test pred: 5-fold average, ID 순서
test_avg = np.mean(test_pred_list, axis=0)
np.save('results/lag_target/lag_test.npy', test_avg)

summary = {
    'oof_mae': float(lag_mae), 'mega33_mae': float(mega_mae_id),
    'delta': float(lag_mae - mega_mae_id),
    'residual_corr': float(res_corr),
    'best_blend_w_lag': float(best_w), 'best_blend_mae': float(best_blend_mae),
}
with open('results/lag_target/summary.json','w',encoding='utf-8') as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)
print("Done. Saved to results/lag_target/", flush=True)
