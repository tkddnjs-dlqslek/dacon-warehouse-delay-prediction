"""
Oracle-XGB-v31 (new): v31 FE cache (메모리 안전 패턴) + oracle lag1/lag2/sc_mean.
- train: ORACLE_COLS (진짜 y 기반 lag), val/test: PROXY_COLS (mega33 예측 기반 lag)
- v31 cache: results/eda_v31/v31_fe_cache.pkl (train+test 동일 파일)
- GroupKFold(layout_id, n_splits=5)
- FIXED blend: mega33×0.7637 + rank_adj×0.1589 + iter_r1×0.0119 + iter_r2×0.0346 + iter_r3×0.0310
- Kill if fold1 > 8.85
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import gc

KILL_THRESH = 8.9

print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_', '').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_', '').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id', 'scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id', 'scenario_id']).cumcount()

y_true      = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

# ─── v31 FE 로드 (메모리 안전 패턴) ───────────────────────────────────────────
print("Loading v31 FE cache...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
feat_cols_v31 = fe_v31['feat_cols']
print(f"  v31 feat_cols: {len(feat_cols_v31)}", flush=True)

_tr = fe_v31['train_fe']
_tr_id2idx = {v: i for i, v in enumerate(_tr['ID'].values)}
_tr_ord    = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr  = _tr[feat_cols_v31].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord; gc.collect()

_te = fe_v31['test_fe']
_te_id2idx = {v: i for i, v in enumerate(_te['ID'].values)}
_te_ord    = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_base_te  = _te[feat_cols_v31].values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_v31; gc.collect()

# ─── mega33 proxy ──────────────────────────────────────────────────────────────
with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

train_ls   = pd.read_csv('train.csv').sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
ls_to_pos  = {row['ID']: i for i, row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls      = pd.read_csv('test.csv').sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']: i for i, row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# ─── Oracle / Proxy features ───────────────────────────────────────────────────
# 학습 시: 진짜 y 기반 lag
train_raw['lag1_y'] = (train_raw.groupby(['layout_id', 'scenario_id'])['avg_delay_minutes_next_30m']
                       .shift(1).fillna(global_mean))
train_raw['lag2_y'] = (train_raw.groupby(['layout_id', 'scenario_id'])['avg_delay_minutes_next_30m']
                       .shift(2).fillna(global_mean))
train_raw['sc_mean_y'] = train_raw.groupby(['layout_id', 'scenario_id'])['avg_delay_minutes_next_30m'].transform('mean')

# 추론 시 (OOF val + test): mega33 예측 기반 lag
train_raw['mega33_oof'] = mega_oof_id
train_raw['lag1_proxy'] = (train_raw.groupby(['layout_id', 'scenario_id'])['mega33_oof']
                           .shift(1).fillna(global_mean))
train_raw['lag2_proxy'] = (train_raw.groupby(['layout_id', 'scenario_id'])['mega33_oof']
                           .shift(2).fillna(global_mean))
train_raw['sc_mean_proxy'] = (train_raw.groupby(['layout_id', 'scenario_id'])['mega33_oof']
                               .transform('mean'))

test_raw['mega33_pred']  = mega_test_id
test_raw['lag1_proxy']   = (test_raw.groupby(['layout_id', 'scenario_id'])['mega33_pred']
                             .shift(1).fillna(global_mean))
test_raw['lag2_proxy']   = (test_raw.groupby(['layout_id', 'scenario_id'])['mega33_pred']
                             .shift(2).fillna(global_mean))
test_raw['sc_mean_proxy'] = (test_raw.groupby(['layout_id', 'scenario_id'])['mega33_pred']
                              .transform('mean'))

ORACLE_COLS = ['lag1_y',     'lag2_y',     'sc_mean_y']
PROXY_COLS  = ['lag1_proxy', 'lag2_proxy', 'sc_mean_proxy']
row_sc_arr  = train_raw['row_in_sc'].values

def make_X(base, sc_feat, row_sc):
    return np.hstack([base, sc_feat, row_sc.reshape(-1, 1)])

# ─── XGB 파라미터 ──────────────────────────────────────────────────────────────
XGB_PARAMS = dict(
    objective='reg:absoluteerror',
    n_estimators=3000, learning_rate=0.05,
    max_depth=7, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbosity=0,
    early_stopping_rounds=100,
    eval_metric='mae'
)

# ─── 학습 ──────────────────────────────────────────────────────────────────────
gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof    = np.full(len(train_raw), np.nan)
test_list = []

print("Training oracle-XGB-v31...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    X_tr  = make_X(X_base_tr[tr_idx],
                   train_raw.iloc[tr_idx][ORACLE_COLS].values,
                   row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx],
                   train_raw.iloc[val_idx][ORACLE_COLS].values,
                   row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx],
              eval_set=[(X_val, y_true[val_idx])],
              verbose=False)

    # OOF: proxy로 추론
    val_idx_sorted = np.sort(val_idx)
    proxy_val = train_raw.iloc[val_idx_sorted][PROXY_COLS].values
    rsc_val   = row_sc_arr[val_idx_sorted]
    fold_pred = np.maximum(0, model.predict(
        make_X(X_base_tr[val_idx_sorted], proxy_val, rsc_val)))
    oof[val_idx_sorted] = fold_pred

    # Test
    proxy_te  = test_raw[PROXY_COLS].values
    rsc_te    = test_raw['row_in_sc'].values
    test_pred = np.maximum(0, model.predict(
        make_X(X_base_te, proxy_te, rsc_te)))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-XGB-v31={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_v31.npy', oof)
np.save('results/oracle_seq/test_C_xgb_v31.npy',   test_avg)
print("Saved oof_seqC_xgb_v31.npy / test_C_xgb_v31.npy", flush=True)

# ─── 블렌드 평가 ───────────────────────────────────────────────────────────────
oof_mae = np.mean(np.abs(oof - y_true))

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
ls2 = {row['ID']: i for i, row in train_ls2.iterrows()}
id2 = [ls2[i] for i in train_raw['ID'].values]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed2 = (fw['mega33'] * d['meta_avg_oof'][id2]
        + fw['rank_adj'] * np.load('results/ranking/rank_adj_oof.npy')[id2]
        + fw['iter_r1'] * np.load('results/iter_pseudo/round1_oof.npy')[id2]
        + fw['iter_r2'] * np.load('results/iter_pseudo/round2_oof.npy')[id2]
        + fw['iter_r3'] * np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_mae = np.mean(np.abs(fixed2 - y_true))

xgb_v30_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof     = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_oof     = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')

print(f"\n=== EVAL ===", flush=True)
print(f"oracle-XGB-v31 OOF MAE  : {oof_mae:.4f}", flush=True)
print(f"FIXED OOF MAE           : {fixed_mae:.4f}", flush=True)
print(f"corr(xgb_v30)           : {np.corrcoef(xgb_v30_oof, oof)[0,1]:.4f}", flush=True)
print(f"corr(lv2)               : {np.corrcoef(lv2_oof,     oof)[0,1]:.4f}", flush=True)

# FIXED + v31 2-way
best_m = fixed_mae; best_w = 0
for w in np.arange(0.02, 0.51, 0.02):
    mm = np.mean(np.abs((1 - w) * fixed2 + w * oof - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"FIXED+v31: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m - fixed_mae:+.4f}", flush=True)

# best4 = 0.64*fixed + 0.12*xgb_v30 + 0.16*lv2 + 0.08*rem → +v31 delta
base4     = 0.64 * fixed2 + 0.12 * xgb_v30_oof + 0.16 * lv2_oof + 0.08 * rem_oof
base4_mae = np.mean(np.abs(base4 - y_true))
best_m4 = base4_mae; best_w4 = 0
for w in np.arange(0.02, 0.21, 0.02):
    mm = np.mean(np.abs((1 - w) * base4 + w * oof - y_true))
    if mm < best_m4: best_m4 = mm; best_w4 = w
print(f"best4+v31: w={best_w4:.2f}  MAE={best_m4:.4f}  delta={best_m4 - base4_mae:+.4f}", flush=True)
print("Done.", flush=True)
