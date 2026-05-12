"""
Oracle-XGB-log2: XGB with log(y+1) target + 2 log lags.
Hypothesis: log transformation compresses distribution → reduces exposure bias magnitude
            while keeping within XGB's ≤2 lag tolerance.
Different from oracle-XGB (raw lags): errors in log space may be distributed differently.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
y_log  = np.log1p(y_true)
global_mean_log = y_log.mean()
global_mean     = y_true.mean()

with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# Log lags of TRUE y for training
NLAGS = 2
train_raw['log_y'] = y_log
grp = train_raw.groupby(['layout_id','scenario_id'])
for k in range(1, NLAGS+1):
    train_raw[f'lag{k}_logy'] = grp['log_y'].shift(k).fillna(global_mean_log)

# Log lags of mega33 proxy for test
test_raw['log_mega33'] = np.log1p(mega_test_id)
m_grp = test_raw.groupby(['layout_id','scenario_id'])
for k in range(1, NLAGS+1):
    test_raw[f'lag{k}_logmega'] = m_grp['log_mega33'].shift(k).fillna(global_mean_log)

X_train_base = fe_train_df[feat_cols].fillna(0).values
train_lags   = [train_raw[f'lag{k}_logy'].values for k in range(1, NLAGS+1)]
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].fillna(0).values
test_lags   = [test_raw[f'lag{k}_logmega'].values for k in range(1, NLAGS+1)]
test_row_sc = test_raw['row_in_sc'].values

def make_X(base, lags_list, row_sc):
    return np.hstack([base, np.column_stack(lags_list + [row_sc])])

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

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_xl2    = np.full(len(train_raw), np.nan)
test_xl2_list = []

print("Training oracle-XGB-log2...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], [train_lags[k][tr_idx] for k in range(NLAGS)], row_sc_arr[tr_idx])
    y_tr = y_log[tr_idx]
    X_vtrue = make_X(X_train_base[val_idx], [train_lags[k][val_idx] for k in range(NLAGS)], row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_log[val_idx])],
              verbose=False)

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted_log = np.log1p(mega_oof_id[val_sorted])

    foldxl2 = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_val_sorted_log[row_in_sc_vals == (pos-k)])
        X_pos = make_X(X_train_base[pos_idx], lags_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        foldxl2[pos_mask] = np.maximum(0, np.expm1(pred_log))
    oof_xl2[val_sorted] = foldxl2

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted_log = np.log1p(mega_test_id[test_sorted])

    testxl2 = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_test_sorted_log[test_rsc_vals == (pos-k)])
        X_pos = make_X(X_test_base[pos_idx], lags_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        testxl2[pos_idx] = np.maximum(0, np.expm1(pred_log))
    test_xl2_list.append(testxl2)

    maexl2 = np.mean(np.abs(foldxl2 - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-XGB-log2={maexl2:.4f}  best_iter={model.best_iteration}  ({elapsed:.0f}s)", flush=True)

test_xl2_avg = np.mean(test_xl2_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_log2.npy', oof_xl2)
np.save('results/oracle_seq/test_C_xgb_log2.npy', test_xl2_avg)
print("Saved oof_seqC_xgb_log2.npy, test_C_xgb_log2.npy", flush=True)

# Quick summary
with open('results/mega33_final.pkl','rb') as f:
    d2 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id_to_ls2 = [ls_pos2[rid] for rid in train_raw['ID'].values]
mega_oof2  = d2['meta_avg_oof'][id_to_ls2]
rank_oof2  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls2]
iter1_oof2 = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls2]
iter2_oof2 = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls2]
iter3_oof2 = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls2]
fw2 = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
           iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
           iter_r3=0.031038826035934514)
fixed2 = (fw2['mega33']*mega_oof2 + fw2['rank_adj']*rank_oof2 +
          fw2['iter_r1']*iter1_oof2 + fw2['iter_r2']*iter2_oof2 + fw2['iter_r3']*iter3_oof2)
y2 = train_raw['avg_delay_minutes_next_30m'].values
xgb_orig = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv22     = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
xl2_mae  = np.mean(np.abs(oof_xl2 - y2))
fixed_mae = np.mean(np.abs(fixed2 - y2))
res_fixed = y2 - fixed2
corr_xl2  = np.corrcoef(res_fixed, oof_xl2 - fixed2)[0,1]
print(f"\noracle-XGB-log2 MAE: {xl2_mae:.4f}  residual_corr={corr_xl2:.4f}", flush=True)
print(f"vs XGB-raw corr:     {np.corrcoef(oof_xl2, xgb_orig)[0,1]:.4f}", flush=True)
print(f"vs LV2 corr:         {np.corrcoef(oof_xl2, lv22)[0,1]:.4f}", flush=True)
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_xl2
    mm = np.mean(np.abs(bl - y2))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static blend: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
