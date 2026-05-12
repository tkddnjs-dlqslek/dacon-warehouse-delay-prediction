"""
Oracle-XGB with lag1 ONLY (most conservative — minimum exposure bias).
Hypothesis: single lag → minimal distribution shift → might succeed where lag2 fails at some positions.
Test if this gives any different per-position error pattern vs the standard XGB (2 lags).
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
global_mean = y_true.mean()

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

# Only 1 lag (most conservative)
train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(1).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].fillna(0).values
train_lag1   = train_raw['lag1_y'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].fillna(0).values
test_lag1_mega = test_raw['lag1_mega'].values

def make_X(base, lag1, row_sc):
    return np.hstack([base, np.column_stack([lag1, row_sc])])

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

oof_xl1    = np.full(len(train_raw), np.nan)
test_xl1_list = []

print("Training oracle-XGB-lag1only...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_vtrue = make_X(X_train_base[val_idx], train_lag1[val_idx], row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_true[val_idx])],
              verbose=False)

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted = mega_oof_id[val_sorted]

    foldxl1 = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        l1 = mega_val_sorted[row_in_sc_vals == (pos-1)] if pos > 0 else np.full(n_pos, global_mean)
        X_pos = make_X(X_train_base[pos_idx], l1, np.full(n_pos, pos))
        foldxl1[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_xl1[val_sorted] = foldxl1

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted = mega_test_id[test_sorted]

    testxl1 = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        l1 = mega_test_sorted[test_rsc_vals == (pos-1)] if pos > 0 else np.full(n_pos, global_mean)
        X_pos = make_X(X_test_base[pos_idx], l1, np.full(n_pos, pos))
        testxl1[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_xl1_list.append(testxl1)

    maexl1 = np.mean(np.abs(foldxl1 - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-XGB-lag1only={maexl1:.4f}  best_iter={model.best_iteration}  ({elapsed:.0f}s)", flush=True)

test_xl1_avg = np.mean(test_xl1_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_lag1.npy', oof_xl1)
np.save('results/oracle_seq/test_C_xgb_lag1.npy', test_xl1_avg)
print("Saved oof_seqC_xgb_lag1.npy, test_C_xgb_lag1.npy", flush=True)

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
fw2 = dict(mega33=0.7637,rank_adj=0.1589,iter_r1=0.0119,iter_r2=0.0346,iter_r3=0.0310)
fixed2 = fw2['mega33']*mega_oof2+fw2['rank_adj']*rank_oof2+fw2['iter_r1']*iter1_oof2+fw2['iter_r2']*iter2_oof2+fw2['iter_r3']*iter3_oof2
y2 = y_true
xgb2 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv22 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae = np.mean(np.abs(fixed2-y2))
xl1_mae = np.mean(np.abs(oof_xl1-y2))
res = y2 - fixed2
corr = np.corrcoef(res, oof_xl1-fixed2)[0,1]
print(f"\noracle-XGB-lag1 MAE: {xl1_mae:.4f}  residual_corr={corr:.4f}", flush=True)
print(f"vs XGB-lag2 corr: {np.corrcoef(oof_xl1, xgb2)[0,1]:.4f}", flush=True)
print(f"vs LV2 corr:      {np.corrcoef(oof_xl1, lv22)[0,1]:.4f}", flush=True)
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_xl1
    mm = np.mean(np.abs(bl - y2))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static blend: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
