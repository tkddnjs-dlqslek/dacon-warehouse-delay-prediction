"""
Oracle-Residual: lag features = residual of FIXED blend, not raw y.
- Training: lag1_residual = (y - FIXED)[t-1], lag2_residual = (y - FIXED)[t-2]
- Test proxy: (mega33 - FIXED)[t-1] as proxy residual lag
- Key: residual is zero-mean, smaller scale → less distribution shift than raw y lag
- Also adds sc_mean_y (SC oracle) as complementary feature
Kill if fold1 > 8.85.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 8.85

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

with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']
fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

# FIXED blend in _row_id order
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id  = d['meta_avg_oof'][id_to_lspos]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d['meta_avg_oof'][id_to_lspos]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos])

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]
fixed_test = (fw['mega33']*d['meta_avg_test'][test_id_to_lspos]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[test_id_to_lspos]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[test_id_to_lspos]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[test_id_to_lspos]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[test_id_to_lspos])

# True residuals (oracle)
train_raw['residual'] = y_true - fixed_oof
global_resid_mean = train_raw['residual'].mean()
grp = train_raw.groupby(['layout_id','scenario_id'])['residual']
train_raw['lag1_resid'] = grp.shift(1).fillna(global_resid_mean)
train_raw['lag2_resid'] = grp.shift(2).fillna(global_resid_mean)

# SC oracle: sc_mean_y
train_raw['y'] = y_true
sc_grp = train_raw.groupby(['layout_id','scenario_id'])['y']
train_raw['sc_mean_y']  = sc_grp.transform('mean')
train_raw['sc_std_y']   = sc_grp.transform('std').fillna(0)

# Proxy residuals for test/val: (mega33 - fixed)
train_raw['proxy_resid']  = mega_oof_id - fixed_oof
pgrp = train_raw.groupby(['layout_id','scenario_id'])['proxy_resid']
train_raw['lag1_proxy_resid'] = pgrp.shift(1).fillna(global_resid_mean)
train_raw['lag2_proxy_resid'] = pgrp.shift(2).fillna(global_resid_mean)

test_raw['proxy_resid']  = mega_test_id - fixed_test
te_pgrp = test_raw.groupby(['layout_id','scenario_id'])['proxy_resid']
test_raw['lag1_proxy_resid'] = te_pgrp.shift(1).fillna(global_resid_mean)
test_raw['lag2_proxy_resid'] = te_pgrp.shift(2).fillna(global_resid_mean)

# SC proxy for val/test
train_raw['mega33_y'] = mega_oof_id
mg_grp = train_raw.groupby(['layout_id','scenario_id'])['mega33_y']
train_raw['sc_mean_proxy'] = mg_grp.transform('mean')
train_raw['sc_std_proxy']  = mg_grp.transform('std').fillna(0)

test_raw['mega33_y'] = mega_test_id
te_mg_grp = test_raw.groupby(['layout_id','scenario_id'])['mega33_y']
test_raw['sc_mean_proxy'] = te_mg_grp.transform('mean')
test_raw['sc_std_proxy']  = te_mg_grp.transform('std').fillna(0)

X_base_tr = fe_train_df[feat_cols].fillna(0).values
row_sc_arr = train_raw['row_in_sc'].values
X_base_te  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_base_te.columns: X_base_te[c] = 0.0
X_base_te = X_base_te[feat_cols].fillna(0).values

ORACLE_COLS = ['lag1_resid','lag2_resid','sc_mean_y','sc_std_y']
PROXY_COLS  = ['lag1_proxy_resid','lag2_proxy_resid','sc_mean_proxy','sc_std_proxy']

def make_X(base, extra, row_sc):
    return np.hstack([base, extra, row_sc.reshape(-1,1)])

XGB_PARAMS = dict(
    objective='reg:absoluteerror', n_estimators=3000, learning_rate=0.05,
    max_depth=7, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=4, random_state=42, verbosity=0,
    early_stopping_rounds=100, eval_metric='mae'
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof = np.full(len(train_raw), np.nan)
test_list = []

print("Training oracle-Residual+SC...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_base_tr[tr_idx],  train_raw.iloc[tr_idx][ORACLE_COLS].values,  row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx], train_raw.iloc[val_idx][ORACLE_COLS].values, row_sc_arr[val_idx])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])], verbose=False)

    # OOF: proxy residual lags + proxy SC
    val_df = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig'] = val_idx
    val_df = val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df['_orig'].values; rsc_vals = val_df['row_in_sc'].values
    proxy_val = train_raw.iloc[val_sorted][PROXY_COLS].values

    fold_pred = np.zeros(len(val_sorted))
    # Non-sequential: proxy SC is scenario-level (same for all pos), residual lag uses proxy
    for pos in range(25):
        pm = rsc_vals == pos; pi = val_sorted[pm]; n = pm.sum()
        X_pos = make_X(X_base_tr[pi], proxy_val[pm], np.full(n, pos))
        fold_pred[pm] = np.maximum(0, model.predict(X_pos))
    oof[val_sorted] = fold_pred

    te_df = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    te_df['_orig'] = np.arange(len(test_raw))
    te_df = te_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted = te_df['_orig'].values; te_rsc = te_df['row_in_sc'].values
    proxy_te = test_raw.iloc[te_sorted][PROXY_COLS].values
    test_pred = np.zeros(len(test_raw))
    for pos in range(25):
        pm = te_rsc == pos; pi = te_sorted[pm]; n = pm.sum()
        X_pos = make_X(X_base_te[pi], proxy_te[pm], np.full(n, pos))
        test_pred[pi] = np.maximum(0, model.predict(X_pos))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_sorted]))
    print(f"Fold {fold_i+1}: oracle-Residual={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        import sys; sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_residual.npy', oof)
np.save('results/oracle_seq/test_C_xgb_residual.npy', test_avg)
print("Saved.", flush=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]+fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy'); lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae=np.mean(np.abs(fixed2-y_true)); oof_mae=np.mean(np.abs(oof-y_true))
print(f"\noracle-Residual OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+Residual: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+Residual: w={best_w4:.2f} MAE={best_m4:.4f} delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
