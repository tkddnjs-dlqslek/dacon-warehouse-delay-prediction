"""
Oracle-Layout-v2: layout-level aggregates ONLY (no position interaction).
- v1 failed: lp_mean_y (layout×pos) too specific, proxy error 1.33 → fold1=9.04
- v2: only ly_mean_y + ly_std_y (proxy error 0.27, much more robust)
- sc_mean_proxy still included as supporting feature
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

with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']
import gc
_tr = fe_tr['train_fe']; _tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()
_te = pd.DataFrame(fe_te); _te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_base_te = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# True layout-level oracle features
train_raw['y'] = y_true
ly = train_raw.groupby('layout_id')['y']
train_raw['ly_mean_y']  = ly.transform('mean')
train_raw['ly_std_y']   = ly.transform('std').fillna(0)
# Removed ly_max/min and lp_mean (too specific, proxy degraded for unseen layouts)

# Proxy layout-level
train_raw['mega33_y'] = mega_oof_id
mg_ly = train_raw.groupby('layout_id')['mega33_y']
train_raw['ly_mean_proxy']  = mg_ly.transform('mean')
train_raw['ly_std_proxy']   = mg_ly.transform('std').fillna(0)

# SC-level proxy as supporting features
sc_mg = train_raw.groupby(['layout_id','scenario_id'])['mega33_y']
train_raw['sc_mean_proxy'] = sc_mg.transform('mean')
train_raw['sc_std_proxy']  = sc_mg.transform('std').fillna(0)

test_raw['mega33_y'] = mega_test_id
te_mg_ly = test_raw.groupby('layout_id')['mega33_y']
test_raw['ly_mean_proxy']  = te_mg_ly.transform('mean')
test_raw['ly_std_proxy']   = te_mg_ly.transform('std').fillna(0)

te_sc_mg = test_raw.groupby(['layout_id','scenario_id'])['mega33_y']
test_raw['sc_mean_proxy'] = te_sc_mg.transform('mean')
test_raw['sc_std_proxy']  = te_sc_mg.transform('std').fillna(0)

LY_ORACLE = ['ly_mean_y','ly_std_y']
LY_PROXY  = ['ly_mean_proxy','ly_std_proxy']
SC_PROXY_COLS = ['sc_mean_proxy','sc_std_proxy']

row_sc_arr = train_raw['row_in_sc'].values
# X_base_tr and X_base_te already built as float32 above

def make_X(base, oracle_feat, sc_feat, row_sc):
    return np.hstack([base, oracle_feat, sc_feat, row_sc.reshape(-1,1)])

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

print("Training oracle-Layout-v2...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_base_tr[tr_idx],
                   train_raw.iloc[tr_idx][LY_ORACLE].values,
                   train_raw.iloc[tr_idx][SC_PROXY_COLS].values,
                   row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx],
                   train_raw.iloc[val_idx][LY_ORACLE].values,
                   train_raw.iloc[val_idx][SC_PROXY_COLS].values,
                   row_sc_arr[val_idx])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])], verbose=False)

    val_idx_sorted = np.sort(val_idx)
    ly_proxy_val = train_raw.iloc[val_idx_sorted][LY_PROXY].values
    sc_proxy_val = train_raw.iloc[val_idx_sorted][SC_PROXY_COLS].values
    rsc_val      = row_sc_arr[val_idx_sorted]
    fold_pred    = np.maximum(0, model.predict(
        make_X(X_base_tr[val_idx_sorted], ly_proxy_val, sc_proxy_val, rsc_val)))
    oof[val_idx_sorted] = fold_pred

    ly_proxy_te = test_raw[LY_PROXY].values
    sc_proxy_te = test_raw[SC_PROXY_COLS].values
    rsc_te      = test_raw['row_in_sc'].values
    test_pred   = np.maximum(0, model.predict(make_X(X_base_te, ly_proxy_te, sc_proxy_te, rsc_te)))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-Layout-v2={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_layout_v2.npy', oof)
np.save('results/oracle_seq/test_C_xgb_layout_v2.npy', test_avg)
print("Saved.", flush=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]+fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae=np.mean(np.abs(fixed2-y_true)); oof_mae=np.mean(np.abs(oof-y_true))
print(f"\noracle-Layout-v2 OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+Layout: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+Layout: w={best_w4:.2f} MAE={best_m4:.4f} delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
