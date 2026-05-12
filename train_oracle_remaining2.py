"""
Oracle-Remaining: sc_mean_remaining = mean(y[t+1:]) per timestep.
Forward-anchoring oracle: tells model "remaining future delay is X".
Proxy error: mega33_MAE / sqrt(n_remaining) → decreases as pos advances.
Combined with sc_mean_y for full scenario anchor.
Kill if fold1 > 8.85.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 8.895

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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

def compute_suffix_stats(df_sorted, col, global_fill):
    """Per-row: mean/std of col[t+1:] within scenario. Sorted by row_in_sc."""
    result_mean = np.full(len(df_sorted), global_fill, dtype=np.float64)
    result_max  = np.full(len(df_sorted), global_fill, dtype=np.float64)
    for (lid, sid), grp in df_sorted.groupby(['layout_id','scenario_id']):
        idx = grp.index.values
        vals = grp[col].values
        n = len(vals)
        # suffix mean: mean(vals[t+1:])
        rev_cumsum = np.cumsum(vals[::-1])[::-1]  # rev_cumsum[t] = sum(vals[t:])
        for t in range(n):
            remaining = n - t - 1
            if remaining > 0:
                result_mean[idx[t]] = (rev_cumsum[t] - vals[t]) / remaining
                result_max[idx[t]]  = vals[t+1:].max()
            else:
                result_mean[idx[t]] = global_fill
                result_max[idx[t]]  = global_fill
    return result_mean, result_max

print("Computing suffix stats (true)...", flush=True)
train_sorted = train_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
train_raw['y'] = y_true

# Reindex to sorted order for computation
tr_sorted_idx = train_sorted['index'].values
tr_y_sorted = y_true[tr_sorted_idx]
train_sorted_tmp = train_raw.iloc[tr_sorted_idx].copy()
train_sorted_tmp['y'] = tr_y_sorted

suffix_mean_true, suffix_max_true = compute_suffix_stats(train_sorted_tmp, 'y', global_mean)
# Put back in _row_id order
train_raw['sc_mean_remaining'] = 0.0
train_raw['sc_max_remaining']  = 0.0
for i_sorted, i_orig in enumerate(tr_sorted_idx):
    train_raw.at[i_orig, 'sc_mean_remaining'] = suffix_mean_true[i_sorted]
    train_raw.at[i_orig, 'sc_max_remaining']  = suffix_max_true[i_sorted]

# Also sc_mean_y (full scenario mean)
train_raw['sc_mean_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].transform('mean')

print("Computing suffix stats (proxy train)...", flush=True)
train_raw['mega33_y'] = mega_oof_id
train_sorted_proxy = train_raw.iloc[tr_sorted_idx].copy()
suffix_mean_proxy, suffix_max_proxy = compute_suffix_stats(train_sorted_proxy, 'mega33_y', global_mean)
train_raw['sc_mean_remaining_proxy'] = 0.0
train_raw['sc_max_remaining_proxy']  = 0.0
for i_sorted, i_orig in enumerate(tr_sorted_idx):
    train_raw.at[i_orig, 'sc_mean_remaining_proxy'] = suffix_mean_proxy[i_sorted]
    train_raw.at[i_orig, 'sc_max_remaining_proxy']  = suffix_max_proxy[i_sorted]

train_raw['sc_mean_proxy'] = train_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')

print("Computing suffix stats (proxy test)...", flush=True)
test_raw['mega33_y'] = mega_test_id
test_sorted = test_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
te_sorted_idx = test_sorted['index'].values
test_sorted_tmp = test_raw.iloc[te_sorted_idx].copy()
suffix_mean_te, suffix_max_te = compute_suffix_stats(test_sorted_tmp, 'mega33_y', global_mean)
test_raw['sc_mean_remaining_proxy'] = 0.0
test_raw['sc_max_remaining_proxy']  = 0.0
for i_sorted, i_orig in enumerate(te_sorted_idx):
    test_raw.at[i_orig, 'sc_mean_remaining_proxy'] = suffix_mean_te[i_sorted]
    test_raw.at[i_orig, 'sc_max_remaining_proxy']  = suffix_max_te[i_sorted]
test_raw['sc_mean_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')

ORACLE_COLS = ['sc_mean_remaining','sc_max_remaining','sc_mean_y']
PROXY_COLS  = ['sc_mean_remaining_proxy','sc_max_remaining_proxy','sc_mean_proxy']

X_base_tr = fe_train_df[feat_cols].fillna(0).values
row_sc_arr = train_raw['row_in_sc'].values
X_base_te  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_base_te.columns: X_base_te[c] = 0.0
X_base_te = X_base_te[feat_cols].fillna(0).values

def make_X(base, sc_feat, row_sc):
    return np.hstack([base, sc_feat, row_sc.reshape(-1,1)])

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

print("Training oracle-Remaining...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_base_tr[tr_idx],  train_raw.iloc[tr_idx][ORACLE_COLS].values,  row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx], train_raw.iloc[val_idx][ORACLE_COLS].values, row_sc_arr[val_idx])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])], verbose=False)

    val_idx_sorted = np.sort(val_idx)
    proxy_val = train_raw.iloc[val_idx_sorted][PROXY_COLS].values
    rsc_val   = row_sc_arr[val_idx_sorted]
    fold_pred = np.maximum(0, model.predict(make_X(X_base_tr[val_idx_sorted], proxy_val, rsc_val)))
    oof[val_idx_sorted] = fold_pred

    proxy_te = test_raw[PROXY_COLS].values
    rsc_te   = test_raw['row_in_sc'].values
    test_pred = np.maximum(0, model.predict(make_X(X_base_te, proxy_te, rsc_te)))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-Remaining={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        import sys; sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_remaining.npy', oof)
np.save('results/oracle_seq/test_C_xgb_remaining.npy', test_avg)
print("Saved.", flush=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]+fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy'); lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae=np.mean(np.abs(fixed2-y_true)); oof_mae=np.mean(np.abs(oof-y_true))
print(f"\noracle-Remaining OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+Remaining: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+Remaining: w={best_w4:.2f} MAE={best_m4:.4f} delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
