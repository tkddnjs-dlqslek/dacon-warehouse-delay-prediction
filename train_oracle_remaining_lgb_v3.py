"""
Oracle-Remaining-LGB-v3: Enhanced suffix oracle with std + q75 + layout context.
Adds distributional shape of future delays beyond mean/max.
Kill if fold1 > 8.895.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
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

def compute_suffix_stats(df_sorted, col, global_fill):
    n_rows = len(df_sorted)
    r_mean = np.full(n_rows, global_fill, dtype=np.float64)
    r_max  = np.full(n_rows, global_fill, dtype=np.float64)
    r_std  = np.full(n_rows, 0.0,         dtype=np.float64)
    r_q75  = np.full(n_rows, global_fill, dtype=np.float64)
    for (lid, sid), grp in df_sorted.groupby(['layout_id','scenario_id']):
        idx = grp.index.values; vals = grp[col].values; n = len(vals)
        rev_cumsum = np.cumsum(vals[::-1])[::-1]
        rev_cumss  = np.cumsum((vals**2)[::-1])[::-1]  # sum of squares for std
        for t in range(n):
            rem = n - t - 1
            if rem > 0:
                suf = vals[t+1:]
                s   = rev_cumsum[t] - vals[t]
                r_mean[idx[t]] = s / rem
                r_max[idx[t]]  = suf.max()
                r_std[idx[t]]  = suf.std()
                r_q75[idx[t]]  = np.percentile(suf, 75)
            else:
                r_mean[idx[t]] = global_fill
                r_max[idx[t]]  = global_fill
                r_std[idx[t]]  = 0.0
                r_q75[idx[t]]  = global_fill
    return r_mean, r_max, r_std, r_q75

print("Computing suffix stats (true)...", flush=True)
train_raw['y'] = y_true
train_sorted = train_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
tr_sorted_idx = train_sorted['index'].values
train_sorted_tmp = train_raw.iloc[tr_sorted_idx].copy(); train_sorted_tmp['y'] = y_true[tr_sorted_idx]
sm_tr, sx_tr, ss_tr, sq_tr = compute_suffix_stats(train_sorted_tmp, 'y', global_mean)
for i_s, i_o in enumerate(tr_sorted_idx):
    train_raw.at[i_o, 'sc_mean_remaining'] = sm_tr[i_s]
    train_raw.at[i_o, 'sc_max_remaining']  = sx_tr[i_s]
    train_raw.at[i_o, 'sc_std_remaining']  = ss_tr[i_s]
    train_raw.at[i_o, 'sc_q75_remaining']  = sq_tr[i_s]
for c in ['sc_mean_remaining','sc_max_remaining','sc_std_remaining','sc_q75_remaining']:
    train_raw[c] = train_raw[c].fillna(global_mean) if c != 'sc_std_remaining' else train_raw[c].fillna(0)
train_raw['sc_mean_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].transform('mean')
train_raw['ly_mean_y'] = train_raw.groupby('layout_id')['y'].transform('mean')

print("Computing suffix stats (proxy train)...", flush=True)
train_raw['mega33_y'] = mega_oof_id
train_sorted_proxy = train_raw.iloc[tr_sorted_idx].copy()
sm_pr, sx_pr, ss_pr, sq_pr = compute_suffix_stats(train_sorted_proxy, 'mega33_y', global_mean)
for i_s, i_o in enumerate(tr_sorted_idx):
    train_raw.at[i_o, 'sc_mean_remaining_proxy'] = sm_pr[i_s]
    train_raw.at[i_o, 'sc_max_remaining_proxy']  = sx_pr[i_s]
    train_raw.at[i_o, 'sc_std_remaining_proxy']  = ss_pr[i_s]
    train_raw.at[i_o, 'sc_q75_remaining_proxy']  = sq_pr[i_s]
train_raw['sc_mean_proxy'] = train_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')
train_raw['ly_mean_proxy'] = train_raw.groupby('layout_id')['mega33_y'].transform('mean')

print("Computing suffix stats (proxy test)...", flush=True)
test_raw['mega33_y'] = mega_test_id
test_sorted = test_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
te_sorted_idx = test_sorted['index'].values
test_sorted_tmp = test_raw.iloc[te_sorted_idx].copy()
sm_te, sx_te, ss_te, sq_te = compute_suffix_stats(test_sorted_tmp, 'mega33_y', global_mean)
for i_s, i_o in enumerate(te_sorted_idx):
    test_raw.at[i_o, 'sc_mean_remaining_proxy'] = sm_te[i_s]
    test_raw.at[i_o, 'sc_max_remaining_proxy']  = sx_te[i_s]
    test_raw.at[i_o, 'sc_std_remaining_proxy']  = ss_te[i_s]
    test_raw.at[i_o, 'sc_q75_remaining_proxy']  = sq_te[i_s]
test_raw['sc_mean_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')
test_raw['ly_mean_proxy'] = test_raw.groupby('layout_id')['mega33_y'].transform('mean')

ORACLE_COLS = ['sc_mean_remaining','sc_max_remaining','sc_std_remaining','sc_q75_remaining','sc_mean_y','ly_mean_y']
PROXY_COLS  = ['sc_mean_remaining_proxy','sc_max_remaining_proxy','sc_std_remaining_proxy','sc_q75_remaining_proxy','sc_mean_proxy','ly_mean_proxy']

row_sc_arr = train_raw['row_in_sc'].values
# X_base_tr and X_base_te already built as float32 above

def make_X(base, sc_feat, row_sc):
    return np.hstack([base, sc_feat, row_sc.reshape(-1,1)])

LGB_PARAMS = dict(
    objective='mae', metric='mae', n_estimators=3000, learning_rate=0.05,
    max_depth=7, num_leaves=63, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=4, random_state=42, verbosity=-1
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof = np.full(len(train_raw), np.nan)
test_list = []

print("Training oracle-Remaining LGB...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_base_tr[tr_idx],  train_raw.iloc[tr_idx][ORACLE_COLS].values,  row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx], train_raw.iloc[val_idx][ORACLE_COLS].values, row_sc_arr[val_idx])
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

    val_idx_sorted = np.sort(val_idx)
    proxy_val = train_raw.iloc[val_idx_sorted][PROXY_COLS].values
    rsc_val   = row_sc_arr[val_idx_sorted]
    fold_pred = np.maximum(0, model.predict(make_X(X_base_tr[val_idx_sorted], proxy_val, rsc_val)))
    oof[val_idx_sorted] = fold_pred

    proxy_te  = test_raw[PROXY_COLS].values
    rsc_te    = test_raw['row_in_sc'].values
    test_pred = np.maximum(0, model.predict(make_X(X_base_te, proxy_te, rsc_te)))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-Remaining-LGB-v3={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_remaining_v3.npy', oof)
np.save('results/oracle_seq/test_C_lgb_remaining_v3.npy', test_avg)
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
print(f"\noracle-Remaining-LGB-v3 OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+Remaining-LGB: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+Remaining-LGB: w={best_w4:.2f} MAE={best_m4:.4f} delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
