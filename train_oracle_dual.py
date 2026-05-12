"""
Oracle-Dual: Combines future suffix + past prefix + layout context simultaneously.
At training: true y for both past (causal) and future (oracle) features.
At test: mega33 proxy for all. Most complete scenario context possible.
Kill if fold1 > 8.895.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc
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

def compute_suffix_stats(df_sorted, col, fill):
    r_mean = np.full(len(df_sorted), fill)
    r_max  = np.full(len(df_sorted), fill)
    for (lid, sid), grp in df_sorted.groupby(['layout_id','scenario_id']):
        idx = grp.index.values; vals = grp[col].values; n = len(vals)
        rev_cs = np.cumsum(vals[::-1])[::-1]
        for t in range(n):
            rem = n - t - 1
            if rem > 0:
                r_mean[idx[t]] = (rev_cs[t] - vals[t]) / rem
                r_max[idx[t]]  = vals[t+1:].max()
    return r_mean, r_max

def compute_prefix_stats(df_sorted, col, fill):
    r_mean = np.full(len(df_sorted), fill)
    r_max  = np.full(len(df_sorted), fill)
    for (lid, sid), grp in df_sorted.groupby(['layout_id','scenario_id']):
        idx = grp.index.values; vals = grp[col].values
        cs = np.cumsum(vals)
        for t in range(len(vals)):
            if t > 0:
                r_mean[idx[t]] = cs[t-1] / t
                r_max[idx[t]]  = vals[:t].max()
    return r_mean, r_max

# --- TRUE oracle features ---
print("Computing oracle features (true)...", flush=True)
train_raw['y'] = y_true
tr_sorted = train_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
tr_idx_s = tr_sorted['index'].values
tr_cp = train_raw.iloc[tr_idx_s].copy(); tr_cp['y'] = y_true[tr_idx_s]

sm_tr, sx_tr = compute_suffix_stats(tr_cp, 'y', global_mean)
pm_tr, px_tr = compute_prefix_stats(tr_cp, 'y', global_mean)

for i_s, i_o in enumerate(tr_idx_s):
    train_raw.at[i_o, 'rem_mean_y'] = sm_tr[i_s]
    train_raw.at[i_o, 'rem_max_y']  = sx_tr[i_s]
    train_raw.at[i_o, 'cum_mean_y'] = pm_tr[i_s]
    train_raw.at[i_o, 'cum_max_y']  = px_tr[i_s]

train_raw['sc_mean_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].transform('mean')
train_raw['ly_mean_y'] = train_raw.groupby('layout_id')['y'].transform('mean')
del sm_tr, sx_tr, pm_tr, px_tr, tr_cp; gc.collect()

# --- PROXY features (train OOF eval) ---
print("Computing proxy features (train)...", flush=True)
train_raw['mega33_y'] = mega_oof_id
tr_proxy = train_raw.iloc[tr_idx_s].copy()
sm_pr, sx_pr = compute_suffix_stats(tr_proxy, 'mega33_y', global_mean)
pm_pr, px_pr = compute_prefix_stats(tr_proxy, 'mega33_y', global_mean)

for i_s, i_o in enumerate(tr_idx_s):
    train_raw.at[i_o, 'rem_mean_proxy'] = sm_pr[i_s]
    train_raw.at[i_o, 'rem_max_proxy']  = sx_pr[i_s]
    train_raw.at[i_o, 'cum_mean_proxy'] = pm_pr[i_s]
    train_raw.at[i_o, 'cum_max_proxy']  = px_pr[i_s]

train_raw['sc_mean_proxy'] = train_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')
train_raw['ly_mean_proxy'] = train_raw.groupby('layout_id')['mega33_y'].transform('mean')
del sm_pr, sx_pr, pm_pr, px_pr, tr_proxy; gc.collect()

# --- PROXY features (test) ---
print("Computing proxy features (test)...", flush=True)
test_raw['mega33_y'] = mega_test_id
te_sorted = test_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
te_idx_s = te_sorted['index'].values
te_cp = test_raw.iloc[te_idx_s].copy()
sm_te, sx_te = compute_suffix_stats(te_cp, 'mega33_y', global_mean)
pm_te, px_te = compute_prefix_stats(te_cp, 'mega33_y', global_mean)

for i_s, i_o in enumerate(te_idx_s):
    test_raw.at[i_o, 'rem_mean_proxy'] = sm_te[i_s]
    test_raw.at[i_o, 'rem_max_proxy']  = sx_te[i_s]
    test_raw.at[i_o, 'cum_mean_proxy'] = pm_te[i_s]
    test_raw.at[i_o, 'cum_max_proxy']  = px_te[i_s]

test_raw['sc_mean_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')
test_raw['ly_mean_proxy'] = test_raw.groupby('layout_id')['mega33_y'].transform('mean')
del sm_te, sx_te, pm_te, px_te, te_cp; gc.collect()

ORACLE_COLS = ['rem_mean_y','rem_max_y','cum_mean_y','cum_max_y','sc_mean_y','ly_mean_y']
PROXY_COLS  = ['rem_mean_proxy','rem_max_proxy','cum_mean_proxy','cum_max_proxy','sc_mean_proxy','ly_mean_proxy']

row_sc_arr = train_raw['row_in_sc'].values

def make_X(base, oracle_feat, row_sc):
    return np.hstack([base, oracle_feat, row_sc.reshape(-1,1)])

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

print("Training oracle-Dual (LGB)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_base_tr[tr_idx],  train_raw.iloc[tr_idx][ORACLE_COLS].values,  row_sc_arr[tr_idx])
    X_val = make_X(X_base_tr[val_idx], train_raw.iloc[val_idx][ORACLE_COLS].values, row_sc_arr[val_idx])
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

    val_idx_sorted = np.sort(val_idx)
    proxy_val = train_raw.iloc[val_idx_sorted][PROXY_COLS].values
    fold_pred = np.maximum(0, model.predict(make_X(X_base_tr[val_idx_sorted], proxy_val, row_sc_arr[val_idx_sorted])))
    oof[val_idx_sorted] = fold_pred

    proxy_te = test_raw[PROXY_COLS].values
    test_pred = np.maximum(0, model.predict(make_X(X_base_te, proxy_te, test_raw['row_in_sc'].values)))
    test_list.append(test_pred)
    del X_tr, X_val; gc.collect()

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-Dual={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_dual.npy', oof)
np.save('results/oracle_seq/test_C_lgb_dual.npy', test_avg)
print("Saved.", flush=True)

# Eval
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]
       +fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_oof=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oof_mae=np.mean(np.abs(oof-y_true)); fixed_mae=np.mean(np.abs(fixed2-y_true))
print(f"\noracle-Dual OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}  corr rem: {np.corrcoef(rem_oof,oof)[0,1]:.4f}")
base5=0.64*fixed2+0.12*xgb_v30+0.16*lv2_oof+0.08*rem_oof
best_m=np.mean(np.abs(base5-y_true)); best_w=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"base5+Dual: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
