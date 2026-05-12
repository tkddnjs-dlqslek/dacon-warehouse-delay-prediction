"""
Oracle-XGB-v31-SC: v31 base (335 features) + suffix {mean, max, min, std}.
Combines richer base features with suffix diversity (no lag → diverse signal).
Kill if fold1 > 8.9.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 8.9

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

print("Loading v31 FE cache...", flush=True)
with open('results/eda_v31/v31_fe_cache.pkl', 'rb') as f:
    fe_v31 = pickle.load(f)
feat_cols_v31 = fe_v31['feat_cols']
print(f"  v31 feat_cols: {len(feat_cols_v31)}", flush=True)
import gc
_tr = fe_v31['train_fe']; _tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr = _tr[feat_cols_v31].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord; gc.collect()
_te = fe_v31['test_fe']; _te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_base_te = _te.reindex(columns=feat_cols_v31, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_v31; gc.collect()

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


def compute_suffix_stats_v2(df_sorted, col, global_fill):
    n_total = len(df_sorted)
    r_mean = np.full(n_total, global_fill, dtype=np.float64)
    r_max  = np.full(n_total, global_fill, dtype=np.float64)
    r_min  = np.full(n_total, global_fill, dtype=np.float64)
    r_std  = np.full(n_total, 0.0,         dtype=np.float64)
    for (lid, sid), grp in df_sorted.groupby(['layout_id','scenario_id']):
        idx  = grp.index.values
        vals = grp[col].values.astype(np.float64)
        n = len(vals)
        if n < 2:
            continue
        rev_cs    = np.cumsum(vals[::-1])[::-1]
        rev_cs_sq = np.cumsum((vals**2)[::-1])[::-1]
        rev_max   = np.maximum.accumulate(vals[::-1])[::-1]
        rev_min   = np.minimum.accumulate(vals[::-1])[::-1]
        for t in range(n - 1):
            rem = n - t - 1
            sm  = rev_cs[t+1]
            sm2 = rev_cs_sq[t+1]
            mu  = sm / rem
            r_mean[idx[t]] = mu
            r_max[idx[t]]  = rev_max[t+1]
            r_min[idx[t]]  = rev_min[t+1]
            r_std[idx[t]]  = np.sqrt(max(0.0, sm2/rem - mu**2))
    return r_mean, r_max, r_min, r_std


print("Computing suffix stats (true)...", flush=True)
train_raw['y'] = y_true
train_sorted = train_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
tr_sorted_idx = train_sorted['index'].values
train_sorted_tmp = train_raw.iloc[tr_sorted_idx].copy(); train_sorted_tmp['y'] = y_true[tr_sorted_idx]
sm, smx, smn, ssd = compute_suffix_stats_v2(train_sorted_tmp, 'y', global_mean)
train_raw['sc_mean_remaining'] = 0.0; train_raw['sc_max_remaining']  = 0.0
train_raw['sc_min_remaining']  = 0.0; train_raw['sc_std_remaining']  = 0.0
for i_s, i_o in enumerate(tr_sorted_idx):
    train_raw.at[i_o, 'sc_mean_remaining'] = sm[i_s]
    train_raw.at[i_o, 'sc_max_remaining']  = smx[i_s]
    train_raw.at[i_o, 'sc_min_remaining']  = smn[i_s]
    train_raw.at[i_o, 'sc_std_remaining']  = ssd[i_s]
train_raw['sc_mean_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].transform('mean')

print("Computing suffix stats (proxy train)...", flush=True)
train_raw['mega33_y'] = mega_oof_id
train_sorted_proxy = train_raw.iloc[tr_sorted_idx].copy()
pm, pmx, pmn, psd = compute_suffix_stats_v2(train_sorted_proxy, 'mega33_y', global_mean)
train_raw['sc_mean_rem_proxy'] = 0.0; train_raw['sc_max_rem_proxy'] = 0.0
train_raw['sc_min_rem_proxy']  = 0.0; train_raw['sc_std_rem_proxy'] = 0.0
for i_s, i_o in enumerate(tr_sorted_idx):
    train_raw.at[i_o, 'sc_mean_rem_proxy'] = pm[i_s]
    train_raw.at[i_o, 'sc_max_rem_proxy']  = pmx[i_s]
    train_raw.at[i_o, 'sc_min_rem_proxy']  = pmn[i_s]
    train_raw.at[i_o, 'sc_std_rem_proxy']  = psd[i_s]
train_raw['sc_mean_proxy'] = train_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')

print("Computing suffix stats (proxy test)...", flush=True)
test_raw['mega33_y'] = mega_test_id
test_sorted = test_raw.sort_values(['layout_id','scenario_id','row_in_sc']).reset_index()
te_sorted_idx = test_sorted['index'].values
test_sorted_tmp = test_raw.iloc[te_sorted_idx].copy()
tem, temx, temn, tesd = compute_suffix_stats_v2(test_sorted_tmp, 'mega33_y', global_mean)
test_raw['sc_mean_rem_proxy'] = 0.0; test_raw['sc_max_rem_proxy'] = 0.0
test_raw['sc_min_rem_proxy']  = 0.0; test_raw['sc_std_rem_proxy'] = 0.0
for i_s, i_o in enumerate(te_sorted_idx):
    test_raw.at[i_o, 'sc_mean_rem_proxy'] = tem[i_s]
    test_raw.at[i_o, 'sc_max_rem_proxy']  = temx[i_s]
    test_raw.at[i_o, 'sc_min_rem_proxy']  = temn[i_s]
    test_raw.at[i_o, 'sc_std_rem_proxy']  = tesd[i_s]
test_raw['sc_mean_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].transform('mean')

# Include sc_mean_y (true) at train; sc_mean_proxy (mega33) at inference
ORACLE_COLS = ['sc_mean_remaining','sc_max_remaining','sc_min_remaining','sc_std_remaining','sc_mean_y']
PROXY_COLS  = ['sc_mean_rem_proxy', 'sc_max_rem_proxy', 'sc_min_rem_proxy', 'sc_std_rem_proxy','sc_mean_proxy']

row_sc_arr = train_raw['row_in_sc'].values

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

print("Training oracle-XGB-v31-SC...", flush=True)
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

    proxy_te  = test_raw[PROXY_COLS].values
    rsc_te    = test_raw['row_in_sc'].values
    test_pred = np.maximum(0, model.predict(make_X(X_base_te, proxy_te, rsc_te)))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-XGB-v31-SC={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_v31_sc.npy', oof)
np.save('results/oracle_seq/test_C_xgb_v31_sc.npy', test_avg)
print("Saved.", flush=True)

rem_oof = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
oof_mae = np.mean(np.abs(oof - y_true))
print(f"\noracle-XGB-v31-SC OOF: {oof_mae:.4f}")
print(f"corr(xgb)={np.corrcoef(xgb_oof,oof)[0,1]:.4f}  corr(lv2)={np.corrcoef(lv2_oof,oof)[0,1]:.4f}  corr(rem)={np.corrcoef(rem_oof,oof)[0,1]:.4f}")

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]
       +fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
base5 = 0.64*fixed2 + 0.12*xgb_oof + 0.16*lv2_oof + 0.08*rem_oof
base5_mae = np.mean(np.abs(base5 - y_true))
best_m = base5_mae; best_w = 0
for w in np.arange(0.02, 0.21, 0.02):
    mm = np.mean(np.abs((1-w)*base5 + w*oof - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"best4+v31-SC: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-base5_mae:+.4f}")
print("Done.")
