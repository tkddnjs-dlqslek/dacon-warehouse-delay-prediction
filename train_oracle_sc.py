"""
Oracle-SC: Scenario-level y aggregates as oracle features.
Instead of per-timestep lag (exposure bias ≈ 8.40/step),
use sc_mean_y, sc_max_y, sc_std_y, sc_range_y, sc_rank_y.
Proxy error for sc_mean ≈ 8.40/sqrt(25) ≈ 1.68 → much less distribution shift.
No sequential scan needed (non-autoregressive).
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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# True SC features (training oracle info)
train_raw['y'] = y_true
sc_grp = train_raw.groupby(['layout_id','scenario_id'])['y']
train_raw['sc_mean_y']  = sc_grp.transform('mean')
train_raw['sc_max_y']   = sc_grp.transform('max')
train_raw['sc_std_y']   = sc_grp.transform('std').fillna(0)
train_raw['sc_range_y'] = train_raw['sc_max_y'] - sc_grp.transform('min')
train_raw['sc_rank_y']  = sc_grp.rank(pct=True)

# Proxy SC features for OOF eval (using mega33_oof per scenario)
train_raw['mega33_y'] = mega_oof_id
mg_grp = train_raw.groupby(['layout_id','scenario_id'])['mega33_y']
train_raw['sc_mean_proxy']  = mg_grp.transform('mean')
train_raw['sc_max_proxy']   = mg_grp.transform('max')
train_raw['sc_std_proxy']   = mg_grp.transform('std').fillna(0)
train_raw['sc_range_proxy'] = train_raw['sc_max_proxy'] - mg_grp.transform('min')
train_raw['sc_rank_proxy']  = mg_grp.rank(pct=True)

# Test SC proxy features (using mega33_test)
test_raw['mega33_y'] = mega_test_id
te_grp = test_raw.groupby(['layout_id','scenario_id'])['mega33_y']
test_raw['sc_mean_proxy']  = te_grp.transform('mean')
test_raw['sc_max_proxy']   = te_grp.transform('max')
test_raw['sc_std_proxy']   = te_grp.transform('std').fillna(0)
test_raw['sc_range_proxy'] = test_raw['sc_max_proxy'] - te_grp.transform('min')
test_raw['sc_rank_proxy']  = te_grp.rank(pct=True)

SC_ORACLE_COLS = ['sc_mean_y','sc_max_y','sc_std_y','sc_range_y','sc_rank_y']
SC_PROXY_COLS  = ['sc_mean_proxy','sc_max_proxy','sc_std_proxy','sc_range_proxy','sc_rank_proxy']

# Also add existing lag2 of y (complementary to SC)
train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['y'].shift(2).fillna(global_mean)
test_raw['lag1_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].shift(1).fillna(global_mean)
test_raw['lag2_proxy'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_y'].shift(2).fillna(global_mean)

X_base_tr = fe_train_df[feat_cols].fillna(0).values
row_sc_arr = train_raw['row_in_sc'].values
X_base_te  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_base_te.columns: X_base_te[c] = 0.0
X_base_te = X_base_te[feat_cols].fillna(0).values

def make_X_train(base, sc_oracle, lag1, lag2, row_sc):
    """Training: use TRUE sc oracle features."""
    return np.hstack([base, sc_oracle, np.column_stack([lag1, lag2, row_sc])])

def make_X_proxy(base, sc_proxy, lag1, lag2, row_sc):
    """Val/Test: use proxy sc features."""
    return np.hstack([base, sc_proxy, np.column_stack([lag1, lag2, row_sc])])

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

print("Training oracle-SC (sc_mean/max/std/range/rank + lag1/2)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    sc_oracle_tr = train_raw.iloc[tr_idx][SC_ORACLE_COLS].values
    lag1_tr = train_raw['lag1_y'].values[tr_idx]
    lag2_tr = train_raw['lag2_y'].values[tr_idx]
    X_tr = make_X_train(X_base_tr[tr_idx], sc_oracle_tr, lag1_tr, lag2_tr, row_sc_arr[tr_idx])

    # Val uses TRUE sc oracle for early stopping signal
    sc_oracle_val = train_raw.iloc[val_idx][SC_ORACLE_COLS].values
    lag1_val = train_raw['lag1_y'].values[val_idx]
    lag2_val = train_raw['lag2_y'].values[val_idx]
    X_val_true = make_X_train(X_base_tr[val_idx], sc_oracle_val, lag1_val, lag2_val, row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val_true, y_true[val_idx])], verbose=False)

    # OOF eval: use PROXY sc features (same as test-time condition)
    val_df = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig'] = val_idx
    val_df = val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df['_orig'].values
    rsc_vals   = val_df['row_in_sc'].values

    sc_proxy_val = train_raw.iloc[val_sorted][SC_PROXY_COLS].values
    mega_val     = mega_oof_id[val_sorted]

    fold_pred = np.zeros(len(val_sorted))
    for pos in range(25):
        pm = rsc_vals == pos; pi = val_sorted[pm]; n = pm.sum()
        l1 = np.full(n, global_mean) if pos == 0 else mega_val[rsc_vals == (pos-1)]
        l2 = np.full(n, global_mean) if pos < 2  else mega_val[rsc_vals == (pos-2)]
        X_pos = make_X_proxy(X_base_tr[pi], sc_proxy_val[pm], l1, l2, np.full(n, pos))
        fold_pred[pm] = np.maximum(0, model.predict(X_pos))
    oof[val_sorted] = fold_pred

    # Test
    test_df = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df['_orig'] = np.arange(len(test_raw))
    test_df = test_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted = test_df['_orig'].values; te_rsc = test_df['row_in_sc'].values
    mega_te = mega_test_id[te_sorted]
    sc_proxy_te = test_raw.iloc[te_sorted][SC_PROXY_COLS].values

    test_pred = np.zeros(len(test_raw))
    for pos in range(25):
        pm = te_rsc == pos; pi = te_sorted[pm]; n = pm.sum()
        l1 = np.full(n, global_mean) if pos == 0 else mega_te[te_rsc == (pos-1)]
        l2 = np.full(n, global_mean) if pos < 2  else mega_te[te_rsc == (pos-2)]
        X_pos = make_X_proxy(X_base_te[pi], sc_proxy_te[pm], l1, l2, np.full(n, pos))
        test_pred[pi] = np.maximum(0, model.predict(X_pos))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_sorted]))
    print(f"Fold {fold_i+1}: oracle-SC={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        import sys; sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_sc.npy', oof)
np.save('results/oracle_seq/test_C_xgb_sc.npy', test_avg)
print("Saved oof_seqC_xgb_sc.npy", flush=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]+fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof =np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae=np.mean(np.abs(fixed2-y_true)); oof_mae=np.mean(np.abs(oof-y_true))
print(f"\noracle-SC OOF: {oof_mae:.4f}")
print(f"oracle-XGB-v30: {np.mean(np.abs(xgb_v30-y_true)):.4f}  oracle-lv2: {np.mean(np.abs(lv2_oof-y_true)):.4f}")
print(f"FIXED: {fixed_mae:.4f}")
print(f"SC corr w/ xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}")
print(f"SC corr w/ lv2:     {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+SC: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+SC: w={best_w4:.2f}  MAE={best_m4:.4f}  delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
