"""
Oracle-XGB with FIXED (5-component blend) as OOF lag proxy instead of mega33.
FIXED MAE=8.3935 < mega33 MAE (~8.40) → less exposure bias → potentially better OOF predictions.
Training: true y lags (same oracle); OOF eval: fixed_oof as lag proxy; Test: fixed_test.
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

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]

# FIXED proxy: use the full 5-way blend instead of mega33 alone
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
mega_oof_id   = d['meta_avg_oof'][id_to_lspos]
rank_oof_id   = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos]
iter1_oof_id  = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos]
iter2_oof_id  = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos]
iter3_oof_id  = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos]
fixed_oof_id  = (fw['mega33']*mega_oof_id + fw['rank_adj']*rank_oof_id +
                 fw['iter_r1']*iter1_oof_id + fw['iter_r2']*iter2_oof_id +
                 fw['iter_r3']*iter3_oof_id)

mega_test_id  = d['meta_avg_test'][test_id_to_lspos]
rank_test_id  = np.load('results/ranking/rank_adj_test.npy')[test_id_to_lspos]
iter1_test_id = np.load('results/iter_pseudo/round1_test.npy')[test_id_to_lspos]
iter2_test_id = np.load('results/iter_pseudo/round2_test.npy')[test_id_to_lspos]
iter3_test_id = np.load('results/iter_pseudo/round3_test.npy')[test_id_to_lspos]
fixed_test_id = (fw['mega33']*mega_test_id + fw['rank_adj']*rank_test_id +
                 fw['iter_r1']*iter1_test_id + fw['iter_r2']*iter2_test_id +
                 fw['iter_r3']*iter3_test_id)

print(f"FIXED proxy OOF MAE: {np.mean(np.abs(fixed_oof_id - y_true)):.4f}")
print(f"mega33 proxy MAE:    {np.mean(np.abs(mega_oof_id  - y_true)):.4f}")

train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)

# FIXED-based test lags
test_raw['fixed_pred'] = fixed_test_id
test_raw['lag1_fixed'] = test_raw.groupby(['layout_id','scenario_id'])['fixed_pred'].shift(1).fillna(global_mean)
test_raw['lag2_fixed'] = test_raw.groupby(['layout_id','scenario_id'])['fixed_pred'].shift(2).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].values
test_lag1_fix  = test_raw['lag1_fixed'].values
test_lag2_fix  = test_raw['lag2_fixed'].values
test_row_sc    = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

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

oof_xfp    = np.full(len(train_raw), np.nan)
test_xfp_list = []

print("Training oracle-XGB-fixedproxy...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_vtrue = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_true[val_idx])],
              verbose=False)

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    # Use FIXED as proxy instead of mega33
    fixed_val_sorted = fixed_oof_id[val_sorted]

    foldCxfp = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = fixed_val_sorted[row_in_sc_vals == (pos-1)]
            l2 = fixed_val_sorted[row_in_sc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, global_mean)
        X_pos = make_X(X_train_base[pos_idx], l1, l2, np.full(n_pos, pos))
        foldCxfp[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_xfp[val_sorted] = foldCxfp

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    fixed_test_sorted = fixed_test_id[test_sorted]

    testCxfp = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = fixed_test_sorted[test_rsc_vals == (pos-1)]
            l2 = fixed_test_sorted[test_rsc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, global_mean)
        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(n_pos, pos))
        testCxfp[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_xfp_list.append(testCxfp)

    maeCxfp = np.mean(np.abs(foldCxfp - y_true[val_sorted]))
    elapsed  = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-XGB-fixedproxy={maeCxfp:.4f}  ({elapsed:.0f}s)", flush=True)

test_xfp_avg = np.mean(test_xfp_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_fixedproxy.npy', oof_xfp)
np.save('results/oracle_seq/test_C_xgb_fixedproxy.npy', test_xfp_avg)
print("Saved oof_seqC_xgb_fixedproxy.npy and test_C_xgb_fixedproxy.npy", flush=True)

# Quick summary
xgb_orig = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2      = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed2   = fixed_oof_id
y2       = y_true
xfp_mae  = np.mean(np.abs(oof_xfp - y2))
xorig_mae = np.mean(np.abs(xgb_orig - y2))
fixed_mae = np.mean(np.abs(fixed2 - y2))
res_fixed = y2 - fixed2
corr_xfp  = np.corrcoef(res_fixed, oof_xfp - fixed2)[0,1]
corr_xorig = np.corrcoef(res_fixed, xgb_orig - fixed2)[0,1]
print(f"\noracle-XGB-fixedproxy MAE:  {xfp_mae:.4f}  residual_corr={corr_xfp:.4f}", flush=True)
print(f"oracle-XGB-mega33 MAE:     {xorig_mae:.4f}  residual_corr={corr_xorig:.4f}", flush=True)
print(f"pred corr (xfp vs xorig):  {np.corrcoef(oof_xfp, xgb_orig)[0,1]:.4f}", flush=True)
print(f"pred corr (xfp vs lv2):    {np.corrcoef(oof_xfp, lv2)[0,1]:.4f}", flush=True)
# Best static blend
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_xfp
    mm = np.mean(np.abs(bl - y2))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static blend: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
