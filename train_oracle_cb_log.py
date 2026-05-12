"""
Oracle-CatBoost-Log: CatBoost + log(y+1) target + log lag features.
Combines CatBoost's symmetric trees with log-space target transform.
Most structurally diverse from existing LGB oracles.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
from catboost import CatBoostRegressor, Pool
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
global_log_mean = np.log1p(global_mean)

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

# Log-space lags for training (oracle): log(y+1)
train_raw['log_y'] = np.log1p(train_raw['avg_delay_minutes_next_30m'])
train_raw['lag1_logy'] = train_raw.groupby(['layout_id','scenario_id'])['log_y'].shift(1).fillna(global_log_mean)
train_raw['lag2_logy'] = train_raw.groupby(['layout_id','scenario_id'])['log_y'].shift(2).fillna(global_log_mean)
train_raw['lag3_logy'] = train_raw.groupby(['layout_id','scenario_id'])['log_y'].shift(3).fillna(global_log_mean)

# Log-space lags for test (mega33 proxy)
test_raw['log_mega'] = np.log1p(mega_test_id)
test_raw['lag1_logm'] = test_raw.groupby(['layout_id','scenario_id'])['log_mega'].shift(1).fillna(global_log_mean)
test_raw['lag2_logm'] = test_raw.groupby(['layout_id','scenario_id'])['log_mega'].shift(2).fillna(global_log_mean)
test_raw['lag3_logm'] = test_raw.groupby(['layout_id','scenario_id'])['log_mega'].shift(3).fillna(global_log_mean)

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_logy'].values
train_lag2   = train_raw['lag2_logy'].values
train_lag3   = train_raw['lag3_logy'].values
y_log        = np.log1p(y_true)
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].values
test_lag1_lm = test_raw['lag1_logm'].values
test_lag2_lm = test_raw['lag2_logm'].values
test_lag3_lm = test_raw['lag3_logm'].values
test_row_sc  = test_raw['row_in_sc'].values
log_mega_test_id = np.log1p(mega_test_id)

def make_X(base, lag1, lag2, lag3, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, lag3, row_sc])])

CB_PARAMS = dict(
    loss_function='MAE',
    iterations=3000,
    learning_rate=0.04,
    depth=8,
    l2_leaf_reg=5.0,
    min_data_in_leaf=20,
    random_seed=42,
    thread_count=4,
    verbose=False,
    eval_metric='MAE',
    early_stopping_rounds=100,
    use_best_model=True,
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_CBlog   = np.full(len(train_raw), np.nan)
test_CBlog_list = []

print("Training oracle-CB-log...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    # Train on log(y+1) with log(y+1) lags
    X_tr  = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], train_lag3[tr_idx], row_sc_arr[tr_idx])
    y_tr  = y_log[tr_idx]
    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], train_lag3[val_idx], row_sc_arr[val_idx])

    tr_pool  = Pool(X_tr, label=y_tr)
    # Early stopping uses log-MAE but we evaluate in original space
    val_log_y = y_log[val_idx]
    val_pool = Pool(X_val_true, label=val_log_y)
    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(tr_pool, eval_set=val_pool)

    # Sequential OOF: use log(mega33) as proxy lag
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    log_mega_val   = np.log1p(mega_oof_id[val_sorted])

    foldCBlog = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if n_pos == 0: continue
        if pos == 0:
            l1 = np.full(n_pos, global_log_mean)
            l2 = np.full(n_pos, global_log_mean)
            l3 = np.full(n_pos, global_log_mean)
        elif pos == 1:
            l1 = log_mega_val[row_in_sc_vals == 0]
            l2 = np.full(n_pos, global_log_mean)
            l3 = np.full(n_pos, global_log_mean)
        elif pos == 2:
            l1 = log_mega_val[row_in_sc_vals == 1]
            l2 = log_mega_val[row_in_sc_vals == 0]
            l3 = np.full(n_pos, global_log_mean)
        else:
            l1 = log_mega_val[row_in_sc_vals == (pos-1)]
            l2 = log_mega_val[row_in_sc_vals == (pos-2)]
            l3 = log_mega_val[row_in_sc_vals == (pos-3)]
        X_pos = make_X(X_train_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        foldCBlog[pos_mask] = np.maximum(0, np.expm1(model.predict(X_pos)))
    oof_CBlog[val_sorted] = foldCBlog

    # Sequential test eval
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    log_mega_te   = np.log1p(mega_test_id[test_sorted])

    testCBlog = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if n_pos == 0: continue
        if pos == 0:
            l1 = np.full(n_pos, global_log_mean)
            l2 = np.full(n_pos, global_log_mean)
            l3 = np.full(n_pos, global_log_mean)
        elif pos == 1:
            l1 = log_mega_te[test_rsc_vals == 0]
            l2 = np.full(n_pos, global_log_mean)
            l3 = np.full(n_pos, global_log_mean)
        elif pos == 2:
            l1 = log_mega_te[test_rsc_vals == 1]
            l2 = log_mega_te[test_rsc_vals == 0]
            l3 = np.full(n_pos, global_log_mean)
        else:
            l1 = log_mega_te[test_rsc_vals == (pos-1)]
            l2 = log_mega_te[test_rsc_vals == (pos-2)]
            l3 = log_mega_te[test_rsc_vals == (pos-3)]
        X_pos = make_X(X_test_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        testCBlog[pos_idx] = np.maximum(0, np.expm1(model.predict(X_pos)))
    test_CBlog_list.append(testCBlog)

    mae_cbl = np.mean(np.abs(foldCBlog - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-CB-log={mae_cbl:.4f}  best_iter={model.best_iteration_}  ({elapsed:.0f}s)", flush=True)

test_CBlog_avg = np.mean(test_CBlog_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_cb_log.npy', oof_CBlog)
np.save('results/oracle_seq/test_C_cb_log.npy', test_CBlog_avg)

# Blend analysis
print("\n=== BLEND ANALYSIS ===", flush=True)
train_id = pd.read_csv('train.csv').copy()
train_id['_row_id'] = train_id['ID'].str.replace('TRAIN_','').astype(int)
train_id = train_id.sort_values('_row_id').reset_index(drop=True)
test_id = pd.read_csv('test.csv').copy()
test_id['_row_id'] = test_id['ID'].str.replace('TEST_','').astype(int)
test_id = test_id.sort_values('_row_id').reset_index(drop=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id_to_lspos2 = [ls_pos2[rid] for rid in train_id['ID'].values]
test_ls2 = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos2 = {row['ID']:i for i,row in test_ls2.iterrows()}
te_id_to_ls2 = [te_ls_pos2[rid] for rid in test_id['ID'].values]

mega_oof_b  = d['meta_avg_oof'][id_to_lspos2]
rank_oof_b  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos2]
iter1_oof_b = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos2]
iter2_oof_b = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos2]
iter3_oof_b = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos2]
xgb_oof_b   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof_b   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
y_true_b    = train_id['avg_delay_minutes_next_30m'].values

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof_b = (fw['mega33']*mega_oof_b + fw['rank_adj']*rank_oof_b +
               fw['iter_r1']*iter1_oof_b + fw['iter_r2']*iter2_oof_b + fw['iter_r3']*iter3_oof_b)
fixed_mae_b = np.mean(np.abs(fixed_oof_b - y_true_b))
print(f"FIXED OOF MAE: {fixed_mae_b:.4f}", flush=True)

cbl_mae   = np.mean(np.abs(oof_CBlog - y_true_b))
corr_cF   = np.corrcoef(oof_CBlog - y_true_b, fixed_oof_b - y_true_b)[0,1]
corr_cXG  = np.corrcoef(oof_CBlog - y_true_b, xgb_oof_b - y_true_b)[0,1]
corr_cLV  = np.corrcoef(oof_CBlog - y_true_b, lv2_oof_b - y_true_b)[0,1]
print(f"oracle-CB-log: MAE={cbl_mae:.4f}  corr_FIXED={corr_cF:.4f}  corr_XGB={corr_cXG:.4f}  corr_Lv2={corr_cLV:.4f}", flush=True)

# 2-way grid
best_2, best_w2 = 9999, 0
for w in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs((1-w)*fixed_oof_b + w*oof_CBlog - y_true_b))
    if m < best_2: best_2, best_w2 = m, w
print(f"2-way FIXED+CBlog: wCBL={best_w2:.2f} MAE={best_2:.4f} delta={best_2-fixed_mae_b:.4f}", flush=True)

# 4-way FIXED + XGB + Lv2 + CBlog
print("\n4-way FIXED+XGB+Lv2+CBlog grid...", flush=True)
best_4, best_wXG, best_wLV, best_wCBL = 9999, 0, 0, 0
for wXG in np.arange(0, 0.41, 0.04):
    for wLV in np.arange(0, 0.41, 0.04):
        for wCBL in np.arange(0, 0.41, 0.04):
            if wXG + wLV + wCBL > 0.60: continue
            blend = (1-wXG-wLV-wCBL)*fixed_oof_b + wXG*xgb_oof_b + wLV*lv2_oof_b + wCBL*oof_CBlog
            m = np.mean(np.abs(blend - y_true_b))
            if m < best_4: best_4, best_wXG, best_wLV, best_wCBL = m, wXG, wLV, wCBL
print(f"4-way: wXG={best_wXG:.2f} wLV={best_wLV:.2f} wCBL={best_wCBL:.2f} MAE={best_4:.4f} delta={best_4-fixed_mae_b:.4f}", flush=True)

gkf2 = GroupKFold(n_splits=5)
groups_id2b = train_id['layout_id'].values
folds_4 = []
for _, val_idx in gkf2.split(np.arange(len(train_id)), groups=groups_id2b):
    bv = ((1-best_wXG-best_wLV-best_wCBL)*fixed_oof_b[val_idx]
          + best_wXG*xgb_oof_b[val_idx] + best_wLV*lv2_oof_b[val_idx] + best_wCBL*oof_CBlog[val_idx])
    folds_4.append(np.mean(np.abs(bv-y_true_b[val_idx])) - np.mean(np.abs(fixed_oof_b[val_idx]-y_true_b[val_idx])))
print(f"Fold deltas: {[f'{x:.4f}' for x in folds_4]} ({sum(x<0 for x in folds_4)}/5 neg)", flush=True)

mega_test_b  = d['meta_avg_test'][te_id_to_ls2]
rank_test_b  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls2]
iter1_test_b = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls2]
iter2_test_b = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls2]
iter3_test_b = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls2]
xgb_test_b   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test_b   = np.load('results/oracle_seq/test_C_log_v2.npy')
fixed_test_b = (fw['mega33']*mega_test_b + fw['rank_adj']*rank_test_b +
               fw['iter_r1']*iter1_test_b + fw['iter_r2']*iter2_test_b + fw['iter_r3']*iter3_test_b)

CURRENT_BEST = 8.3831
if best_4 < CURRENT_BEST - 0.0003 and sum(x < 0 for x in folds_4) >= 4:
    tb = np.maximum(0, (1-best_wXG-best_wLV-best_wCBL)*fixed_test_b
                    + best_wXG*xgb_test_b + best_wLV*lv2_test_b + best_wCBL*test_CBlog_avg)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': tb})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_XGB_Lv2_CBlog_OOF{best_4:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo new best beyond {CURRENT_BEST:.4f}. 4-way={best_4:.4f}", flush=True)

print("Done.", flush=True)
