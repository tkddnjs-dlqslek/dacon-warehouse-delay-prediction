"""
Oracle-CatBoost: CatBoost with symmetric/oblivious trees + 3 oracle lags
Key difference: ordered boosting + symmetric trees vs LGB leaf-wise vs XGB level-wise
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

# Precompute training lags (TRUE y, oracle)
train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)
train_raw['lag3_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(3).fillna(global_mean)

# Test lags (mega33 proxy)
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(1).fillna(global_mean)
test_raw['lag2_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(2).fillna(global_mean)
test_raw['lag3_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(3).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
train_lag3   = train_raw['lag3_y'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].values
test_lag1_mega = test_raw['lag1_mega'].values
test_lag2_mega = test_raw['lag2_mega'].values
test_lag3_mega = test_raw['lag3_mega'].values
test_row_sc    = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, lag3, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, lag3, row_sc])])

CB_PARAMS = dict(
    loss_function='MAE',
    iterations=3000,
    learning_rate=0.04,
    depth=8,           # oblivious trees: 2^8=256 leaf-equivalent
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

oof_Ccb   = np.full(len(train_raw), np.nan)
test_Ccb_list = []

print("Training oracle-CatBoost...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr  = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], train_lag3[tr_idx], row_sc_arr[tr_idx])
    y_tr  = y_true[tr_idx]
    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], train_lag3[val_idx], row_sc_arr[val_idx])

    tr_pool  = Pool(X_tr, label=y_tr)
    val_pool = Pool(X_val_true, label=y_true[val_idx])
    model = CatBoostRegressor(**CB_PARAMS)
    model.fit(tr_pool, eval_set=val_pool)

    # Sequential OOF eval using mega33 as proxy
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted = mega_oof_id[val_sorted]

    foldCcb = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if n_pos == 0: continue
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 1:
            l1 = mega_val_sorted[row_in_sc_vals == 0]
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 2:
            l1 = mega_val_sorted[row_in_sc_vals == 1]
            l2 = mega_val_sorted[row_in_sc_vals == 0]
            l3 = np.full(n_pos, global_mean)
        else:
            l1 = mega_val_sorted[row_in_sc_vals == (pos-1)]
            l2 = mega_val_sorted[row_in_sc_vals == (pos-2)]
            l3 = mega_val_sorted[row_in_sc_vals == (pos-3)]
        X_pos = make_X(X_train_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        foldCcb[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_Ccb[val_sorted] = foldCcb

    # Sequential test eval
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted = mega_test_id[test_sorted]

    testCcb = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if n_pos == 0: continue
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 1:
            l1 = mega_test_sorted[test_rsc_vals == 0]
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 2:
            l1 = mega_test_sorted[test_rsc_vals == 1]
            l2 = mega_test_sorted[test_rsc_vals == 0]
            l3 = np.full(n_pos, global_mean)
        else:
            l1 = mega_test_sorted[test_rsc_vals == (pos-1)]
            l2 = mega_test_sorted[test_rsc_vals == (pos-2)]
            l3 = mega_test_sorted[test_rsc_vals == (pos-3)]
        X_pos = make_X(X_test_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        testCcb[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_Ccb_list.append(testCcb)

    mae_cb = np.mean(np.abs(foldCcb - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-CB={mae_cb:.4f}  best_iter={model.best_iteration_}  ({elapsed:.0f}s)", flush=True)

test_Ccb_avg = np.mean(test_Ccb_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_cb.npy', oof_Ccb)
np.save('results/oracle_seq/test_C_cb.npy', test_Ccb_avg)

# Blend analysis
print("\n=== BLEND ANALYSIS ===", flush=True)
train_id = pd.read_csv('train.csv').copy()
train_id['_row_id'] = train_id['ID'].str.replace('TRAIN_','').astype(int)
train_id = train_id.sort_values('_row_id').reset_index(drop=True)
test_id = pd.read_csv('test.csv').copy()
test_id['_row_id'] = test_id['ID'].str.replace('TEST_','').astype(int)
test_id = test_id.sort_values('_row_id').reset_index(drop=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls2.iterrows()}
id_to_lspos2 = [ls_pos[rid] for rid in train_id['ID'].values]
test_ls2 = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls2.iterrows()}
te_id_to_ls2 = [te_ls_pos[rid] for rid in test_id['ID'].values]

mega_oof_a  = d['meta_avg_oof'][id_to_lspos2]
rank_oof_a  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos2]
iter1_oof_a = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos2]
iter2_oof_a = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos2]
iter3_oof_a = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos2]
xgb_oof_a   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof_a   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
y_true_a = train_id['avg_delay_minutes_next_30m'].values

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof_a + fw['rank_adj']*rank_oof_a +
             fw['iter_r1']*iter1_oof_a + fw['iter_r2']*iter2_oof_a + fw['iter_r3']*iter3_oof_a)
fixed_mae = np.mean(np.abs(fixed_oof - y_true_a))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

cb_mae   = np.mean(np.abs(oof_Ccb - y_true_a))
corr_cF  = np.corrcoef(oof_Ccb - y_true_a, fixed_oof - y_true_a)[0,1]
corr_cXG = np.corrcoef(oof_Ccb - y_true_a, xgb_oof_a - y_true_a)[0,1]
corr_cLV = np.corrcoef(oof_Ccb - y_true_a, lv2_oof_a - y_true_a)[0,1]
print(f"oracle-CB: MAE={cb_mae:.4f}  corr_FIXED={corr_cF:.4f}  corr_XGB={corr_cXG:.4f}  corr_Lv2={corr_cLV:.4f}", flush=True)

# 2-way grid
best_2, best_w2 = 9999, 0
for wCB in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs((1-wCB)*fixed_oof + wCB*oof_Ccb - y_true_a))
    if m < best_2: best_2, best_w2 = m, wCB
print(f"2-way FIXED+CB: wCB={best_w2:.2f} MAE={best_2:.4f} delta={best_2-fixed_mae:.4f}", flush=True)

# 4-way: FIXED + XGB + Lv2 + CB
print("\n4-way FIXED+XGB+Lv2+CB grid...", flush=True)
best_4, best_wXG, best_wLV, best_wCB = 9999, 0, 0, 0
for wXG in np.arange(0, 0.41, 0.04):
    for wLV in np.arange(0, 0.41, 0.04):
        for wCB in np.arange(0, 0.41, 0.04):
            if wXG + wLV + wCB > 0.60: continue
            blend = (1-wXG-wLV-wCB)*fixed_oof + wXG*xgb_oof_a + wLV*lv2_oof_a + wCB*oof_Ccb
            m = np.mean(np.abs(blend - y_true_a))
            if m < best_4: best_4, best_wXG, best_wLV, best_wCB = m, wXG, wLV, wCB
print(f"4-way: wXG={best_wXG:.2f} wLV={best_wLV:.2f} wCB={best_wCB:.2f} MAE={best_4:.4f} delta={best_4-fixed_mae:.4f}", flush=True)

# Fold analysis
gkf2 = GroupKFold(n_splits=5)
groups_id2 = train_id['layout_id'].values
folds_4 = []
for _, val_idx in gkf2.split(np.arange(len(train_id)), groups=groups_id2):
    bv = ((1-best_wXG-best_wLV-best_wCB)*fixed_oof[val_idx]
          + best_wXG*xgb_oof_a[val_idx] + best_wLV*lv2_oof_a[val_idx] + best_wCB*oof_Ccb[val_idx])
    folds_4.append(np.mean(np.abs(bv-y_true_a[val_idx])) - np.mean(np.abs(fixed_oof[val_idx]-y_true_a[val_idx])))
print(f"Fold deltas: {[f'{x:.4f}' for x in folds_4]} ({sum(x<0 for x in folds_4)}/5 neg)", flush=True)

# Load test arrays for submission
mega_test_a = d['meta_avg_test'][te_id_to_ls2]
rank_test_a  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls2]
iter1_test_a = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls2]
iter2_test_a = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls2]
iter3_test_a = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls2]
xgb_test_a   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test_a   = np.load('results/oracle_seq/test_C_log_v2.npy')
fixed_test_a = (fw['mega33']*mega_test_a + fw['rank_adj']*rank_test_a +
               fw['iter_r1']*iter1_test_a + fw['iter_r2']*iter2_test_a + fw['iter_r3']*iter3_test_a)

CURRENT_BEST = 8.3831
if best_4 < CURRENT_BEST - 0.0003 and sum(x < 0 for x in folds_4) >= 4:
    tb = np.maximum(0, (1-best_wXG-best_wLV-best_wCB)*fixed_test_a
                    + best_wXG*xgb_test_a + best_wLV*lv2_test_a + best_wCB*test_Ccb_avg)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': tb})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_XGB_Lv2_CB_OOF{best_4:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
elif best_2 < CURRENT_BEST - 0.0003 and best_w2 > 0:
    tb = np.maximum(0, (1-best_w2)*fixed_test_a + best_w2*test_Ccb_avg)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': tb})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_FIXED_CB_OOF{best_2:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\nSaved 2-way: {fname}", flush=True)
else:
    print(f"\nNo new best beyond {CURRENT_BEST:.4f}. 4-way={best_4:.4f}", flush=True)

print("Done.", flush=True)
