"""
2nd-gen oracle: use 5-way blend OOF as lag proxy instead of mega33.
Better lag quality → less exposure bias → potentially different error structure.
Formula: blend5 = 0.68*FIXED + 0.12*oracle-XGB + 0.20*oracle-log_v2
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
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

mega_oof_id   = d['meta_avg_oof'][id_to_lspos]
mega_test_id  = d['meta_avg_test'][test_id_to_lspos]

# Build FIXED + blend5 for proxy
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
train_id2 = pd.read_csv('train.csv').copy()
train_id2['_row_id'] = train_id2['ID'].str.replace('TRAIN_','').astype(int)
train_id2 = train_id2.sort_values('_row_id').reset_index(drop=True)
test_id2 = pd.read_csv('test.csv').copy()
test_id2['_row_id'] = test_id2['ID'].str.replace('TEST_','').astype(int)
test_id2 = test_id2.sort_values('_row_id').reset_index(drop=True)

ls_pos2 = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos2 = [ls_pos2[rid] for rid in train_id2['ID'].values]
te_ls_pos2 = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls2 = [te_ls_pos2[rid] for rid in test_id2['ID'].values]

mega_oof_a  = d['meta_avg_oof'][id_to_lspos2]
rank_oof_a  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos2]
iter1_oof_a = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos2]
iter2_oof_a = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos2]
iter3_oof_a = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos2]
xgb_oof_a   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof_a   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
y_true_a    = train_id2['avg_delay_minutes_next_30m'].values

fixed_oof = (fw['mega33']*mega_oof_a + fw['rank_adj']*rank_oof_a +
             fw['iter_r1']*iter1_oof_a + fw['iter_r2']*iter2_oof_a + fw['iter_r3']*iter3_oof_a)
blend5_oof = 0.68*fixed_oof + 0.12*xgb_oof_a + 0.20*lv2_oof_a  # ID order
fixed_mae  = np.mean(np.abs(fixed_oof - y_true_a))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)
print(f"5-way blend OOF MAE: {np.mean(np.abs(blend5_oof - y_true_a)):.4f}", flush=True)

# Build test blend5 proxy
mega_test_a  = d['meta_avg_test'][te_id_to_ls2]
rank_test_a  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls2]
iter1_test_a = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls2]
iter2_test_a = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls2]
iter3_test_a = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls2]
xgb_test_a   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test_a   = np.load('results/oracle_seq/test_C_log_v2.npy')
fixed_test_a = (fw['mega33']*mega_test_a + fw['rank_adj']*rank_test_a +
               fw['iter_r1']*iter1_test_a + fw['iter_r2']*iter2_test_a + fw['iter_r3']*iter3_test_a)
blend5_test  = 0.68*fixed_test_a + 0.12*xgb_test_a + 0.20*lv2_test_a  # ID order (same as train_id2)

# Align blend5 to train_raw and test_raw order
# train_raw is sorted by _row_id = same as train_id2 → blend5_oof is already in this order
blend5_oof_proxy = blend5_oof   # aligned with train_raw
blend5_test_proxy = blend5_test  # aligned with test_raw (test_id2 order = test_raw order since both sorted by _row_id)

# Assign proxy as test column for lag shift
test_raw['blend5_pred'] = blend5_test_proxy
test_raw['lag1_blend5'] = test_raw.groupby(['layout_id','scenario_id'])['blend5_pred'].shift(1).fillna(global_mean)
test_raw['lag2_blend5'] = test_raw.groupby(['layout_id','scenario_id'])['blend5_pred'].shift(2).fillna(global_mean)
test_raw['lag3_blend5'] = test_raw.groupby(['layout_id','scenario_id'])['blend5_pred'].shift(3).fillna(global_mean)

# Training: use TRUE y (oracle)
train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)
train_raw['lag3_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(3).fillna(global_mean)

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
test_lag1_b5 = test_raw['lag1_blend5'].values
test_lag2_b5 = test_raw['lag2_blend5'].values
test_lag3_b5 = test_raw['lag3_blend5'].values
test_row_sc  = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, lag3, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, lag3, row_sc])])

LGB_PARAMS = {
    'objective': 'regression_l1',
    'num_leaves': 256,
    'learning_rate': 0.04,
    'n_estimators': 4000,
    'min_child_samples': 20,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': 4,
    'verbose': -1,
}
CALLBACKS = [lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)]

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_G2   = np.full(len(train_raw), np.nan)
test_G2_list = []
train_raw['blend5_proxy'] = blend5_oof_proxy  # for val lag extraction

print("Training oracle-Gen2 (blend5 proxy)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], train_lag3[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], train_lag3[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val_true, y_true[val_idx])], callbacks=CALLBACKS)

    # Sequential OOF eval using blend5 as proxy (better quality than mega33)
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    blend5_val_sorted = blend5_oof_proxy[val_sorted]

    foldG2 = np.zeros(len(val_sorted))
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
            l1 = blend5_val_sorted[row_in_sc_vals == 0]
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 2:
            l1 = blend5_val_sorted[row_in_sc_vals == 1]
            l2 = blend5_val_sorted[row_in_sc_vals == 0]
            l3 = np.full(n_pos, global_mean)
        else:
            l1 = blend5_val_sorted[row_in_sc_vals == (pos-1)]
            l2 = blend5_val_sorted[row_in_sc_vals == (pos-2)]
            l3 = blend5_val_sorted[row_in_sc_vals == (pos-3)]
        X_pos = make_X(X_train_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        foldG2[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_G2[val_sorted] = foldG2

    # Sequential test eval using blend5_test as proxy
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    blend5_test_sorted = blend5_test_proxy[test_sorted]

    testG2 = np.zeros(len(test_raw))
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
            l1 = blend5_test_sorted[test_rsc_vals == 0]
            l2 = np.full(n_pos, global_mean)
            l3 = np.full(n_pos, global_mean)
        elif pos == 2:
            l1 = blend5_test_sorted[test_rsc_vals == 1]
            l2 = blend5_test_sorted[test_rsc_vals == 0]
            l3 = np.full(n_pos, global_mean)
        else:
            l1 = blend5_test_sorted[test_rsc_vals == (pos-1)]
            l2 = blend5_test_sorted[test_rsc_vals == (pos-2)]
            l3 = blend5_test_sorted[test_rsc_vals == (pos-3)]
        X_pos = make_X(X_test_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        testG2[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_G2_list.append(testG2)

    mae_g2 = np.mean(np.abs(foldG2 - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-Gen2={mae_g2:.4f}  best_iter={model.best_iteration_}  ({elapsed:.0f}s)", flush=True)

test_G2_avg = np.mean(test_G2_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_gen2.npy', oof_G2)
np.save('results/oracle_seq/test_C_gen2.npy', test_G2_avg)

# Blend analysis
print("\n=== BLEND ANALYSIS ===", flush=True)
g2_mae   = np.mean(np.abs(oof_G2 - y_true_a))
corr_gF  = np.corrcoef(oof_G2 - y_true_a, fixed_oof - y_true_a)[0,1]
corr_gXG = np.corrcoef(oof_G2 - y_true_a, xgb_oof_a - y_true_a)[0,1]
corr_gLV = np.corrcoef(oof_G2 - y_true_a, lv2_oof_a - y_true_a)[0,1]
print(f"oracle-Gen2: MAE={g2_mae:.4f}  corr_FIXED={corr_gF:.4f}  corr_XGB={corr_gXG:.4f}  corr_Lv2={corr_gLV:.4f}", flush=True)

# 2-way FIXED + Gen2
best_2, best_w2 = 9999, 0
for w in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs((1-w)*fixed_oof + w*oof_G2 - y_true_a))
    if m < best_2: best_2, best_w2 = m, w
print(f"2-way FIXED+Gen2: wG2={best_w2:.2f} MAE={best_2:.4f} delta={best_2-fixed_mae:.4f}", flush=True)

# 4-way: FIXED + XGB + Lv2 + Gen2
print("\n4-way FIXED+XGB+Lv2+Gen2 grid...", flush=True)
best_4, best_wXG, best_wLV, best_wG2 = 9999, 0, 0, 0
for wXG in np.arange(0, 0.41, 0.04):
    for wLV in np.arange(0, 0.41, 0.04):
        for wG2 in np.arange(0, 0.41, 0.04):
            if wXG + wLV + wG2 > 0.60: continue
            blend = (1-wXG-wLV-wG2)*fixed_oof + wXG*xgb_oof_a + wLV*lv2_oof_a + wG2*oof_G2
            m = np.mean(np.abs(blend - y_true_a))
            if m < best_4: best_4, best_wXG, best_wLV, best_wG2 = m, wXG, wLV, wG2
print(f"4-way: wXG={best_wXG:.2f} wLV={best_wLV:.2f} wG2={best_wG2:.2f} MAE={best_4:.4f} delta={best_4-fixed_mae:.4f}", flush=True)

# Fold analysis
gkf2 = GroupKFold(n_splits=5)
groups_id2b = train_id2['layout_id'].values
folds_4 = []
for _, val_idx in gkf2.split(np.arange(len(train_id2)), groups=groups_id2b):
    bv = ((1-best_wXG-best_wLV-best_wG2)*fixed_oof[val_idx]
          + best_wXG*xgb_oof_a[val_idx] + best_wLV*lv2_oof_a[val_idx] + best_wG2*oof_G2[val_idx])
    folds_4.append(np.mean(np.abs(bv-y_true_a[val_idx])) - np.mean(np.abs(fixed_oof[val_idx]-y_true_a[val_idx])))
print(f"Fold deltas: {[f'{x:.4f}' for x in folds_4]} ({sum(x<0 for x in folds_4)}/5 neg)", flush=True)

CURRENT_BEST = 8.3831
if best_4 < CURRENT_BEST - 0.0003 and sum(x < 0 for x in folds_4) >= 4:
    tb = np.maximum(0, (1-best_wXG-best_wLV-best_wG2)*fixed_test_a
                    + best_wXG*xgb_test_a + best_wLV*lv2_test_a + best_wG2*test_G2_avg)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id2['ID'].values, 'avg_delay_minutes_next_30m': tb})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_XGB_Lv2_Gen2_OOF{best_4:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo new best beyond {CURRENT_BEST:.4f}. 4-way={best_4:.4f}", flush=True)

print("Done.", flush=True)
