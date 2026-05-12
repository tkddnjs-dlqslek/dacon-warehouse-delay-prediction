"""
XGBoost Oracle + Monotone Constraints
Physical causality: order_inflow/congestion/fault → delay↑, success_rate/path_opt → delay↓
Hypothesis: constraints prevent seen-layout spurious correlations → better unseen generalization
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqC_xgb_monotone.npy'
OUT_TEST = 'results/oracle_seq/test_C_xgb_monotone.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); import sys; sys.exit(0)

t0 = time.time()
print('='*60)
print('XGBoost Oracle + Monotone Constraints')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = float(y_true.mean())
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# v30 features
print('\n[1] v30 feature cache 로드...')
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

X_train_base = fe_train_df[feat_cols].values.astype(np.float32)
X_test_base  = fe_test_df[[c if c in fe_test_df.columns else None for c in feat_cols]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].values.astype(np.float32)

print(f'  X_train_base: {X_train_base.shape}')

# mega33 proxy for oracle sequential features
print('\n[2] mega33 proxy 로드...')
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

# True lag features for training
train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)

# Mega proxy lag features for test
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(1).fillna(global_mean)
test_raw['lag2_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(2).fillna(global_mean)

train_lag1   = train_raw['lag1_y'].values.astype(np.float32)
train_lag2   = train_raw['lag2_y'].values.astype(np.float32)
row_sc_arr   = train_raw['row_in_sc'].values.astype(np.float32)
test_lag1    = test_raw['lag1_mega'].values.astype(np.float32)
test_lag2    = test_raw['lag2_mega'].values.astype(np.float32)
test_row_sc  = test_raw['row_in_sc'].values.astype(np.float32)

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])]).astype(np.float32)

# Monotone constraints
# v30 features (149) + lag1 (149) + lag2 (150) + row_sc (151) = 152 features total
# 1=increasing (feature↑ → delay↑), -1=decreasing (feature↑ → delay↓), 0=unconstrained
n_base = len(feat_cols)  # 149
n_total = n_base + 3     # 152

# Build constraint dict: {col_name: direction}
pos_cols = {
    'order_inflow_15m_sc_mean', 'order_inflow_15m_sc_max', 'order_inflow_15m_sc_min',
    'order_inflow_15m_sc_std', 'order_inflow_15m_sc_range',
    'order_inflow_15m_lag1', 'order_inflow_15m_lead1', 'order_inflow_15m_lead2',
    'fault_count_15m_sc_mean', 'fault_count_15m_sc_std',
    'congestion_score_sc_mean', 'congestion_score_sc_max', 'congestion_score_sc_std',
    'congestion_score_lead1', 'congestion_per_robot',
    'blocked_path_15m_sc_mean', 'blocked_path_15m_sc_max', 'blocked_path_15m_sc_std',
    'charge_queue_length_sc_mean', 'charge_queue_length_sc_max', 'charge_queue_length_sc_std',
    'intersection_wait_time_avg', 'aisle_traffic_score',
    'wms_response_time_ms', 'outbound_truck_wait_min',
}
neg_cols = {
    'agv_task_success_rate', 'barcode_read_success_rate',
    'path_optimization_score', 'sort_accuracy_pct',
    'battery_mean_sc_mean', 'battery_mean_sc_min',
}

mc = [0] * n_total
for i, c in enumerate(feat_cols):
    if c in pos_cols:
        mc[i] = 1
    elif c in neg_cols:
        mc[i] = -1
# oracle sequential features: lag1↑ → delay↑, lag2↑ → delay↑, row_sc unconstrained
mc[n_base]   = 1  # lag1
mc[n_base+1] = 1  # lag2

pos_count = sum(1 for x in mc if x == 1)
neg_count = sum(1 for x in mc if x == -1)
print(f'\n[3] Monotone constraints: +1={pos_count}, -1={neg_count}, 0={n_total-pos_count-neg_count}')

mc_tuple = tuple(mc)

XGB_PARAMS = dict(
    objective='reg:absoluteerror',
    n_estimators=3000, learning_rate=0.05,
    max_depth=7, min_child_weight=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbosity=0,
    early_stopping_rounds=100,
    eval_metric='mae',
    monotone_constraints=mc_tuple,
)

print('\n[4] 5-fold GroupKFold 학습 (sequential oracle)...')
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)
oof    = np.full(len(train_raw), np.nan)
test_preds = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()

    X_tr  = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr  = y_true[tr_idx]
    X_val = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_true[val_idx])], verbose=False)

    # Sequential oracle evaluation on val
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted      = val_df_tmp['_orig'].values
    rsc_vals        = val_df_tmp['row_in_sc'].values
    mega_val_sorted = mega_oof_id[val_sorted]

    fold_pred = np.zeros(len(val_sorted))
    for pos in range(25):
        pm   = rsc_vals == pos
        pidx = val_sorted[pm]
        np_  = pm.sum()
        if np_ == 0:
            continue
        if pos == 0:
            l1 = np.full(np_, global_mean, dtype=np.float32)
            l2 = np.full(np_, global_mean, dtype=np.float32)
        else:
            l1 = mega_val_sorted[rsc_vals == (pos-1)].astype(np.float32)
            l2 = mega_val_sorted[rsc_vals == (pos-2)].astype(np.float32) if pos >= 2 else np.full(np_, global_mean, dtype=np.float32)
        Xp = make_X(X_train_base[pidx], l1, l2, np.full(np_, pos, dtype=np.float32))
        fold_pred[pm] = np.maximum(0, model.predict(Xp))
    oof[val_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_sorted], fold_pred)

    # Sequential oracle test prediction
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    t_rsc_vals    = test_df_tmp['row_in_sc'].values
    mega_test_sorted = mega_test_id[test_sorted]

    test_fold = np.zeros(len(test_raw))
    for pos in range(25):
        pm   = t_rsc_vals == pos
        pidx = test_sorted[pm]
        np_  = pm.sum()
        if np_ == 0:
            continue
        if pos == 0:
            l1 = np.full(np_, global_mean, dtype=np.float32)
            l2 = np.full(np_, global_mean, dtype=np.float32)
        else:
            l1 = mega_test_sorted[t_rsc_vals == (pos-1)].astype(np.float32)
            l2 = mega_test_sorted[t_rsc_vals == (pos-2)].astype(np.float32) if pos >= 2 else np.full(np_, global_mean, dtype=np.float32)
        Xp = make_X(X_test_base[pidx], l1, l2, np.full(np_, pos, dtype=np.float32))
        test_fold[pidx] = np.maximum(0, model.predict(Xp))
    test_preds.append(test_fold)

    print(f'  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration}  ({time.time()-t1:.0f}s)', flush=True)
    del model

overall_mae = mean_absolute_error(y_true, oof)
test_avg = np.mean(test_preds, axis=0)
print(f'\nOverall OOF MAE: {overall_mae:.5f}  ({time.time()-t0:.0f}s total)')

np.save(OUT_OOF,  oof)
np.save(OUT_TEST, test_avg)
print(f'Saved: {OUT_OOF}, {OUT_TEST}')

# Compare with oracle_NEW
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2_2 = [ls2[i] for i in train_raw['ID'].values]
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d33['meta_avg_oof'][id2_2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2_2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2_2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2_2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2_2])
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_new_oof = 0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'\noracle_NEW OOF: {mae(oracle_new_oof):.5f}')
print(f'xgb_monotone  OOF: {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, xgb_monotone): {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')
print('Done.')
