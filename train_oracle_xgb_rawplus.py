"""
Oracle-XGB-rawplus: v30 feat_cols + 17 raw current-state cols (unused in v30).
Key insight: v30 only has SC aggregates and lag/diff features of raw cols,
but NOT the raw current-timestep values themselves.
Special: task_reassign_15m, avg_recovery_time, replenishment_overlap,
         staff_on_floor, forklift_active_count, order_wave_count
         → completely absent from v30 FE (new signals).
Kill if fold1 > 8.85.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 8.85

RAW_EXTRA = [
    "order_inflow_15m", "robot_charging", "robot_utilization", "task_reassign_15m",
    "battery_mean", "low_battery_ratio", "charge_queue_length", "avg_charge_wait",
    "congestion_score", "blocked_path_15m", "near_collision_15m", "fault_count_15m",
    "avg_recovery_time", "replenishment_overlap", "staff_on_floor",
    "forklift_active_count", "order_wave_count",
]

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

base_feat_cols = fe_tr['feat_cols']
fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

# Add raw extra cols
raw_avail_tr = [c for c in RAW_EXTRA if c in fe_train_df.columns]
raw_avail_te = [c for c in raw_avail_tr if c in fe_test_df.columns]
raw_avail = [c for c in raw_avail_tr if c in raw_avail_te]
feat_cols = list(base_feat_cols) + raw_avail
print(f"v30 base: {len(base_feat_cols)}  + raw: {len(raw_avail)}  = {len(feat_cols)} total", flush=True)
print(f"New raw cols: {raw_avail}", flush=True)

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

train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(1).fillna(global_mean)
test_raw['lag2_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(2).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].fillna(0).values
row_sc_arr   = train_raw['row_in_sc'].values
X_test_base  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns: X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].fillna(0).values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

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

print("Training oracle-XGB-rawplus...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_raw['lag1_y'].values[tr_idx],
                  train_raw['lag2_y'].values[tr_idx], row_sc_arr[tr_idx])
    X_val = make_X(X_train_base[val_idx], train_raw['lag1_y'].values[val_idx],
                   train_raw['lag2_y'].values[val_idx], row_sc_arr[val_idx])
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_true[tr_idx], eval_set=[(X_val, y_true[val_idx])], verbose=False)

    val_df = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig'] = val_idx
    val_df = val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df['_orig'].values
    rsc_vals   = val_df['row_in_sc'].values
    mega_val   = mega_oof_id[val_sorted]

    fold_pred = np.zeros(len(val_sorted))
    for pos in range(25):
        pm = rsc_vals == pos; pi = val_sorted[pm]; n = pm.sum()
        l1 = np.full(n, global_mean) if pos == 0 else mega_val[rsc_vals == (pos-1)]
        l2 = np.full(n, global_mean) if pos < 2  else mega_val[rsc_vals == (pos-2)]
        fold_pred[pm] = np.maximum(0, model.predict(make_X(X_train_base[pi], l1, l2, np.full(n, pos))))
    oof[val_sorted] = fold_pred

    test_df = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df['_orig'] = np.arange(len(test_raw))
    test_df = test_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted = test_df['_orig'].values; te_rsc = test_df['row_in_sc'].values
    mega_te   = mega_test_id[te_sorted]
    test_pred = np.zeros(len(test_raw))
    for pos in range(25):
        pm = te_rsc == pos; pi = te_sorted[pm]; n = pm.sum()
        l1 = np.full(n, global_mean) if pos == 0 else mega_te[te_rsc == (pos-1)]
        l2 = np.full(n, global_mean) if pos < 2  else mega_te[te_rsc == (pos-2)]
        test_pred[pi] = np.maximum(0, model.predict(make_X(X_test_base[pi], l1, l2, np.full(n, pos))))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_sorted]))
    print(f"Fold {fold_i+1}: oracle-XGB-rawplus={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: no improvement. Kill. ***", flush=True)
        import sys; sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb_rawplus.npy', oof)
np.save('results/oracle_seq/test_C_xgb_rawplus.npy', test_avg)
print("Saved oof_seqC_xgb_rawplus.npy", flush=True)

# Blend eval
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2 = [ls2[i] for i in train_raw['ID'].values]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed2 = (fw['mega33']*d['meta_avg_oof'][id2]
        + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
        + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
        + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
        + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae = np.mean(np.abs(fixed2 - y_true))
oof_mae   = np.mean(np.abs(oof - y_true))
print(f"\noracle-XGB-rawplus OOF MAE: {oof_mae:.4f}", flush=True)
print(f"oracle-XGB-v30     OOF MAE: {np.mean(np.abs(xgb_v30-y_true)):.4f}", flush=True)
print(f"FIXED MAE: {fixed_mae:.4f}", flush=True)
print(f"rawplus corr w/ xgb_v30: {np.corrcoef(xgb_v30, oof)[0,1]:.4f}", flush=True)
print(f"rawplus corr w/ lv2:     {np.corrcoef(lv2_oof, oof)[0,1]:.4f}", flush=True)

best_m = fixed_mae; best_w = 0
for w in np.arange(0.02, 0.51, 0.02):
    mm = np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm < best_m: best_m=mm; best_w=w
print(f"FIXED+rawplus: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)

base5 = (1-0.12-0.20)*fixed2 + 0.12*xgb_v30 + 0.20*lv2_oof
best_m4 = np.mean(np.abs(base5-y_true)); best_w4 = 0
for w in np.arange(0.02, 0.21, 0.02):
    mm = np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm < best_m4: best_m4=mm; best_w4=w
print(f"base5+rawplus: w={best_w4:.2f}  MAE={best_m4:.4f}  delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}", flush=True)
print("Done.", flush=True)
