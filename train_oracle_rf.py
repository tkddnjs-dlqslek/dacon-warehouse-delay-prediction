"""
Oracle-C Random Forest: same oracle setup (true lags in training, mega33_oof at OOF eval).
RF with bagging+feature subsampling may tolerate exposure bias better than NN/CatBoost.
Uses absolute_error criterion to match MAE objective of LGB/XGB oracles.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor

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

train_raw['lag1_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(1).fillna(global_mean)
train_raw['lag2_y'] = train_raw.groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m'].shift(2).fillna(global_mean)

test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(1).fillna(global_mean)
test_raw['lag2_mega'] = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred'].shift(2).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].fillna(0).values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].fillna(0).values
test_lag1_mega = test_raw['lag1_mega'].values
test_lag2_mega = test_raw['lag2_mega'].values
test_row_sc    = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

RF_PARAMS = dict(
    n_estimators=500,
    criterion='squared_error',  # faster than absolute_error (no sorting), ~5x speedup
    max_depth=20,               # cap depth to prevent overfitting + speed
    min_samples_leaf=20,
    max_features='sqrt',
    n_jobs=6,
    random_state=42,
    verbose=0,
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_Crf   = np.full(len(train_raw), np.nan)
test_Crf_list = []

print("Training oracle-RF...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]

    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(X_tr, y_tr)

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted = mega_oof_id[val_sorted]

    foldCrf = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = mega_val_sorted[row_in_sc_vals == (pos-1)]
            l2 = mega_val_sorted[row_in_sc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, global_mean)
        X_pos = make_X(X_train_base[pos_idx], l1, l2, np.full(n_pos, pos))
        foldCrf[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_Crf[val_sorted] = foldCrf

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted = mega_test_id[test_sorted]

    testCrf = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = mega_test_sorted[test_rsc_vals == (pos-1)]
            l2 = mega_test_sorted[test_rsc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, global_mean)
        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(n_pos, pos))
        testCrf[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_Crf_list.append(testCrf)

    maeCrf = np.mean(np.abs(foldCrf - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-RF={maeCrf:.4f}  ({elapsed:.0f}s)", flush=True)

test_Crf_avg = np.mean(test_Crf_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_rf.npy', oof_Crf)
np.save('results/oracle_seq/test_C_rf.npy', test_Crf_avg)
print("Saved oof_seqC_rf.npy and test_C_rf.npy", flush=True)

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

mega_oof_a  = d['meta_avg_oof'][id_to_lspos2]
rank_oof_a  = np.load('results/ranking/rank_adj_oof.npy')[id_to_lspos2]
iter1_oof_a = np.load('results/iter_pseudo/round1_oof.npy')[id_to_lspos2]
iter2_oof_a = np.load('results/iter_pseudo/round2_oof.npy')[id_to_lspos2]
iter3_oof_a = np.load('results/iter_pseudo/round3_oof.npy')[id_to_lspos2]
xgb_oof_a   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof_a   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
y_true_a    = train_id['avg_delay_minutes_next_30m'].values

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof_a + fw['rank_adj']*rank_oof_a +
             fw['iter_r1']*iter1_oof_a + fw['iter_r2']*iter2_oof_a + fw['iter_r3']*iter3_oof_a)
fixed_mae  = np.mean(np.abs(fixed_oof - y_true_a))

rf_mae     = np.mean(np.abs(oof_Crf - y_true_a))
rf_residual = y_true_a - fixed_oof
rf_pred_delta = oof_Crf - fixed_oof
corr_rf = np.corrcoef(rf_residual, rf_pred_delta)[0,1]
print(f"FIXED MAE: {fixed_mae:.4f}", flush=True)
print(f"oracle-RF OOF MAE: {rf_mae:.4f}", flush=True)
print(f"Residual corr(oracle-RF): {corr_rf:.4f}", flush=True)

# Optimal 3-way blend (FIXED + XGB + Lv2 + RF)
best_mae = 9999; best_w = None
for wX in np.arange(0, 0.36, 0.04):
    for wL in np.arange(0, 0.36 - wX, 0.04):
        for wR in np.arange(0, 0.36 - wX - wL, 0.04):
            if wX + wL + wR > 0.35: continue
            blend = (1-wX-wL-wR)*fixed_oof + wX*xgb_oof_a + wL*lv2_oof_a + wR*oof_Crf
            m = np.mean(np.abs(blend - y_true_a))
            if m < best_mae:
                best_mae = m; best_w = (wX, wL, wR)

print(f"Best 4-way (FIXED+XGB+Lv2+RF): wXGB={best_w[0]:.2f} wLv2={best_w[1]:.2f} wRF={best_w[2]:.2f}  MAE={best_mae:.4f}  delta={best_mae-fixed_mae:.4f}", flush=True)

# Simple RF-only blend
for wR in np.arange(0, 0.41, 0.04):
    blend = (1-wR)*fixed_oof + wR*oof_Crf
    m = np.mean(np.abs(blend - y_true_a))
    if m < best_mae - 0.0001:
        best_mae = m; best_w = (0, 0, wR)
        print(f"  RF-only blend wR={wR:.2f} MAE={m:.4f}", flush=True)

# Cross-correlation with XGB and Lv2
corr_xgb = np.corrcoef(y_true_a - fixed_oof, xgb_oof_a - fixed_oof)[0,1]
corr_lv2 = np.corrcoef(y_true_a - fixed_oof, lv2_oof_a - fixed_oof)[0,1]
rf_xgb_corr = np.corrcoef(oof_Crf, xgb_oof_a)[0,1]
rf_lv2_corr = np.corrcoef(oof_Crf, lv2_oof_a)[0,1]
print(f"\nResidual corr vs FIXED: XGB={corr_xgb:.4f}  Lv2={corr_lv2:.4f}  RF={corr_rf:.4f}", flush=True)
print(f"Pred corr: RF-XGB={rf_xgb_corr:.4f}  RF-Lv2={rf_lv2_corr:.4f}", flush=True)

print("Done.", flush=True)
