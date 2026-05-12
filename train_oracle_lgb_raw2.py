"""
Oracle-LGB-raw2: LGB with 2 RAW y lags (no log transform).
Untested combination: LGB leaf-wise + raw lags.
LGB tolerates 3 LOG lags (lv2). Does it tolerate 2 RAW lags?
If yes: adds diversity vs oracle-XGB (level-wise + raw lags).
Kill if fold1 > 8.5.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

NLAGS = 2

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

# RAW y lags for training
grp = train_raw.groupby(['layout_id','scenario_id'])
for k in range(1, NLAGS+1):
    train_raw[f'lag{k}_y'] = grp['avg_delay_minutes_next_30m'].shift(k).fillna(global_mean)

# mega33 as proxy for test lags
test_raw['mega33_pred'] = mega_test_id
m_grp = test_raw.groupby(['layout_id','scenario_id'])['mega33_pred']
for k in range(1, NLAGS+1):
    test_raw[f'lag{k}_mega'] = m_grp.shift(k).fillna(global_mean)

X_train_base = fe_train_df[feat_cols].fillna(0).values
X_test_base  = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns: X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].fillna(0).values

train_lags = [train_raw[f'lag{k}_y'].values for k in range(1, NLAGS+1)]
test_lags  = [test_raw[f'lag{k}_mega'].values for k in range(1, NLAGS+1)]
row_sc_arr  = train_raw['row_in_sc'].values
test_row_sc = test_raw['row_in_sc'].values

def make_X(base, lags, row_sc):
    return np.hstack([base, np.column_stack(lags + [row_sc])])

LGB_PARAMS = dict(
    objective='mae', n_estimators=4000, learning_rate=0.04,
    num_leaves=256, max_depth=-1,
    min_child_samples=20, min_child_weight=0.001,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=6, random_state=42, verbose=-1,
)
EARLY = 100

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_lgbraw = np.full(len(train_raw), np.nan)
test_lgbraw_list = []

print(f"Training oracle-LGB-raw2 ({NLAGS} raw lags)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], [train_lags[k][tr_idx] for k in range(NLAGS)], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_vl = make_X(X_train_base[val_idx], [train_lags[k][val_idx] for k in range(NLAGS)], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_true[val_idx])],
              callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(-1)])

    # Sequential OOF eval with mega33 proxy
    val_df = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig'] = val_idx
    val_df = val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df['_orig'].values
    rsc_vals   = val_df['row_in_sc'].values
    mega_val   = mega_oof_id[val_sorted]

    fold_pred = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = rsc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean))
            else:
                lags_pos.append(mega_val[rsc_vals == (pos-k)])
        X_pos = make_X(X_train_base[pos_idx], lags_pos, np.full(n_pos, pos))
        fold_pred[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_lgbraw[val_sorted] = fold_pred

    # Test eval
    test_df = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df['_orig'] = np.arange(len(test_raw))
    test_df = test_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted = test_df['_orig'].values
    te_rsc    = test_df['row_in_sc'].values
    mega_te   = mega_test_id[te_sorted]

    test_pred = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = te_rsc == pos
        pos_idx  = te_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean))
            else:
                lags_pos.append(mega_te[te_rsc == (pos-k)])
        X_pos = make_X(X_test_base[pos_idx], lags_pos, np.full(n_pos, pos))
        test_pred[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_lgbraw_list.append(test_pred)

    mae_fold = np.mean(np.abs(fold_pred - y_true[val_sorted]))
    elapsed  = time.time() - t0
    best_it  = model.best_iteration_ if hasattr(model, 'best_iteration_') else -1
    print(f"Fold {fold_i+1}: oracle-LGB-raw2={mae_fold:.4f}  best_iter={best_it}  ({elapsed:.0f}s)", flush=True)
    if fold_i == 0 and mae_fold > 8.5:
        print(f"*** FAIL: fold1={mae_fold:.4f} > 8.5. LGB cannot handle 2 raw lags. Kill! ***", flush=True)
        sys.exit(1)

test_lgbraw_avg = np.mean(test_lgbraw_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_raw2.npy', oof_lgbraw)
np.save('results/oracle_seq/test_C_lgb_raw2.npy', test_lgbraw_avg)
print("Saved oof_seqC_lgb_raw2.npy, test_C_lgb_raw2.npy", flush=True)

# Quick eval
oof_mae = np.mean(np.abs(oof_lgbraw - y_true))
print(f"\noracle-LGB-raw2 OOF MAE: {oof_mae:.4f}", flush=True)

# Load fixed
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2 = [ls2[i] for i in train_raw['ID'].values]
mega2 = d['meta_avg_oof'][id2]
rank2 = np.load('results/ranking/rank_adj_oof.npy')[id2]
it1 = np.load('results/iter_pseudo/round1_oof.npy')[id2]
it2 = np.load('results/iter_pseudo/round2_oof.npy')[id2]
it3 = np.load('results/iter_pseudo/round3_oof.npy')[id2]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538, iter_r3=0.031038826035934514)
fixed2 = fw['mega33']*mega2+fw['rank_adj']*rank2+fw['iter_r1']*it1+fw['iter_r2']*it2+fw['iter_r3']*it3
fixed_mae = np.mean(np.abs(fixed2 - y_true))
res = y_true - fixed2
xgb_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
print(f"FIXED MAE: {fixed_mae:.4f}", flush=True)
print(f"xgb corr: {np.corrcoef(xgb_oof, oof_lgbraw)[0,1]:.4f}", flush=True)
print(f"lv2 corr: {np.corrcoef(lv2_oof, oof_lgbraw)[0,1]:.4f}", flush=True)
res_c = np.corrcoef(res, oof_lgbraw - fixed2)[0,1]
print(f"residual_corr with FIXED: {res_c:.4f}", flush=True)

# Best static blend
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_lgbraw
    mm = np.mean(np.abs(bl - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
