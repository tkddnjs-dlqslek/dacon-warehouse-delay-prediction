"""
Oracle-LGB-fixedproxy: LGB + 3 log lags + FIXED (5-way blend) as proxy instead of mega33.
FIXED OOF MAE=8.3935 < mega33 OOF MAE~8.40 → smaller exposure bias at test time.
Both OOF eval AND test use log(FIXED+1) as lag proxy.
Hypothesis: more accurate proxy → predictions closer to test distribution → different/better OOF.
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
y_log  = np.log1p(y_true)
global_mean_log = y_log.mean()

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

# Build FIXED proxy
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

print(f"FIXED proxy OOF MAE: {np.mean(np.abs(fixed_oof_id - y_true)):.4f}", flush=True)
print(f"mega33 proxy MAE:    {np.mean(np.abs(mega_oof_id  - y_true)):.4f}", flush=True)

# True log-lags (3 lags) for training
train_raw['log_y'] = y_log
grp = train_raw.groupby(['layout_id','scenario_id'])['log_y']
train_raw['lag1_logy'] = grp.shift(1).fillna(global_mean_log)
train_raw['lag2_logy'] = grp.shift(2).fillna(global_mean_log)
train_raw['lag3_logy'] = grp.shift(3).fillna(global_mean_log)

# log(FIXED+1) lags for test
test_raw['log_fixed'] = np.log1p(fixed_test_id)
m_grp = test_raw.groupby(['layout_id','scenario_id'])['log_fixed']
test_raw['lag1_logfixed'] = m_grp.shift(1).fillna(global_mean_log)
test_raw['lag2_logfixed'] = m_grp.shift(2).fillna(global_mean_log)
test_raw['lag3_logfixed'] = m_grp.shift(3).fillna(global_mean_log)

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_logy'].values
train_lag2   = train_raw['lag2_logy'].values
train_lag3   = train_raw['lag3_logy'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base   = X_test_base[feat_cols].values
test_lag1_fix = test_raw['lag1_logfixed'].values
test_lag2_fix = test_raw['lag2_logfixed'].values
test_lag3_fix = test_raw['lag3_logfixed'].values
test_row_sc   = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, lag3, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, lag3, row_sc])])

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

oof_lfp   = np.full(len(train_raw), np.nan)
test_lfp_list = []

print("Training oracle-LGB-fixedproxy (3 log lags, FIXED proxy)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx],
                  train_lag3[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_log[tr_idx]
    X_vtrue = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx],
                     train_lag3[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_log[val_idx])],
              callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(-1)])

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    # Use log(FIXED+1) as proxy for OOF eval
    log_fixed_val  = np.log1p(fixed_oof_id[val_sorted])

    fold_lfp = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        gml = global_mean_log
        if pos == 0:
            l1 = np.full(n_pos, gml); l2 = np.full(n_pos, gml); l3 = np.full(n_pos, gml)
        else:
            l1 = log_fixed_val[row_in_sc_vals == (pos-1)]
            l2 = log_fixed_val[row_in_sc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, gml)
            l3 = log_fixed_val[row_in_sc_vals == (pos-3)] if pos >= 3 else np.full(n_pos, gml)
        X_pos = make_X(X_train_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        fold_lfp[pos_mask] = np.maximum(0, np.expm1(model.predict(X_pos)))
    oof_lfp[val_sorted] = fold_lfp

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    log_fixed_test = np.log1p(fixed_test_id[test_sorted])

    test_lfp = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        gml = global_mean_log
        if pos == 0:
            l1 = np.full(n_pos, gml); l2 = np.full(n_pos, gml); l3 = np.full(n_pos, gml)
        else:
            l1 = log_fixed_test[test_rsc_vals == (pos-1)]
            l2 = log_fixed_test[test_rsc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, gml)
            l3 = log_fixed_test[test_rsc_vals == (pos-3)] if pos >= 3 else np.full(n_pos, gml)
        X_pos = make_X(X_test_base[pos_idx], l1, l2, l3, np.full(n_pos, pos))
        test_lfp[pos_idx] = np.maximum(0, np.expm1(model.predict(X_pos)))
    test_lfp_list.append(test_lfp)

    mae_fold = np.mean(np.abs(fold_lfp - y_true[val_sorted]))
    elapsed  = time.time() - t0
    best_it  = model.best_iteration_ if hasattr(model, 'best_iteration_') else -1
    print(f"Fold {fold_i+1}: oracle-LGB-fixedproxy={mae_fold:.4f}  best_iter={best_it}  ({elapsed:.0f}s)", flush=True)
    if fold_i == 0 and mae_fold > 8.6:
        print(f"WARN: fold1={mae_fold:.4f} > 8.6", flush=True)

test_lfp_avg = np.mean(test_lfp_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_fixedproxy.npy', oof_lfp)
np.save('results/oracle_seq/test_C_lgb_fixedproxy.npy', test_lfp_avg)
print("Saved oof_seqC_lgb_fixedproxy.npy, test_C_lgb_fixedproxy.npy", flush=True)

# Quick eval
oof_mae = np.mean(np.abs(oof_lfp - y_true))
fixed_mae = np.mean(np.abs(fixed_oof_id - y_true))
lv2_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
xgb_oof  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
res = y_true - fixed_oof_id
print(f"\noracle-LGB-fixedproxy OOF MAE: {oof_mae:.4f}", flush=True)
print(f"FIXED MAE: {fixed_mae:.4f}", flush=True)
print(f"lv2_corr:  {np.corrcoef(lv2_oof, oof_lfp)[0,1]:.4f}", flush=True)
print(f"xgb_corr:  {np.corrcoef(xgb_oof, oof_lfp)[0,1]:.4f}", flush=True)
print(f"residual_corr: {np.corrcoef(res, oof_lfp - fixed_oof_id)[0,1]:.4f}", flush=True)

best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed_oof_id + w*oof_lfp
    mm = np.mean(np.abs(bl - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static 1-way: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)

# 3-way blend vs current best
for w in np.arange(0.02, 0.21, 0.02):
    if 0.12 + 0.16 + w > 0.60: break
    bl = (1-0.12-0.16-w)*fixed_oof_id + 0.12*xgb_oof + 0.16*lv2_oof + w*oof_lfp
    mm = np.mean(np.abs(bl - y_true))
    if mm < 8.3828:
        print(f"  xgb=0.12, lv2=0.16, lgbfp={w:.2f}: {mm:.4f}", flush=True)
print("Done.", flush=True)
