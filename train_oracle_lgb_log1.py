"""
Oracle-LGB-log1: LGB with log(y+1) target + 1 log lag only (vs lv2 which has 3).
Hypothesis: fewer lags = less temporal context, more reliance on features = different error pattern.
pos 0: no lag available → global_mean (same as lv2)
pos 1+: only lag1 (vs lv2 which uses lag1,2,3)
→ Different from lv2 especially for positions 1-2 where lv2 has identical or near-identical lag history.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

NLAGS = 1

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
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# 1 log lag for training
train_raw['log_y'] = y_log
grp = train_raw.groupby(['layout_id','scenario_id'])['log_y']
train_raw['lag1_logy'] = grp.shift(1).fillna(global_mean_log)

# 1 log mega lag for test
test_raw['log_mega33'] = np.log1p(mega_test_id)
m_grp = test_raw.groupby(['layout_id','scenario_id'])['log_mega33']
test_raw['lag1_logmega'] = m_grp.shift(1).fillna(global_mean_log)

X_train_base  = fe_train_df[feat_cols].fillna(0).values
train_lag1    = train_raw['lag1_logy'].values
row_sc_arr    = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns: X_test_base[c] = 0.0
X_test_base   = X_test_base[feat_cols].fillna(0).values
test_lag1_mega = test_raw['lag1_logmega'].values
test_row_sc   = test_raw['row_in_sc'].values

def make_X(base, lag1, row_sc):
    return np.hstack([base, np.column_stack([lag1, row_sc])])

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

oof_l1   = np.full(len(train_raw), np.nan)
test_l1_list = []

print(f"Training oracle-LGB-log1 (1 log lag)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_log[tr_idx]
    X_vl = make_X(X_train_base[val_idx], train_lag1[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vl, y_log[val_idx])],
              callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(-1)])

    val_df = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig'] = val_idx
    val_df = val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted = val_df['_orig'].values
    rsc_vals   = val_df['row_in_sc'].values
    log_mega_val = np.log1p(mega_oof_id[val_sorted])

    fold_pred = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = rsc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lag1_pos = np.full(n_pos, global_mean_log) if pos == 0 else log_mega_val[rsc_vals == (pos-1)]
        X_pos = make_X(X_train_base[pos_idx], lag1_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        fold_pred[pos_mask] = np.maximum(0, np.expm1(pred_log))
    oof_l1[val_sorted] = fold_pred

    test_df = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df['_orig'] = np.arange(len(test_raw))
    test_df = test_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted = test_df['_orig'].values
    te_rsc    = test_df['row_in_sc'].values
    log_mega_te = np.log1p(mega_test_id[te_sorted])

    test_pred = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = te_rsc == pos
        pos_idx  = te_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lag1_pos = np.full(n_pos, global_mean_log) if pos == 0 else log_mega_te[te_rsc == (pos-1)]
        X_pos = make_X(X_test_base[pos_idx], lag1_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        test_pred[pos_idx] = np.maximum(0, np.expm1(pred_log))
    test_l1_list.append(test_pred)

    mae_fold = np.mean(np.abs(fold_pred - y_true[val_sorted]))
    elapsed  = time.time() - t0
    best_it  = model.best_iteration_ if hasattr(model, 'best_iteration_') else -1
    print(f"Fold {fold_i+1}: oracle-LGB-log1={mae_fold:.4f}  best_iter={best_it}  ({elapsed:.0f}s)", flush=True)
    if fold_i == 0 and mae_fold > 8.6:
        print(f"WARN: fold1={mae_fold:.4f} > 8.6, check result", flush=True)

test_l1_avg = np.mean(test_l1_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_log1.npy', oof_l1)
np.save('results/oracle_seq/test_C_lgb_log1.npy', test_l1_avg)
print("Saved oof_seqC_lgb_log1.npy, test_C_lgb_log1.npy", flush=True)

# Quick eval
oof_mae = np.mean(np.abs(oof_l1 - y_true))
print(f"\noracle-LGB-log1 OOF MAE: {oof_mae:.4f}", flush=True)

with open('results/mega33_final.pkl','rb') as f: d2 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2 = [ls2[i] for i in train_raw['ID'].values]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538, iter_r3=0.031038826035934514)
mega2=d2['meta_avg_oof'][id2]; rank2=np.load('results/ranking/rank_adj_oof.npy')[id2]
it1=np.load('results/iter_pseudo/round1_oof.npy')[id2]; it2=np.load('results/iter_pseudo/round2_oof.npy')[id2]; it3=np.load('results/iter_pseudo/round3_oof.npy')[id2]
fixed2 = fw['mega33']*mega2+fw['rank_adj']*rank2+fw['iter_r1']*it1+fw['iter_r2']*it2+fw['iter_r3']*it3
fixed_mae = np.mean(np.abs(fixed2 - y_true))
xgb_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
res = y_true - fixed2
print(f"FIXED MAE: {fixed_mae:.4f}", flush=True)
print(f"xgb_corr: {np.corrcoef(xgb_oof, oof_l1)[0,1]:.4f}  lv2_corr: {np.corrcoef(lv2_oof, oof_l1)[0,1]:.4f}", flush=True)
print(f"residual_corr_fixed: {np.corrcoef(res, oof_l1-fixed2)[0,1]:.4f}", flush=True)

best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_l1
    mm = np.mean(np.abs(bl - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static 1-way: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)

# Best 3-way including xgb and lv2
best3_m = np.mean(np.abs((1-0.12-0.16)*fixed2+0.12*xgb_oof+0.16*lv2_oof - y_true))
for w in np.arange(0.02, 0.21, 0.02):
    if 0.12+0.16+w > 0.60: break
    bl = (1-0.12-0.16-w)*fixed2+0.12*xgb_oof+0.16*lv2_oof+w*oof_l1
    mm = np.mean(np.abs(bl - y_true))
    if mm < best3_m: best3_m = mm; print(f"  xgb=0.12, lv2=0.16, log1={w:.2f}: {mm:.4f}", flush=True)
print(f"Done.", flush=True)
