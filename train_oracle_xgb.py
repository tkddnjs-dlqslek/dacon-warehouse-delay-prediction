"""
Oracle-C XGBoost: same oracle setup (true lags in training, mega33_oof at OOF eval),
but uses XGBoost instead of LightGBM. Different algorithm family → different error profile.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
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

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base    = X_test_base[feat_cols].values
test_lag1_mega = test_raw['lag1_mega'].values
test_lag2_mega = test_raw['lag2_mega'].values
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

oof_Cxgb   = np.full(len(train_raw), np.nan)
test_Cxgb_list = []

print("Training oracle-XGB...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx], train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx], train_lag2[val_idx], row_sc_arr[val_idx])

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_true, y_true[val_idx])],
              verbose=False)

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted = mega_oof_id[val_sorted]

    foldCxgb = np.zeros(len(val_sorted))
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
        foldCxgb[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_Cxgb[val_sorted] = foldCxgb

    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted = mega_test_id[test_sorted]

    testCxgb = np.zeros(len(test_raw))
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
        testCxgb[pos_idx] = np.maximum(0, model.predict(X_pos))
    test_Cxgb_list.append(testCxgb)

    maeXgb = np.mean(np.abs(foldCxgb - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: oracle-XGB={maeXgb:.4f}  ({elapsed:.0f}s)", flush=True)

test_Cxgb_avg = np.mean(test_Cxgb_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_xgb.npy', oof_Cxgb)
np.save('results/oracle_seq/test_C_xgb.npy', test_Cxgb_avg)

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
oracC2_oof  = np.load('results/oracle_seq/oof_seqC_v2.npy')
y_true_a = train_id['avg_delay_minutes_next_30m'].values

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof_a + fw['rank_adj']*rank_oof_a +
             fw['iter_r1']*iter1_oof_a + fw['iter_r2']*iter2_oof_a + fw['iter_r3']*iter3_oof_a)
fixed_mae = np.mean(np.abs(fixed_oof - y_true_a))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

xgb_mae  = np.mean(np.abs(oof_Cxgb - y_true_a))
corr_xF  = np.corrcoef(oof_Cxgb - y_true_a, fixed_oof - y_true_a)[0,1]
corr_xC2 = np.corrcoef(oof_Cxgb - y_true_a, oracC2_oof - y_true_a)[0,1]
print(f"oracle-XGB: MAE={xgb_mae:.4f}  corr_FIXED={corr_xF:.4f}  corr_C2={corr_xC2:.4f}", flush=True)

best_mx, best_wx = 9999, 0
for w in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs(w*oof_Cxgb + (1-w)*fixed_oof - y_true_a))
    if m < best_mx:
        best_mx, best_wx = m, w
print(f"FIXED + XGB: w={best_wx:.2f} MAE={best_mx:.4f}  delta={best_mx-fixed_mae:.4f}", flush=True)

# Triple: FIXED + C2 + XGB
best_t3, best_wC2, best_wXG = 9999, 0, 0
for wC2 in np.arange(0, 0.41, 0.04):
    for wXG in np.arange(0, 0.41, 0.04):
        if wC2 + wXG > 0.6:
            continue
        blend = (1-wC2-wXG)*fixed_oof + wC2*oracC2_oof + wXG*oof_Cxgb
        m = np.mean(np.abs(blend - y_true_a))
        if m < best_t3:
            best_t3, best_wC2, best_wXG = m, wC2, wXG
print(f"FIXED + C2 + XGB triple: wC2={best_wC2:.2f} wXG={best_wXG:.2f} MAE={best_t3:.4f}  delta={best_t3-fixed_mae:.4f}", flush=True)

from sklearn.model_selection import GroupKFold as GKF
gkf2 = GKF(n_splits=5)
groups2 = train_id['layout_id'].values
folds_delta = []
best_overall = min(best_mx, best_t3)
for _, val_idx in gkf2.split(np.arange(len(train_id)), groups=groups2):
    fixed_val = fixed_oof[val_idx]
    xgb_val = oof_Cxgb[val_idx]
    C2_val = oracC2_oof[val_idx]
    y_val = y_true_a[val_idx]
    if best_t3 < best_mx:
        blend_val = (1-best_wC2-best_wXG)*fixed_val + best_wC2*C2_val + best_wXG*xgb_val
    else:
        blend_val = best_wx*xgb_val + (1-best_wx)*fixed_val
    delta = np.mean(np.abs(blend_val - y_val)) - np.mean(np.abs(fixed_val - y_val))
    folds_delta.append(delta)
print(f"Fold deltas: {[f'{x:.4f}' for x in folds_delta]}", flush=True)

if best_overall < fixed_mae - 0.001 and sum(x < 0 for x in folds_delta) >= 4:
    mega_test_a  = d['meta_avg_test'][te_id_to_ls2]
    rank_test_a  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls2]
    iter1_test_a = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls2]
    iter2_test_a = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls2]
    iter3_test_a = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls2]
    oracC2_test  = np.load('results/oracle_seq/test_C_v2.npy')
    fixed_test = (fw['mega33']*mega_test_a + fw['rank_adj']*rank_test_a +
                  fw['iter_r1']*iter1_test_a + fw['iter_r2']*iter2_test_a +
                  fw['iter_r3']*iter3_test_a)
    if best_t3 < best_mx:
        test_blend = np.maximum(0, (1-best_wC2-best_wXG)*fixed_test + best_wC2*oracC2_test + best_wXG*test_Cxgb_avg)
        fname = f'submission_oracle_C2_XGB_triple_OOF{best_t3:.4f}.csv'
    else:
        test_blend = np.maximum(0, best_wx*test_Cxgb_avg + (1-best_wx)*fixed_test)
        fname = f'submission_oracle_XGB_w{best_wx:.2f}_OOF{best_mx:.4f}.csv'
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_id['ID'].values, 'avg_delay_minutes_next_30m': test_blend})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    sub_df.to_csv(fname, index=False)
    print(f"Saved: {fname}", flush=True)
else:
    print("No submission generated.", flush=True)

with open('results/oracle_seq/blend_summary_xgb.json', 'w') as f:
    json.dump({
        'fixed_mae': float(fixed_mae),
        'oracle_xgb_mae': float(xgb_mae),
        'corr_xgb_FIXED': float(corr_xF),
        'corr_xgb_C2': float(corr_xC2),
        'best_2way_w': float(best_wx),
        'best_2way_mae': float(best_mx),
        'best_triple_wC2': float(best_wC2),
        'best_triple_wXG': float(best_wXG),
        'best_triple_mae': float(best_t3),
        'fold_deltas': [float(x) for x in folds_delta],
    }, f, indent=2)
print("Done.", flush=True)
