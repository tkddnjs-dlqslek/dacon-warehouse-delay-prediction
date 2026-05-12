"""
Oracle-LGB with cumulative scenario statistics (cummax, cumvar, cummean of y).
Late positions (17-24) where oracle helps most should benefit from richer scenario history.
Uses log target + 3 log lags + cumulative stats. All stats derived from true y during training.
OOF eval: mega33_oof-based cumulative stats as proxy.
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
global_mean     = y_true.mean()

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

NLAGS = 3

# Training: cumulative stats from TRUE y (oracle features)
train_raw['log_y'] = y_log
train_raw['raw_y'] = y_true
grp = train_raw.groupby(['layout_id','scenario_id'])

# Log lags (true)
for k in range(1, NLAGS+1):
    train_raw[f'lag{k}_logy'] = grp['log_y'].shift(k).fillna(global_mean_log)

# Cumulative stats of raw y (up to, not including, current position)
# Use reset_index + sort_index to recover original row order after groupby.expanding()
def safe_cumstats(df, col, fillval, keys):
    shifted = df.groupby(keys)[col].shift(1).fillna(fillval)
    g2 = shifted.groupby([df[k] for k in keys])
    mx   = g2.expanding().max().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    mn   = g2.expanding().mean().reset_index(level=list(range(len(keys))), drop=True).sort_index()
    std  = g2.expanding().std().fillna(0).reset_index(level=list(range(len(keys))), drop=True).sort_index()
    return mx.values, mn.values, std.values

KEYS = ['layout_id','scenario_id']
train_raw['cum_max_y'], train_raw['cum_mean_y'], train_raw['cum_std_y'] = safe_cumstats(train_raw, 'raw_y', global_mean, KEYS)
train_raw['cum_count'] = train_raw['row_in_sc'].clip(upper=24)

# Test: cumulative stats from mega33_oof proxy
test_raw['mega33_pred'] = mega_test_id
m_grp = test_raw.groupby(['layout_id','scenario_id'])

test_raw['log_mega33'] = np.log1p(mega_test_id)
for k in range(1, NLAGS+1):
    test_raw[f'lag{k}_logmega'] = m_grp['log_mega33'].shift(k).fillna(global_mean_log)

test_raw['cum_max_m'], test_raw['cum_mean_m'], test_raw['cum_std_m'] = safe_cumstats(test_raw, 'mega33_pred', global_mean, KEYS)

EXTRA_TRAIN = ['cum_max_y','cum_mean_y','cum_std_y','cum_count']
EXTRA_TEST  = ['cum_max_m','cum_mean_m','cum_std_m','cum_count']

X_train_base = fe_train_df[feat_cols].values
train_lags   = [train_raw[f'lag{k}_logy'].values for k in range(1, NLAGS+1)]
train_cumstats = np.column_stack([train_raw[c].values for c in EXTRA_TRAIN])
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base  = X_test_base[feat_cols].values
test_lags    = [test_raw[f'lag{k}_logmega'].values for k in range(1, NLAGS+1)]
test_row_sc  = test_raw['row_in_sc'].values

def make_X(base, lags_list, cumstats, row_sc):
    return np.hstack([base, np.column_stack(lags_list + [row_sc]), cumstats])

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

oof_cumstats   = np.full(len(train_raw), np.nan)
test_cumstats_list = []

print("Training oracle-LGB-cumstats...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr    = make_X(X_train_base[tr_idx],
                     [train_lags[k][tr_idx] for k in range(NLAGS)],
                     train_cumstats[tr_idx], row_sc_arr[tr_idx])
    y_tr    = y_log[tr_idx]
    X_vtrue = make_X(X_train_base[val_idx],
                     [train_lags[k][val_idx] for k in range(NLAGS)],
                     train_cumstats[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_log[val_idx])],
              callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(-1)])

    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted_log = np.log1p(mega_oof_id[val_sorted])
    mega_val_sorted_raw = mega_oof_id[val_sorted]

    fold_cs = np.zeros(len(val_sorted))
    cum_max_v  = np.full(len(val_sorted), global_mean)
    cum_mean_v = np.full(len(val_sorted), global_mean)
    cum_sum_v  = np.zeros(len(val_sorted))
    cum_sum2_v = np.zeros(len(val_sorted))
    cum_n_v    = np.zeros(len(val_sorted))
    prev_preds = np.full(len(val_sorted), global_mean)  # mega33 proxy for cumstats

    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()

        # Lag features
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_val_sorted_log[row_in_sc_vals == (pos-k)])

        # Cumulative stats from mega33 proxy
        cs_max  = cum_max_v[pos_mask]
        cs_mean = np.where(cum_n_v[pos_mask] > 0,
                           cum_sum_v[pos_mask] / cum_n_v[pos_mask],
                           np.full(n_pos, global_mean))
        cs_std  = np.where(cum_n_v[pos_mask] > 1,
                           np.sqrt(np.maximum(0, cum_sum2_v[pos_mask]/cum_n_v[pos_mask] - cs_mean**2)),
                           np.zeros(n_pos))
        cs_n    = cum_n_v[pos_mask]
        cs_arr  = np.column_stack([cs_max, cs_mean, cs_std, cs_n])

        X_pos = make_X(X_train_base[pos_idx], lags_pos, cs_arr, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        fold_cs[pos_mask] = np.maximum(0, np.expm1(pred_log))

        # Update cumstats with mega33 proxy at this position
        if pos < 24:
            cur_raw = mega_val_sorted_raw[row_in_sc_vals == pos]
            cum_max_v[pos_mask]  = np.maximum(cum_max_v[pos_mask], cur_raw)
            cum_sum_v[pos_mask]  += cur_raw
            cum_sum2_v[pos_mask] += cur_raw**2
            cum_n_v[pos_mask]    += 1

    oof_cumstats[val_sorted] = fold_cs

    # Test inference (similar cumstats building)
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted_log = np.log1p(mega_test_id[test_sorted])
    mega_test_sorted_raw = mega_test_id[test_sorted]

    test_cs = np.zeros(len(test_raw))
    t_cum_max  = np.full(len(test_raw), global_mean)
    t_cum_sum  = np.zeros(len(test_raw))
    t_cum_sum2 = np.zeros(len(test_raw))
    t_cum_n    = np.zeros(len(test_raw))

    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_test_sorted_log[test_rsc_vals == (pos-k)])
        tm = t_cum_max[pos_mask]
        tn = t_cum_n[pos_mask]
        ts = t_cum_sum[pos_mask]
        ts2 = t_cum_sum2[pos_mask]
        t_mean = np.where(tn > 0, ts/tn, np.full(n_pos, global_mean))
        t_std  = np.where(tn > 1, np.sqrt(np.maximum(0, ts2/tn - t_mean**2)), np.zeros(n_pos))
        cs_arr = np.column_stack([tm, t_mean, t_std, tn])
        X_pos  = make_X(X_test_base[pos_idx], lags_pos, cs_arr, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        test_cs[pos_idx] = np.maximum(0, np.expm1(pred_log))
        if pos < 24:
            cur_raw = mega_test_sorted_raw[test_rsc_vals == pos]
            t_cum_max[pos_mask]  = np.maximum(t_cum_max[pos_mask], cur_raw)
            t_cum_sum[pos_mask]  += cur_raw
            t_cum_sum2[pos_mask] += cur_raw**2
            t_cum_n[pos_mask]    += 1
    test_cumstats_list.append(test_cs)

    mae_fold = np.mean(np.abs(fold_cs - y_true[val_sorted]))
    elapsed  = time.time() - t0
    best_it  = model.best_iteration_ if hasattr(model, 'best_iteration_') else -1
    print(f"Fold {fold_i+1}: oracle-cumstats={mae_fold:.4f}  best_iter={best_it}  ({elapsed:.0f}s)", flush=True)

test_cs_avg = np.mean(test_cumstats_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_cumstats.npy', oof_cumstats)
np.save('results/oracle_seq/test_C_cumstats.npy', test_cs_avg)
print("Saved oof_seqC_cumstats.npy, test_C_cumstats.npy", flush=True)

# Quick summary
with open('results/mega33_final.pkl','rb') as f:
    d2 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id_to_ls2 = [ls_pos2[rid] for rid in train_raw['ID'].values]
mega_oof2  = d2['meta_avg_oof'][id_to_ls2]
rank_oof2  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls2]
iter1_oof2 = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls2]
iter2_oof2 = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls2]
iter3_oof2 = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls2]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed2 = (fw['mega33']*mega_oof2 + fw['rank_adj']*rank_oof2 +
          fw['iter_r1']*iter1_oof2 + fw['iter_r2']*iter2_oof2 + fw['iter_r3']*iter3_oof2)
y2 = train_raw['avg_delay_minutes_next_30m'].values
xgb2 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv22 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
cs_mae = np.mean(np.abs(oof_cumstats - y2))
fixed_mae = np.mean(np.abs(fixed2 - y2))
res_fixed = y2 - fixed2
corr_cs = np.corrcoef(res_fixed, oof_cumstats - fixed2)[0,1]
print(f"\noracle-cumstats MAE: {cs_mae:.4f}  residual_corr={corr_cs:.4f}", flush=True)
print(f"vs XGB corr: {np.corrcoef(oof_cumstats, xgb2)[0,1]:.4f}", flush=True)
print(f"vs Lv2 corr: {np.corrcoef(oof_cumstats, lv22)[0,1]:.4f}", flush=True)
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = (1-w)*fixed2 + w*oof_cumstats
    mm = np.mean(np.abs(bl - y2))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best static blend: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
