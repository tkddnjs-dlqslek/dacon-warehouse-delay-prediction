"""
KNN Scenario Oracle
Hypothesis: For test unseen layouts (64% busier), find k most similar training scenarios
by warehouse load profile → use their ACTUAL avg_delay as prediction signal.

Unlike global_rank (percentile-only, no target info), KNN directly maps
"similar load → similar delay" from training data.

Leave-one-layout-out KNN:
- For layout L's training scenarios: neighbors = scenarios from OTHER layouts
- For test scenarios (unseen): neighbors = ALL training scenarios
This avoids target leakage while being consistent across folds.

New features (per scenario, broadcast to all 25 rows):
- knn5_mean: mean target of 5 nearest training scenarios
- knn5_std: std target of 5 nearest scenarios (uncertainty)
- knn10_mean: mean target of 10 nearest scenarios
- knn5_dist: L2 distance to nearest scenario (typicality)
- knn_nearest_congestion: congestion of nearest scenario
- knn_nearest_inflow: inflow of nearest scenario
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist

OUT_OOF  = 'results/oracle_seq/oof_seqC_knn_scenario.npy'
OUT_TEST = 'results/oracle_seq/test_C_knn_scenario.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('KNN Scenario Oracle')
print('  Direct target mapping: similar load profile → similar delay')
print('  Leave-one-layout-out KNN (no target leakage)')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# v30 base features
print('\n[1] v30 feature cache 로드...')
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

_tr = fe_tr['train_fe']
_tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord    = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_tr_base = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()

_te = pd.DataFrame(fe_te)
_te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord    = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_te_base = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()

# Build scenario-level aggregates
print('\n[2] KNN Scenario Features 계산...')
KNN_AGG_COLS = [
    'order_inflow_15m', 'congestion_score', 'fault_count_15m',
    'blocked_path_15m', 'charge_queue_length', 'low_battery_ratio', 'robot_active',
]
# Scenario aggregates for KNN search space
train_sc = train_raw.groupby(['layout_id','scenario_id']).agg(
    **{f'{c}_mean': (c, 'mean') for c in KNN_AGG_COLS},
    **{f'{c}_max': (c, 'max') for c in ['order_inflow_15m','congestion_score','fault_count_15m']},
    sc_target_mean=('avg_delay_minutes_next_30m', 'mean'),
).reset_index()

test_sc = test_raw.groupby(['layout_id','scenario_id']).agg(
    **{f'{c}_mean': (c, 'mean') for c in KNN_AGG_COLS},
    **{f'{c}_max': (c, 'max') for c in ['order_inflow_15m','congestion_score','fault_count_15m']},
).reset_index()

agg_feat_cols = [f'{c}_mean' for c in KNN_AGG_COLS] + \
                [f'{c}_max' for c in ['order_inflow_15m','congestion_score','fault_count_15m']]

print(f'  Train scenarios: {len(train_sc)}, Test scenarios: {len(test_sc)}')
print(f'  KNN feature space: {len(agg_feat_cols)} dims')

# Normalize for distance computation using training statistics
scaler = StandardScaler()
train_feats = scaler.fit_transform(train_sc[agg_feat_cols].values.astype(np.float32))
test_feats  = scaler.transform(test_sc[agg_feat_cols].values.astype(np.float32))

train_targets = train_sc['sc_target_mean'].values.astype(np.float32)
train_layout_ids = train_sc['layout_id'].values
train_sc_ids = train_sc['scenario_id'].values

K_VALS = [5, 10, 20]
K_MAX = max(K_VALS)

print(f'\n  Computing leave-one-layout-out KNN for train scenarios...')
t1 = time.time()
unique_layouts = np.unique(train_layout_ids)

knn_feat_names_pre = [f'knn{k}_mean' for k in K_VALS] + ['knn1_target', 'knn5_std', 'knn1_dist']
n_knn_feats_pre = len(knn_feat_names_pre)  # = 6
# Pre-allocate KNN feature arrays for train scenarios
knn_tr = np.full((len(train_sc), n_knn_feats_pre), np.nan, dtype=np.float32)

CHUNK = 50  # process this many layouts at once to balance memory vs speed
layout_chunks = [unique_layouts[i:i+CHUNK] for i in range(0, len(unique_layouts), CHUNK)]

for chunk_i, chunk_layouts in enumerate(layout_chunks):
    # Query indices = scenarios from chunk layouts
    query_mask = np.isin(train_layout_ids, chunk_layouts)
    query_feats = train_feats[query_mask]
    query_idxs  = np.where(query_mask)[0]

    # Reference indices = scenarios NOT from ANY of these chunk layouts
    ref_mask = ~np.isin(train_layout_ids, chunk_layouts)
    ref_feats   = train_feats[ref_mask]
    ref_targets = train_targets[ref_mask]

    if len(ref_feats) < K_MAX:
        continue

    # Distance matrix: (n_query, n_ref)
    dists = cdist(query_feats, ref_feats, metric='euclidean').astype(np.float32)

    # Get K_MAX nearest for each query
    part_idx = np.argpartition(dists, K_MAX, axis=1)[:, :K_MAX]
    sorted_k = np.argsort(dists[np.arange(len(query_feats))[:, None], part_idx], axis=1)
    top_k_idx = part_idx[np.arange(len(query_feats))[:, None], sorted_k]  # (n_q, K_MAX)

    for fi, k in enumerate(K_VALS):
        top_k_targets = ref_targets[top_k_idx[:, :k]]  # (n_q, k)
        knn_tr[query_idxs, fi] = top_k_targets.mean(axis=1)

    knn_tr[query_idxs, len(K_VALS)]     = ref_targets[top_k_idx[:, 0]]  # nearest target
    knn_tr[query_idxs, len(K_VALS) + 1] = ref_targets[top_k_idx[:, :5]].std(axis=1)  # std5
    knn_tr[query_idxs, len(K_VALS) + 2] = dists[np.arange(len(query_feats)), top_k_idx[:, 0]]  # dist

    if (chunk_i+1) % 5 == 0 or chunk_i == len(layout_chunks)-1:
        print(f'    chunk {chunk_i+1}/{len(layout_chunks)} done', flush=True)

print(f'  Train KNN done ({time.time()-t1:.0f}s)')

# Replace NaN with column means
for j in range(knn_tr.shape[1]):
    col = knn_tr[:, j]
    nan_mask = np.isnan(col)
    if nan_mask.any():
        knn_tr[nan_mask, j] = np.nanmean(col)

print(f'  NaN count after fill: {np.isnan(knn_tr).sum()} (should be 0)')
print(f'\n  Computing test scenario KNN (all training scenarios as reference)...')
t1 = time.time()
# Full distance matrix: test vs train
dists_te = cdist(test_feats, train_feats, metric='euclidean').astype(np.float32)
part_idx_te = np.argpartition(dists_te, K_MAX, axis=1)[:, :K_MAX]
sorted_te = np.argsort(dists_te[np.arange(len(test_feats))[:, None], part_idx_te], axis=1)
top_k_idx_te = part_idx_te[np.arange(len(test_feats))[:, None], sorted_te]

knn_feat_names = [f'knn{k}_mean' for k in K_VALS] + ['knn1_target', 'knn5_std', 'knn1_dist']
n_knn_feats = len(knn_feat_names)
knn_te = np.full((len(test_sc), n_knn_feats), np.nan, dtype=np.float32)
for fi, k in enumerate(K_VALS):
    top_k_targets = train_targets[top_k_idx_te[:, :k]]
    knn_te[:, fi] = top_k_targets.mean(axis=1)
knn_te[:, len(K_VALS)]     = train_targets[top_k_idx_te[:, 0]]
knn_te[:, len(K_VALS) + 1] = train_targets[top_k_idx_te[:, :5]].std(axis=1)
knn_te[:, len(K_VALS) + 2] = dists_te[np.arange(len(test_feats)), top_k_idx_te[:, 0]]

del dists_te; gc.collect()
print(f'  Test KNN done ({time.time()-t1:.0f}s)')

knn_feat_names = knn_feat_names_pre
n_knn_feats = n_knn_feats_pre

# Domain shift analysis
print(f'\n  Domain shift (train vs test KNN features):')
for i, fn in enumerate(knn_feat_names):
    tr_m = np.mean(knn_tr[:, i])
    te_m = np.mean(knn_te[:, i])
    print(f'    {fn}: train={tr_m:.3f}, test={te_m:.3f}, delta={te_m-tr_m:+.3f}')

# Correlation with target (train scenarios only)
print(f'\n  KNN feature correlations with scenario target mean:')
for i, fn in enumerate(knn_feat_names):
    corr = float(np.corrcoef(knn_tr[:, i], train_targets)[0,1])
    print(f'    {fn}: {corr:+.4f}')

# Now broadcast scenario-level KNN features to row level (vectorized)
print(f'\n  Broadcasting KNN features to row level...')
train_sc['_knn_sc_idx'] = np.arange(len(train_sc))
test_sc['_knn_sc_idx']  = np.arange(len(test_sc))

tr_sc_idx = train_raw[['layout_id','scenario_id']].merge(
    train_sc[['layout_id','scenario_id','_knn_sc_idx']], on=['layout_id','scenario_id'], how='left'
)['_knn_sc_idx'].values
knn_tr_row = knn_tr[tr_sc_idx.astype(int)]

te_sc_idx = test_raw[['layout_id','scenario_id']].merge(
    test_sc[['layout_id','scenario_id','_knn_sc_idx']], on=['layout_id','scenario_id'], how='left'
)['_knn_sc_idx'].values
knn_te_row = knn_te[te_sc_idx.astype(int)]

print(f'  Done. knn_tr_row: {knn_tr_row.shape}, knn_te_row: {knn_te_row.shape}')

# mega33 proxy for oracle sequential features
print('\n[3] mega33 proxy 로드...')
with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
mega_oof = d['meta_avg_oof'][id2]

test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]
mega_test = d['meta_avg_test'][te_id2]
del d; gc.collect()

train_raw['mega_oof'] = mega_oof
test_raw['mega_test'] = mega_test
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

def add_oracle_feats(df, pred_col):
    grp = df.groupby(['layout_id','scenario_id'])
    df = df.copy()
    df['lag1']    = grp[pred_col].shift(1).fillna(df[pred_col].mean())
    df['lag2']    = grp[pred_col].shift(2).fillna(df[pred_col].mean())
    df['sc_mean'] = grp[pred_col].transform('mean')
    return np.column_stack([
        df['lag1'].values, df['lag2'].values, df['sc_mean'].values,
        df['row_in_sc'].values.astype(np.float32)/25.0,
    ]).astype(np.float32)

of_tr = add_oracle_feats(train_raw, 'mega_oof')
of_te = add_oracle_feats(test_raw,  'mega_test')

X_tr_full = np.hstack([X_tr_base, knn_tr_row, of_tr])
X_te_full = np.hstack([X_te_base, knn_te_row, of_te])
print(f'  X_tr_full: {X_tr_full.shape} (v30:{X_tr_base.shape[1]} + knn:{n_knn_feats} + oracle:4)')

PARAMS = dict(
    objective='huber', alpha=0.9,
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, max_depth=8,
    min_child_samples=80,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

print('\n[4] 5-fold GroupKFold 학습...')
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)
oof    = np.zeros(len(train_raw))
test_preds = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    val_df = train_raw.iloc[val_idx_sorted].copy()
    val_df['mega_oof_val'] = mega_oof[val_idx_sorted]
    of_val = add_oracle_feats(val_df, 'mega_oof_val')
    X_val = np.hstack([X_tr_base[val_idx_sorted], knn_tr_row[val_idx_sorted], of_val])

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr_full[tr_idx], y_true[tr_idx],
              eval_set=[(X_val, y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    fold_pred = np.clip(model.predict(X_val), 0, None)
    oof[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds.append(np.clip(model.predict(X_te_full), 0, None))
    print(f'  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model, val_df, X_val; gc.collect()

overall_mae = mean_absolute_error(y_true, oof)
test_avg = np.mean(test_preds, axis=0)
print(f'\nOverall OOF: {overall_mae:.5f}  ({time.time()-t0:.0f}s total)')

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
print(f'\noracle_NEW OOF:    {mae(oracle_new_oof):.5f}')
print(f'knn_scenario OOF:  {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, knn_scenario): {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')
print('Done.')
