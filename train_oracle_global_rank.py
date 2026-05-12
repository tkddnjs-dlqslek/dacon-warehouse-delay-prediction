"""
Cross-Scenario Global Rank Oracle
=================================
v30/v31 has sc_mean, sc_max etc. as ABSOLUTE values.
Tree models can't extrapolate: test scenario with order_inflow at 164% training max
→ same last leaf as all training samples near max → under-prediction.

NEW: Compute scenario-level aggregates then rank them GLOBALLY across all
training scenarios. Use training distribution to rank test scenarios.
  sc_inflow_pct: percentile of this scenario's mean inflow among ALL training scenarios
  sc_congestion_pct: same for congestion
  sc_stress_pct: composite stress percentile

Why this helps domain shift:
  - Unseen test scenarios map to 95th-99th percentile globally
  - Model has seen 99th-percentile scenarios for SOME metrics in training
  - Through percentile space, cross-metric knowledge transfer is possible
  - "99th pct stress scenario → high delay" generalizes regardless of which specific
    metric is driving stress
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from scipy.stats import percentileofscore

OUT_OOF  = 'results/oracle_seq/oof_seqC_global_rank.npy'
OUT_TEST = 'results/oracle_seq/test_C_global_rank.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); import sys; sys.exit(0)

t0 = time.time()
print('='*60)
print('Cross-Scenario Global Rank Oracle')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# ── [1] v30 base features ───────────────────────────────────────────────────
print('\n[1] v30 feature cache 로드...')
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f: fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f: fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

_tr = fe_tr['train_fe']
_tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord    = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_tr_base  = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()

_te = pd.DataFrame(fe_te)
_te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord    = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_te_base  = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()

# ── [2] Cross-Scenario Global Rank Features ─────────────────────────────────
print('\n[2] Cross-Scenario Global Rank 피처 계산...')

# Scenario-level aggregates from RAW data (not from scenario aggregates)
SCENARIO_AGGS = {
    'order_inflow_15m':    ['mean', 'max', 'sum'],
    'congestion_score':    ['mean', 'max'],
    'fault_count_15m':     ['sum', 'mean'],
    'blocked_path_15m':    ['sum', 'mean'],
    'charge_queue_length': ['mean', 'max'],
    'low_battery_ratio':   ['mean', 'max'],
    'robot_active':        ['mean', 'min'],
}

def compute_scenario_aggs(df):
    sc_grp = df.groupby(['layout_id', 'scenario_id'])
    rows = []
    for (lid, sid), grp in sc_grp:
        row = {'layout_id': lid, 'scenario_id': sid}
        for col, funcs in SCENARIO_AGGS.items():
            if col in grp.columns:
                for fn in funcs:
                    row[f'sc_{col}_{fn}'] = getattr(grp[col], fn)()
        rows.append(row)
    return pd.DataFrame(rows)

print('  Computing train scenario aggregates...')
train_sc = compute_scenario_aggs(train_raw)
print(f'  Train scenarios: {len(train_sc)}, features: {len(train_sc.columns)-2}')

# Composite stress score (higher = more stress)
train_sc['sc_composite_stress'] = (
    train_sc['sc_order_inflow_15m_mean'].fillna(0) / (train_sc['sc_order_inflow_15m_mean'].fillna(0).max() + 1e-6) +
    train_sc['sc_congestion_score_mean'].fillna(0) / (train_sc['sc_congestion_score_mean'].fillna(0).max() + 1e-6) +
    train_sc['sc_charge_queue_length_mean'].fillna(0) / (train_sc['sc_charge_queue_length_mean'].fillna(0).max() + 1e-6) +
    train_sc['sc_fault_count_15m_sum'].fillna(0) / (train_sc['sc_fault_count_15m_sum'].fillna(0).max() + 1e-6)
) / 4.0

agg_cols = [c for c in train_sc.columns if c not in ['layout_id','scenario_id']]
print(f'  Scenario agg columns: {len(agg_cols)}')

# Compute global percentile ranks using training distribution
print('  Computing global percentile ranks (training-based)...')
print('  (using vectorized approach for speed)')

def get_pct_rank(train_vals_sorted, query_vals):
    """Fast vectorized percentile using sorted training array."""
    n = len(train_vals_sorted)
    idx = np.searchsorted(train_vals_sorted, query_vals, side='right')
    return (idx / n).astype(np.float32)

# Build sorted arrays for each agg feature
sorted_train = {}
for col in agg_cols:
    sv = np.sort(train_sc[col].fillna(0).values)
    sorted_train[col] = sv

# Compute test scenario aggregates
print('  Computing test scenario aggregates...')
test_sc = compute_scenario_aggs(test_raw)
test_sc['sc_composite_stress'] = (
    test_sc['sc_order_inflow_15m_mean'].fillna(0) / (train_sc['sc_order_inflow_15m_mean'].fillna(0).max() + 1e-6) +
    test_sc['sc_congestion_score_mean'].fillna(0) / (train_sc['sc_congestion_score_mean'].fillna(0).max() + 1e-6) +
    test_sc['sc_charge_queue_length_mean'].fillna(0) / (train_sc['sc_charge_queue_length_mean'].fillna(0).max() + 1e-6) +
    test_sc['sc_fault_count_15m_sum'].fillna(0) / (train_sc['sc_fault_count_15m_sum'].fillna(0).max() + 1e-6)
) / 4.0

# Compute percentile ranks for train and test scenarios
train_rank_feats = pd.DataFrame({'layout_id': train_sc['layout_id'], 'scenario_id': train_sc['scenario_id']})
test_rank_feats  = pd.DataFrame({'layout_id': test_sc['layout_id'],  'scenario_id': test_sc['scenario_id']})

for col in agg_cols:
    sv = sorted_train[col]
    train_rank_feats[f'grk_{col}'] = get_pct_rank(sv, train_sc[col].fillna(0).values)
    test_rank_feats[f'grk_{col}']  = get_pct_rank(sv, test_sc[col].fillna(0).values)

rank_feat_cols = [c for c in train_rank_feats.columns if c.startswith('grk_')]
print(f'  Global rank features: {len(rank_feat_cols)}')

# Domain shift check: compare train vs test rank distributions
print('\n  Domain shift analysis (train vs test scenario ranks):')
for col in ['grk_sc_order_inflow_15m_mean', 'grk_sc_congestion_score_mean', 'grk_sc_composite_stress']:
    if col in rank_feat_cols:
        tr_mean = train_rank_feats[col].mean()
        te_mean = test_rank_feats[col].mean()
        print(f'    {col}: train_mean={tr_mean:.3f}, test_mean={te_mean:.3f}, delta={te_mean-tr_mean:+.3f}')

# Merge rank features back to row level (all rows in same scenario get same rank)
train_merged = train_raw[['ID','layout_id','scenario_id']].merge(
    train_rank_feats, on=['layout_id','scenario_id'], how='left'
)
test_merged  = test_raw[['ID','layout_id','scenario_id']].merge(
    test_rank_feats, on=['layout_id','scenario_id'], how='left'
)

# Align to row order
train_id2idx = {v:i for i,v in enumerate(train_merged['ID'].values)}
train_ord    = np.array([train_id2idx[i] for i in train_raw['ID'].values])
test_id2idx  = {v:i for i,v in enumerate(test_merged['ID'].values)}
test_ord     = np.array([test_id2idx[i] for i in test_raw['ID'].values])

GR_tr = train_merged[rank_feat_cols].values[train_ord].astype(np.float32)
GR_te = test_merged[rank_feat_cols].values[test_ord].astype(np.float32)

# Correlation with target
print('\n  Top feature correlations with target:')
corrs = []
for i, col in enumerate(rank_feat_cols):
    c = float(np.corrcoef(GR_tr[:,i], y_true)[0,1])
    corrs.append((abs(c), c, col))
corrs.sort(reverse=True)
for _, c, col in corrs[:8]:
    print(f'    {col}: {c:+.4f}')

del train_merged, test_merged, train_rank_feats, test_rank_feats
gc.collect()

# ── [3] mega33 proxy oracle seq features ────────────────────────────────────
print('\n[3] mega33 proxy 로드...')
with open('results/mega33_final.pkl', 'rb') as f: d = pickle.load(f)
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
    df = df.copy()
    grp = df.groupby(['layout_id','scenario_id'])
    df['lag1']    = grp[pred_col].shift(1).fillna(df[pred_col].mean())
    df['lag2']    = grp[pred_col].shift(2).fillna(df[pred_col].mean())
    df['sc_mean'] = grp[pred_col].transform('mean')
    return np.column_stack([
        df['lag1'].values, df['lag2'].values, df['sc_mean'].values,
        df['row_in_sc'].values.astype(np.float32)/25.0,
    ]).astype(np.float32)

of_tr = add_oracle_feats(train_raw, 'mega_oof')
of_te = add_oracle_feats(test_raw,  'mega_test')

# Full feature matrix: v30 + global rank + oracle seq
X_tr_full = np.hstack([X_tr_base, GR_tr, of_tr])
X_te_full = np.hstack([X_te_base, GR_te, of_te])
print(f'  X_tr_full: {X_tr_full.shape} (v30:{X_tr_base.shape[1]} + grk:{GR_tr.shape[1]} + oracle:4)')

# ── [4] GroupKFold training ──────────────────────────────────────────────────
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
    X_val = np.hstack([X_tr_base[val_idx_sorted], GR_tr[val_idx_sorted], of_val])

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
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'\noracle_NEW OOF: {mae_fn(oracle_new_oof):.5f}')
print(f'global_rank  OOF: {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, global_rank): {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae_fn(b)-mae_fn(oracle_new_oof):+.5f}')
print('Done.')
