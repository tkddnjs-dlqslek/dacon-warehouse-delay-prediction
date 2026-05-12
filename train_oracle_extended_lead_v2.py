"""
Extended Lookahead Oracle v2 — Adding Pack Utilization Future Features
Key fix: v1 only had congestion/inflow future features.
Fold 2 has LOW congestion but HIGH pack_utilization — packing stage bottleneck.
v1 added confusing signals: fut_congestion low → model predicts low delay → WRONG for fold 2.

v2 adds PACKING bottleneck future features:
- fut_pack_util_5_max:  max pack_utilization in next 5 steps (HIGH for fold 2)
- fut_pack_util_5_mean: mean pack_utilization in next 5 steps
- fut_pack_util_10_max: max over 10 steps
- fut_robot_idle_5_mean: mean robot idle count (robots waiting for packing)
- fut_truck_wait_5_max: max outbound_truck_wait_min in next 5 steps

When model sees: fut_congestion LOW + fut_pack_util HIGH → packing bottleneck → HIGH DELAY
When model sees: fut_congestion HIGH → floor bottleneck → HIGH DELAY
Both mechanisms covered.

v30 only has lead1/lead2 for pack_utilization. v2 extends to lead5/lead10.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqC_extended_lead_v2.npy'
OUT_TEST = 'results/oracle_seq/test_C_extended_lead_v2.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Extended Lookahead Oracle v2 — Floor + Packing Bottleneck')
print('  NEW: fut_pack_util, fut_robot_idle, fut_truck_wait features')
print('  Fixes fold 2 packing bottleneck blind spot')
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

# Extended lookahead features v2 — floor + packing bottleneck
print('\n[2] Extended Lookahead Features v2 계산...')

LEAD_COLS = {
    # Floor bottleneck features (v1)
    'congestion_score':    {'windows': [5, 10], 'aggs': ['max', 'mean']},
    'order_inflow_15m':    {'windows': [5, 10], 'aggs': ['max', 'mean']},
    'fault_count_15m':     {'windows': [5],     'aggs': ['sum', 'max']},
    'blocked_path_15m':    {'windows': [5],     'aggs': ['sum']},
    # Packing bottleneck features (v2 NEW)
    'pack_utilization':    {'windows': [5, 10], 'aggs': ['max', 'mean']},
    'robot_idle':          {'windows': [5],     'aggs': ['mean']},
    'outbound_truck_wait_min': {'windows': [5], 'aggs': ['max', 'mean']},
    'charge_queue_length': {'windows': [5],     'aggs': ['max', 'mean']},
}

def compute_extended_lead_v2(df):
    """Compute extended lookahead features including packing bottleneck."""
    df = df.copy()
    df['row_in_sc'] = df.groupby(['layout_id','scenario_id']).cumcount()
    df['sc_size']   = df.groupby(['layout_id','scenario_id'])['row_in_sc'].transform('max') + 1

    all_feat_cols = []
    all_feat_data = []

    for col, spec in LEAD_COLS.items():
        col_vals = df[col].fillna(df[col].mean()).values.astype(np.float32) if col in df.columns else np.zeros(len(df), dtype=np.float32)
        col_mean = float(col_vals.mean())

        for window in spec['windows']:
            for agg in spec['aggs']:
                feat_name = f'fut_{col[:10]}_{window}_{agg}'
                all_feat_cols.append(feat_name)
                feat_vals = np.full(len(df), col_mean, dtype=np.float32)

                for sc_id, grp in df.groupby(['layout_id','scenario_id']):
                    idx = grp.index.values
                    n = len(idx)
                    sc_vals = col_vals[idx]

                    row_feats = np.zeros(n, dtype=np.float32)
                    for i in range(n):
                        future = sc_vals[i+1:i+1+window]
                        if len(future) == 0:
                            row_feats[i] = sc_vals[i]
                        elif agg == 'max':
                            row_feats[i] = float(future.max())
                        elif agg == 'mean':
                            row_feats[i] = float(future.mean())
                        elif agg == 'sum':
                            row_feats[i] = float(future.sum())
                    feat_vals[idx] = row_feats
                all_feat_data.append(feat_vals)

    # Pack/floor contrast feature (v2 key feature)
    # pack_high_congestion_low = fut_pack_util_5_max - fut_congestion_5_max/100
    # When this is large: packing bottleneck while floor is clear → fold 2 pattern
    # When both high: floor + packing → worst case
    # all_feat_cols.append('pack_floor_contrast')  # computed after

    # Steps remaining
    all_feat_cols.append('steps_remaining')
    all_feat_data.append(((df['sc_size'] - df['row_in_sc'] - 1) / 25.0).values.astype(np.float32))

    result = np.column_stack(all_feat_data).astype(np.float32)
    return result, all_feat_cols

print('  Computing train extended lead v2 features...')
t1 = time.time()
lead_tr, lead_feat_names = compute_extended_lead_v2(train_raw)
print(f'  Done ({time.time()-t1:.0f}s). shape: {lead_tr.shape}')

print('  Computing test extended lead v2 features...')
t1 = time.time()
lead_te, _ = compute_extended_lead_v2(test_raw)
print(f'  Done ({time.time()-t1:.0f}s). shape: {lead_te.shape}')

# Correlation analysis (top by abs)
print(f'\n  Feature correlations with target (top 15):')
corrs = []
for i, fn in enumerate(lead_feat_names):
    valid = ~np.isnan(lead_tr[:, i])
    if valid.sum() > 100:
        c = float(np.corrcoef(lead_tr[valid, i], y_true[valid])[0,1])
        corrs.append((abs(c), fn, c))
corrs.sort(reverse=True)
for _, fn, c in corrs[:15]:
    print(f'    {fn}: {c:+.4f}')

# Domain shift for pack features
print(f'\n  Pack features domain shift:')
pack_indices = [i for i, fn in enumerate(lead_feat_names) if 'pack' in fn.lower() or 'idle' in fn.lower() or 'truck' in fn.lower()]
for i in pack_indices[:8]:
    fn = lead_feat_names[i]
    tr_m = np.nanmean(lead_tr[:, i])
    te_m = np.nanmean(lead_te[:, i])
    print(f'    {fn}: train={tr_m:.3f}, test={te_m:.3f}, delta={te_m-tr_m:+.3f}')

# mega33 proxy
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

X_tr_full = np.hstack([X_tr_base, lead_tr, of_tr])
X_te_full = np.hstack([X_te_base, lead_te, of_te])
print(f'  X_tr_full: {X_tr_full.shape} (v30:{X_tr_base.shape[1]} + lead:{lead_tr.shape[1]} + oracle:4)')

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
    X_val = np.hstack([X_tr_base[val_idx_sorted], lead_tr[val_idx_sorted], of_val])

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
print(f'\noracle_NEW OOF:         {mae(oracle_new_oof):.5f}')
print(f'extended_lead_v2 OOF:   {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, extended_lead_v2): {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')
print('Done.')
