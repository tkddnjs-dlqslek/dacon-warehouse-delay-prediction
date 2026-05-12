"""
Layout-Stats Oracle — Layout-Level Aggregate Features
Key insight: v30 has ONLY within-scenario features (sc_mean, sc_max, lag, lead).
It has ZERO layout-level statistics (mean across ALL scenarios for a given layout).

Genuine gap: layout_mean_inflow for training = 94.5, for test = 131.7 (+39%).
If model has layout_mean_inflow as a feature, it learns: high-inflow layouts → high delays.
This cross-layout pattern EXTRAPOLATES to test layouts with even higher inflow.

Without this feature, oracle_NEW only knows "this scenario's inflow is X" but not
"this layout's TYPICAL inflow is X → this is a fundamentally busy warehouse."

v30 has sc_mean (per scenario), but NOT ly_mean (per layout across all scenarios).
These are different: a scenario can be busy in an otherwise quiet layout, or
the layout itself can be busy (all scenarios high inflow).

Layout-level features created:
- ly_inflow_mean/max/std: typical order load for this layout
- ly_congestion_mean/max: typical floor congestion
- ly_pack_mean/max: typical packing utilization (fold 2: high pack + low congestion = bottleneck)
- ly_robot_mean: typical robot count
- ly_fault_sum: typical fault frequency
- rel_sc_inflow = sc_inflow / ly_inflow - 1: how loaded is this scenario vs layout normal
- rel_sc_pack = sc_pack / ly_pack - 1: similar for packing
- rel_sc_congestion = sc_congestion / ly_congestion - 1
- ly_scenario_count: layout size (more data → more reliable layout mean)

Expected benefit: model can learn the cross-layout relationship between
layout load level and delay → generalizes to higher-load test layouts.

Note: layout stats computed from ALL training data (no target used → no leakage).
For test: layout stats from ALL test data (valid, test layout IDs don't appear in training).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqD_layout_stats.npy'
OUT_TEST = 'results/oracle_seq/test_D_layout_stats.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Layout-Stats Oracle — Cross-layout generalization features')
print('  ly_* features: layout-level aggregates across all scenarios')
print('  Captures: training ly_inflow≈94 vs test ly_inflow≈132 (+39%)')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# --- 1. v30 base features ---
print('\n[1] v30 feature 로드...')
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

# --- 2. Layout-level statistics ---
print('\n[2] Layout-level statistics 계산...')
LOAD_COLS = ['order_inflow_15m', 'congestion_score', 'pack_utilization',
             'robot_active', 'fault_count_15m', 'robot_idle',
             'blocked_path_15m', 'outbound_truck_wait_min']

def compute_layout_stats(df, split='train'):
    """
    Compute layout-level aggregate features.
    These capture the BASE LOAD LEVEL of each layout across all its scenarios.
    For training: computed from all training rows for each layout.
    For test: computed from all test rows for each layout (valid, no target leakage).
    """
    df2 = df.copy()

    # Compute scenario-level means first (to avoid outlier-heavy instantaneous values)
    sc_aggs = {}
    for col in LOAD_COLS:
        if col in df2.columns:
            sc_aggs[f'_{col}_sc_mean'] = (df2.groupby(['layout_id','scenario_id'])[col]
                                          .transform('mean'))

    # Then aggregate to layout level
    all_cols = []
    all_data = []

    for col in LOAD_COLS:
        if col not in df2.columns:
            for agg in ['mean', 'max', 'std']:
                all_cols.append(f'ly_{col}_{agg}')
                all_data.append(np.zeros(len(df2), dtype=np.float32))
            continue

        col_sc = df2.groupby(['layout_id','scenario_id'])[col].transform('mean')
        grp_ly = df2.groupby('layout_id')

        for agg in ['mean', 'max', 'std']:
            feat_name = f'ly_{col}_{agg}'
            all_cols.append(feat_name)
            if agg == 'mean':
                val = grp_ly[col].transform('mean')
            elif agg == 'max':
                val = grp_ly[col].transform('max')
            elif agg == 'std':
                val = grp_ly[col].transform('std').fillna(0)
            all_data.append(val.fillna(0).values.astype(np.float32))

    # Relative load features: current scenario vs layout norm
    # rel_sc_X = (sc_mean_X - ly_mean_X) / (ly_std_X + eps)
    # Captures: "is THIS scenario busier/quieter than typical for this layout?"
    for col in ['order_inflow_15m', 'congestion_score', 'pack_utilization']:
        if col not in df2.columns:
            all_cols.append(f'rel_sc_{col}')
            all_data.append(np.zeros(len(df2), dtype=np.float32))
            continue

        sc_mean = df2.groupby(['layout_id','scenario_id'])[col].transform('mean')
        ly_mean = df2.groupby('layout_id')[col].transform('mean')
        ly_std  = df2.groupby('layout_id')[col].transform('std').fillna(1).clip(lower=0.01)

        rel_val = ((sc_mean - ly_mean) / ly_std).fillna(0).clip(-5, 5)
        all_cols.append(f'rel_sc_{col}')
        all_data.append(rel_val.values.astype(np.float32))

    # Layout scenario count (data richness)
    ly_sc_count = df2.groupby('layout_id')['scenario_id'].transform('nunique')
    all_cols.append('ly_scenario_count')
    all_data.append(ly_sc_count.values.astype(np.float32))

    # Pack / congestion ratio at layout level (identifies packing-bottleneck layouts)
    if 'pack_utilization' in df2.columns and 'congestion_score' in df2.columns:
        ly_pack_mean = df2.groupby('layout_id')['pack_utilization'].transform('mean')
        ly_cong_mean = df2.groupby('layout_id')['congestion_score'].transform('mean')
        ly_pack_cong_ratio = (ly_pack_mean / (ly_cong_mean + 0.01)).clip(0, 10)
        all_cols.append('ly_pack_congestion_ratio')
        all_data.append(ly_pack_cong_ratio.values.astype(np.float32))
    else:
        all_cols.append('ly_pack_congestion_ratio')
        all_data.append(np.zeros(len(df2), dtype=np.float32))

    result = np.column_stack(all_data)
    return result, all_cols

print('  Computing training layout stats...')
t1 = time.time()
ly_tr, ly_feat_names = compute_layout_stats(train_raw, 'train')
print(f'  Done ({time.time()-t1:.1f}s). Features: {len(ly_feat_names)}')

print('  Computing test layout stats...')
t1 = time.time()
ly_te, _ = compute_layout_stats(test_raw, 'test')
print(f'  Done ({time.time()-t1:.1f}s).')

# Domain shift analysis
print(f'\n  Layout stats domain shift:')
for i, fn in enumerate(ly_feat_names):
    tr_m = ly_tr[:, i].mean()
    te_m = ly_te[:, i].mean()
    shift = (te_m - tr_m) / (abs(tr_m) + 1e-6) * 100
    if abs(shift) > 10:
        print(f'    {fn}: train={tr_m:.3f}, test={te_m:.3f}, shift={shift:+.1f}%')

# Fold 2 layout analysis
print(f'\n  Fold 2 layouts ly_* comparison:')
groups = train_raw['layout_id'].values
gkf_temp = GroupKFold(n_splits=5)
fold_ly_stats = {}
for fi, (_, val_idx) in enumerate(gkf_temp.split(np.arange(len(train_raw)), groups=groups)):
    val_idx_sorted = np.sort(val_idx)
    fold_ly_stats[fi+1] = ly_tr[val_idx_sorted].mean(axis=0)

for i, fn in enumerate(ly_feat_names[:10]):
    vals = [f'{fold_ly_stats[f][i]:.3f}' for f in range(1, 6)]
    print(f'    {fn}: [' + ', '.join(vals) + ']')

# --- 3. mega33 proxy ---
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

# Full feature matrix: v30 + layout_stats + oracle_lag
X_tr_full = np.hstack([X_tr_base, ly_tr, of_tr])
X_te_full = np.hstack([X_te_base, ly_te, of_te])
print(f'  X_tr_full: {X_tr_full.shape} (v30:{X_tr_base.shape[1]} + layout:{ly_tr.shape[1]} + oracle:4)')

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
gkf    = GroupKFold(n_splits=5)
oof    = np.zeros(len(train_raw))
test_preds = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    val_df = train_raw.iloc[val_idx_sorted].copy()
    val_df['mega_oof_val'] = mega_oof[val_idx_sorted]
    of_val = add_oracle_feats(val_df, 'mega_oof_val')
    X_val = np.hstack([X_tr_base[val_idx_sorted], ly_tr[val_idx_sorted], of_val])

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
print(f'\noracle_NEW OOF:       {mae(oracle_new_oof):.5f}')
print(f'layout_stats OOF:    {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, layout_stats): {corr:.4f}')
print(f'\nFold-level MAE:')
for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    val_idx_sorted = np.sort(val_idx)
    fm_new = mean_absolute_error(y_true[val_idx_sorted], oracle_new_oof[val_idx_sorted])
    fm_lay = mean_absolute_error(y_true[val_idx_sorted], oof[val_idx_sorted])
    print(f'  Fold {fi+1}: layout_stats={fm_lay:.5f} oracle_NEW={fm_new:.5f} delta={fm_lay-fm_new:+.5f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')

# Test prediction comparison
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
oracle_test = np.clip(0.64*fixed_test + 0.12*np.load('results/oracle_seq/test_C_xgb.npy')
                    + 0.16*np.load('results/oracle_seq/test_C_log_v2.npy')
                    + 0.08*np.load('results/oracle_seq/test_C_xgb_remaining.npy'), 0, None)
print(f'\nTest pred analysis:')
print(f'  oracle_NEW test mean: {oracle_test.mean():.3f}')
print(f'  layout_stats test mean: {test_avg.mean():.3f} (delta={test_avg.mean()-oracle_test.mean():+.3f})')
print('Done.')
