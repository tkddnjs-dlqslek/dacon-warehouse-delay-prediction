"""
Layout-Stats Oracle with Monotone Constraints on Load Features
Extension of layout_stats oracle: add monotone constraints on key load features
to ensure: higher layout load → at least as high predicted delay.

Monotone constraints on:
- ly_order_inflow_15m_mean (index varies, set programmatically)
- ly_congestion_score_mean
- ly_robot_active_mean

Why: without constraints, model might learn non-monotone patterns from training
layout noise. With constraints, model must extrapolate correctly to higher-load
test layouts.

Expected: same OOF as layout_stats, but better LB due to correct extrapolation.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqD_layout_mono.npy'
OUT_TEST = 'results/oracle_seq/test_D_layout_mono.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Layout-Stats Oracle + Monotone Constraints')
print('  Same features as layout_stats oracle')
print('  ly_inflow, ly_congestion, ly_robot: monotone increasing')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

# v30 base
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f: fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f: fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']
_tr = fe_tr['train_fe']
_tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
X_tr_base = _tr[feat_cols].values[[_tr_id2idx[i] for i in train_raw['ID'].values]].astype(np.float32)
del _tr, _tr_id2idx, fe_tr; gc.collect()
_te = pd.DataFrame(fe_te)
_te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
X_te_base = _te.reindex(columns=feat_cols, fill_value=0).values[[_te_id2idx[i] for i in test_raw['ID'].values]].astype(np.float32)
del _te, _te_id2idx, fe_te; gc.collect()

# Layout stats
LOAD_COLS = ['order_inflow_15m', 'congestion_score', 'pack_utilization',
             'robot_active', 'fault_count_15m', 'robot_idle',
             'blocked_path_15m', 'outbound_truck_wait_min']

def compute_layout_stats(df):
    df2 = df.copy()
    all_cols, all_data = [], []
    for col in LOAD_COLS:
        if col not in df2.columns:
            for agg in ['mean','max','std']:
                all_cols.append(f'ly_{col}_{agg}'); all_data.append(np.zeros(len(df2), np.float32))
            continue
        grp = df2.groupby('layout_id')
        for agg in ['mean','max','std']:
            all_cols.append(f'ly_{col}_{agg}')
            if agg=='mean': v = grp[col].transform('mean')
            elif agg=='max': v = grp[col].transform('max')
            else: v = grp[col].transform('std').fillna(0)
            all_data.append(v.fillna(0).values.astype(np.float32))
    for col in ['order_inflow_15m','congestion_score','pack_utilization']:
        if col not in df2.columns:
            all_cols.append(f'rel_sc_{col}'); all_data.append(np.zeros(len(df2), np.float32)); continue
        sc_mean = df2.groupby(['layout_id','scenario_id'])[col].transform('mean')
        ly_mean = df2.groupby('layout_id')[col].transform('mean')
        ly_std  = df2.groupby('layout_id')[col].transform('std').fillna(1).clip(lower=0.01)
        all_cols.append(f'rel_sc_{col}'); all_data.append(((sc_mean-ly_mean)/ly_std).fillna(0).clip(-5,5).values.astype(np.float32))
    ly_sc_count = df2.groupby('layout_id')['scenario_id'].transform('nunique')
    all_cols.append('ly_scenario_count'); all_data.append(ly_sc_count.values.astype(np.float32))
    if 'pack_utilization' in df2.columns and 'congestion_score' in df2.columns:
        ly_p = df2.groupby('layout_id')['pack_utilization'].transform('mean')
        ly_c = df2.groupby('layout_id')['congestion_score'].transform('mean')
        all_cols.append('ly_pack_congestion_ratio'); all_data.append((ly_p/(ly_c+0.01)).clip(0,10).values.astype(np.float32))
    else:
        all_cols.append('ly_pack_congestion_ratio'); all_data.append(np.zeros(len(df2), np.float32))
    return np.column_stack(all_data), all_cols

print('[1] Layout stats...')
ly_tr, ly_feat_names = compute_layout_stats(train_raw)
ly_te, _ = compute_layout_stats(test_raw)
print(f'  Layout features: {len(ly_feat_names)}')

# Set monotone constraints on key load features
MONOTONE_INCREASING = {'ly_order_inflow_15m_mean', 'ly_congestion_score_mean', 'ly_robot_active_mean',
                        'ly_fault_count_15m_mean', 'ly_blocked_path_15m_mean'}

# mega33 proxy
print('[2] mega33 proxy...')
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

train_raw['mega_oof'] = mega_oof; test_raw['mega_test'] = mega_test
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

def add_oracle_feats(df, pred_col):
    grp = df.groupby(['layout_id','scenario_id'])
    df = df.copy()
    df['lag1']    = grp[pred_col].shift(1).fillna(df[pred_col].mean())
    df['lag2']    = grp[pred_col].shift(2).fillna(df[pred_col].mean())
    df['sc_mean'] = grp[pred_col].transform('mean')
    return np.column_stack([df['lag1'].values, df['lag2'].values, df['sc_mean'].values,
                             df['row_in_sc'].values.astype(np.float32)/25.0]).astype(np.float32)

of_tr = add_oracle_feats(train_raw, 'mega_oof')
of_te = add_oracle_feats(test_raw, 'mega_test')

X_tr_full = np.hstack([X_tr_base, ly_tr, of_tr])
X_te_full = np.hstack([X_te_base, ly_te, of_te])

# Build monotone constraints array: +1 for load features, 0 for others
n_v30 = X_tr_base.shape[1]
n_ly  = ly_tr.shape[1]
n_oc  = of_tr.shape[1]
mono_constraints = [0] * n_v30  # v30 features: no constraint
for fn in ly_feat_names:
    mono_constraints.append(1 if fn in MONOTONE_INCREASING else 0)
mono_constraints.extend([0] * n_oc)  # oracle lag: no constraint
print(f'  Monotone +1 features: {sum(mono_constraints)} out of {len(mono_constraints)}')
print(f'  Features: v30:{n_v30} + layout:{n_ly} (mono:{sum(1 for fn in ly_feat_names if fn in MONOTONE_INCREASING)}) + oracle:{n_oc}')

PARAMS = dict(
    objective='huber', alpha=0.9,
    n_estimators=2000, learning_rate=0.05,
    num_leaves=63, max_depth=8,
    min_child_samples=80,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    monotone_constraints=mono_constraints,
    random_state=42, verbose=-1, n_jobs=-1,
)

print('\n[3] 5-fold GroupKFold 학습...')
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
print(f'\nOverall OOF: {overall_mae:.5f}')

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
oracle_new_oof = 0.64*fixed_oof + 0.12*np.load('results/oracle_seq/oof_seqC_xgb.npy') \
               + 0.16*np.load('results/oracle_seq/oof_seqC_log_v2.npy') \
               + 0.08*np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'oracle_NEW OOF:     {mae(oracle_new_oof):.5f}')
print(f'layout_mono OOF:    {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, layout_mono): {corr:.4f}')
print(f'\nFold-level MAE comparison:')
for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    print(f'  Fold {fi+1}: layout_mono={mean_absolute_error(y_true[vs], oof[vs]):.5f}  oracle_NEW={mean_absolute_error(y_true[vs], oracle_new_oof[vs]):.5f}')
print(f'\nBlend with oracle_NEW:')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')
# Test comparison
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
oracle_test = np.clip(0.64*fixed_test + 0.12*np.load('results/oracle_seq/test_C_xgb.npy')
                    + 0.16*np.load('results/oracle_seq/test_C_log_v2.npy')
                    + 0.08*np.load('results/oracle_seq/test_C_xgb_remaining.npy'), 0, None)
print(f'\nTest pred: oracle_NEW={oracle_test.mean():.3f}, layout_mono={test_avg.mean():.3f} (delta={test_avg.mean()-oracle_test.mean():+.3f})')
print('Done.')
