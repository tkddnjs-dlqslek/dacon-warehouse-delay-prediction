"""
Scenario Trajectory Oracle
Key insight: v30 has static stats (mean/max/min) but NO dynamic/trend features.
A warehouse going from low to HIGH congestion in 30 min predicts more delay
than one stable at the same mean — current models can't distinguish these.

New features (per scenario, broadcast to all rows):
- sc_congestion_slope: linear trend of congestion over 25 steps
- sc_inflow_slope: linear trend of order_inflow over 25 steps
- sc_congestion_momentum: late_mean - early_mean (last8 - first8)
- sc_inflow_momentum: same for inflow
- sc_late_congestion: mean of last 8 rows (most recent, predictive of next 30min)
- sc_early_congestion: mean of first 8 rows (initial state)
- sc_late_inflow: mean of last 8 rows
- sc_early_inflow: mean of first 8 rows
- sc_high_stress_frac: fraction where inflow>p75 AND congestion>0.5 simultaneously
- sc_peak_inflow_frac: fraction where inflow > global 75th percentile
- sc_peak_congestion_frac: fraction where congestion > 0.5

For test unseen (64% busier): higher slopes, higher momentum, more peak rows
→ distinct signal from oracle_NEW's static aggregates
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqC_trajectory.npy'
OUT_TEST = 'results/oracle_seq/test_C_trajectory.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Scenario Trajectory Oracle')
print('  NEW: slope, momentum, late/early split, peak fraction')
print('  NOT in v30/v31 — captures dynamics within scenario')
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

# Compute scenario trajectory features
print('\n[2] Scenario Trajectory Features 계산...')

# Global thresholds from training data
p75_inflow = np.percentile(train_raw['order_inflow_15m'].values, 75)
p75_congestion = np.percentile(train_raw['congestion_score'].values, 75)
# Use 0.5 as congestion threshold (mid-range)
cong_threshold = 0.5
print(f'  p75_inflow: {p75_inflow:.3f}, p75_congestion: {p75_congestion:.3f}')
print(f'  congestion threshold: {cong_threshold}')

# Time steps within scenario (0..24)
x_steps = np.arange(25, dtype=np.float32)
# Precompute for linear regression slope: slope = cov(x, y) / var(x)
x_demean = x_steps - x_steps.mean()
x_var = (x_demean ** 2).sum()

def compute_trajectory_features(df):
    """Compute per-scenario trajectory features and broadcast to rows."""
    df = df.copy()
    df['row_in_sc'] = df.groupby(['layout_id','scenario_id']).cumcount()

    # Sort by scenario then position
    sc_key = ['layout_id', 'scenario_id']

    results = []
    sc_groups = df.groupby(sc_key)

    inflow_vals = df['order_inflow_15m'].values
    cong_vals   = df['congestion_score'].values

    # Fast approach: use sorted scenario groups
    n_rows = len(df)

    # Build scenario position index
    df_sorted = df.sort_values(sc_key + ['row_in_sc']).reset_index(drop=True)
    orig_idx = df_sorted.index.values  # after reset — not useful

    # We'll process per scenario group
    feat_names = [
        'sc_congestion_slope', 'sc_inflow_slope',
        'sc_congestion_momentum', 'sc_inflow_momentum',
        'sc_late_congestion', 'sc_early_congestion',
        'sc_late_inflow', 'sc_early_inflow',
        'sc_high_stress_frac', 'sc_peak_inflow_frac', 'sc_peak_congestion_frac',
    ]
    n_feats = len(feat_names)
    traj_feats = np.zeros((n_rows, n_feats), dtype=np.float32)

    row_ptr = 0
    for sc_id, grp in df.groupby(sc_key):
        n = len(grp)
        cong = grp['congestion_score'].values.astype(np.float32)
        inflow = grp['order_inflow_15m'].values.astype(np.float32)
        idx = grp.index.values

        # Slopes (linear regression over scenario steps)
        steps = np.arange(n, dtype=np.float32)
        steps_dm = steps - steps.mean()
        steps_var = (steps_dm ** 2).sum() + 1e-8

        cong_slope  = float(np.dot(steps_dm, cong - cong.mean()) / steps_var)
        inflow_slope = float(np.dot(steps_dm, inflow - inflow.mean()) / steps_var)

        # Momentum: late (last third) - early (first third)
        split = max(1, n // 3)
        early_cong  = cong[:split].mean()
        late_cong   = cong[-split:].mean()
        early_inflow = inflow[:split].mean()
        late_inflow  = inflow[-split:].mean()
        cong_momentum  = float(late_cong - early_cong)
        inflow_momentum = float(late_inflow - early_inflow)

        # Peak fractions
        high_stress_frac     = float(np.mean((inflow > p75_inflow) & (cong > cong_threshold)))
        peak_inflow_frac     = float(np.mean(inflow > p75_inflow))
        peak_congestion_frac = float(np.mean(cong > p75_congestion))

        row_feats = np.array([
            cong_slope, inflow_slope,
            cong_momentum, inflow_momentum,
            late_cong, early_cong,
            late_inflow, early_inflow,
            high_stress_frac, peak_inflow_frac, peak_congestion_frac,
        ], dtype=np.float32)

        # Broadcast to all rows in this scenario
        traj_feats[idx] = row_feats[np.newaxis, :]

    return traj_feats, feat_names

print('  Computing train trajectory features...')
t1 = time.time()
traj_tr, feat_names = compute_trajectory_features(train_raw)
print(f'  Done ({time.time()-t1:.0f}s). shape: {traj_tr.shape}')

print('  Computing test trajectory features...')
t1 = time.time()
traj_te, _ = compute_trajectory_features(test_raw)
print(f'  Done ({time.time()-t1:.0f}s). shape: {traj_te.shape}')

# Correlation analysis
print(f'\n  Feature correlations with target:')
for i, fn in enumerate(feat_names):
    valid = ~np.isnan(traj_tr[:, i])
    if valid.sum() > 100:
        corr = float(np.corrcoef(traj_tr[valid, i], y_true[valid])[0,1])
        print(f'    {fn}: {corr:+.4f}')

# Domain shift analysis
print(f'\n  Domain shift (train vs test):')
for i, fn in enumerate(feat_names):
    tr_m = np.nanmean(traj_tr[:, i])
    te_m = np.nanmean(traj_te[:, i])
    print(f'    {fn}: train={tr_m:.4f}, test={te_m:.4f}, delta={te_m-tr_m:+.4f}')

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

X_tr_full = np.hstack([X_tr_base, traj_tr, of_tr])
X_te_full = np.hstack([X_te_base, traj_te, of_te])
print(f'  X_tr_full: {X_tr_full.shape} (v30:{X_tr_base.shape[1]} + traj:{traj_tr.shape[1]} + oracle:4)')

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
    X_val = np.hstack([X_tr_base[val_idx_sorted], traj_tr[val_idx_sorted], of_val])

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
print(f'\noracle_NEW OOF: {mae(oracle_new_oof):.5f}')
print(f'trajectory OOF: {overall_mae:.5f}')
corr = float(np.corrcoef(oracle_new_oof, oof)[0,1])
print(f'corr(oracle_NEW, trajectory): {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof + w*oof
    print(f'  blend w={w}: delta={mae(b)-mae(oracle_new_oof):+.5f}')
print('Done.')
