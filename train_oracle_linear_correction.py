"""
Linear Residual Correction for oracle_NEW
Key insight: oracle_NEW under-predicts systematically (mean pred=15.7 vs true=18.96).
If this under-prediction INCREASES with load level (which domain shift predicts),
then a LINEAR correction using load features extrapolates to unseen high-load test layouts.

Why linear (not tree/isotonic):
- Linear extrapolates beyond training range (trees don't)
- Test load is +39% higher than training → OUTSIDE training tree leaves
- Linear correction: correction = beta_inflow * inflow + ... works at any load level
- Isotonic (previous attempt OOF=8.817) failed because it maps pred→true directly
  (step function that doesn't generalize across folds)
- This maps LOAD FEATURES → RESIDUAL (different and more generalizable)

Approach:
1. Compute oracle_NEW residuals per row: r_i = y_true_i - oracle_oof_i
2. Aggregate to scenario level: r_sc_mean = mean(r_i) per scenario
3. GroupKFold(5): for each fold, fit LinearRegression(r_sc_mean ~ load_features) on training layouts
4. Predict residual for validation layouts using their load features
5. Apply: final = oracle_oof + alpha * predicted_residual (per row = scenario mean correction)
6. Choose alpha that minimizes OOF MAE

Key question: do oracle_NEW residuals correlate with load features?
If yes AND correlation direction matches domain shift → linear correction generalizes.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error

t0 = time.time()
print('='*60)
print('Linear Residual Correction for oracle_NEW')
print('  residual ~ inflow + congestion + pack_util + robot_active')
print('  Linear extrapolation handles domain shift better than trees')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# --- 1. oracle_NEW OOF and test ---
print('\n[1] oracle_NEW 재구성...')
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]

test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof  = (fw['mega33']*d33['meta_avg_oof'][id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
xgb_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_o_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o_test = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o_test = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_oof  = np.clip(0.64*fixed_oof  + 0.12*xgb_o_oof  + 0.16*lv2_o_oof  + 0.08*rem_o_oof,  0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)
del d33; gc.collect()

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'  oracle_NEW OOF: {mae(oracle_oof):.5f}')

# --- 2. Scenario-level features ---
print('\n[2] 시나리오 수준 피처 계산...')
train_raw['residual'] = y_true - oracle_oof

# Scenario-level aggregates for training
tr_sc = train_raw.groupby(['layout_id','scenario_id']).agg(
    r_mean=('residual', 'mean'),
    r_std=('residual', 'std'),
    inflow_mean=('order_inflow_15m', 'mean'),
    inflow_max=('order_inflow_15m', 'max'),
    congestion_mean=('congestion_score', 'mean'),
    congestion_max=('congestion_score', 'max'),
    pack_mean=('pack_utilization', 'mean'),
    pack_max=('pack_utilization', 'max'),
    robot_mean=('robot_active', 'mean'),
    fault_sum=('fault_count_15m', 'sum'),
    oracle_mean=('residual', 'count'),  # count as proxy for scenario size
).reset_index()
tr_sc.rename(columns={'oracle_mean': 'sc_size'}, inplace=True)

# True oracle pred mean per scenario
tr_oracle_sc = train_raw.copy()
tr_oracle_sc['oracle_pred'] = oracle_oof
oracle_sc_mean = tr_oracle_sc.groupby(['layout_id','scenario_id'])['oracle_pred'].mean()
tr_sc = tr_sc.merge(oracle_sc_mean.rename('oracle_pred_mean'), on=['layout_id','scenario_id'])

# Same for test
te_sc = test_raw.groupby(['layout_id','scenario_id']).agg(
    inflow_mean=('order_inflow_15m', 'mean'),
    inflow_max=('order_inflow_15m', 'max'),
    congestion_mean=('congestion_score', 'mean'),
    congestion_max=('congestion_score', 'max'),
    pack_mean=('pack_utilization', 'mean'),
    pack_max=('pack_utilization', 'max'),
    robot_mean=('robot_active', 'mean'),
    fault_sum=('fault_count_15m', 'sum'),
).reset_index()

te_oracle_sc = test_raw.copy()
te_oracle_sc['oracle_pred'] = oracle_test
oracle_sc_mean_te = te_oracle_sc.groupby(['layout_id','scenario_id'])['oracle_pred'].mean()
te_sc = te_sc.merge(oracle_sc_mean_te.rename('oracle_pred_mean'), on=['layout_id','scenario_id'])

print(f'  Training scenarios: {len(tr_sc)}, Test scenarios: {len(te_sc)}')

# --- 3. Residual analysis ---
print('\n[3] Residual correlation analysis...')
feat_names = ['inflow_mean', 'inflow_max', 'congestion_mean', 'congestion_max',
              'pack_mean', 'pack_max', 'robot_mean', 'fault_sum', 'oracle_pred_mean']

print(f'  Correlation with r_mean (oracle_NEW scenario residuals):')
for fn in feat_names:
    if fn in tr_sc.columns:
        c = np.corrcoef(tr_sc[fn].fillna(0), tr_sc['r_mean'])[0, 1]
        print(f'    {fn}: {c:+.4f}')

print(f'\n  Residual stats:')
print(f'    r_mean overall: {tr_sc["r_mean"].mean():.3f} (should be positive → under-prediction)')
print(f'    r_std: {tr_sc["r_mean"].std():.3f}')

# Domain shift: train vs test load
print(f'\n  Load domain shift (scenario level):')
for fn in ['inflow_mean', 'congestion_mean', 'pack_mean', 'robot_mean']:
    if fn in tr_sc.columns and fn in te_sc.columns:
        tr_m = tr_sc[fn].mean()
        te_m = te_sc[fn].mean()
        print(f'    {fn}: train={tr_m:.3f}, test={te_m:.3f}, shift={100*(te_m-tr_m)/(tr_m+1e-6):.1f}%')

# --- 4. Linear correction via GroupKFold ---
print('\n[4] GroupKFold linear residual correction...')
LOAD_FEATS = ['inflow_mean', 'congestion_mean', 'pack_mean', 'robot_mean', 'fault_sum']
# Normalize features
load_mean = tr_sc[LOAD_FEATS].mean()
load_std  = tr_sc[LOAD_FEATS].std().replace(0, 1)

tr_sc_norm = (tr_sc[LOAD_FEATS] - load_mean) / load_std
te_sc_norm = (te_sc[LOAD_FEATS] - load_mean) / load_std

# Build lookup: (layout_id, scenario_id) → row index in tr_sc
tr_sc['_sc_idx'] = np.arange(len(tr_sc))
sc_idx_map = tr_sc.set_index(['layout_id','scenario_id'])['_sc_idx'].to_dict()
tr_row_sc_idx = train_raw.apply(
    lambda r: sc_idx_map.get((r['layout_id'], r['scenario_id']), -1), axis=1
).values

te_sc['_sc_idx'] = np.arange(len(te_sc))
te_sc_idx_map = te_sc.set_index(['layout_id','scenario_id'])['_sc_idx'].to_dict()
te_row_sc_idx = test_raw.apply(
    lambda r: te_sc_idx_map.get((r['layout_id'], r['scenario_id']), -1), axis=1
).values

print(f'  Sanity: tr_row_sc_idx range [{tr_row_sc_idx.min()}, {tr_row_sc_idx.max()}]')

groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)

# Scenario-level layout groups
tr_sc_layout = tr_sc['layout_id'].values

correction_oof = np.zeros(len(train_raw))
correction_test_list = []
betas_list = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    val_idx_sorted = np.sort(val_idx)

    # Which scenarios are in training vs validation for this fold
    tr_layouts  = set(train_raw['layout_id'].values[tr_idx])
    val_layouts = set(train_raw['layout_id'].values[val_idx])

    tr_sc_mask  = tr_sc['layout_id'].isin(tr_layouts).values
    val_sc_mask = tr_sc['layout_id'].isin(val_layouts).values

    X_tr_sc = tr_sc_norm.values[tr_sc_mask]
    y_tr_sc = tr_sc['r_mean'].values[tr_sc_mask]
    X_val_sc = tr_sc_norm.values[val_sc_mask]

    # Ridge regression (scenarios as samples, ~200 training scenarios per fold)
    reg = Ridge(alpha=1.0, fit_intercept=True)
    reg.fit(X_tr_sc, y_tr_sc)
    betas_list.append((fold_i, reg.coef_, reg.intercept_))

    # Predict residual correction for validation scenarios
    val_sc_pred = reg.predict(X_val_sc)

    # Map from scenario-level back to row-level
    # val_sc indices in tr_sc that correspond to val layouts
    val_sc_indices = np.where(val_sc_mask)[0]
    sc_correction_map = dict(zip(val_sc_indices, val_sc_pred))

    for i, sc_idx in enumerate(tr_row_sc_idx[val_idx_sorted]):
        if sc_idx in sc_correction_map:
            correction_oof[val_idx_sorted[i]] = sc_correction_map[sc_idx]

    # Test correction (fit on all training scenarios)
    reg_full = Ridge(alpha=1.0, fit_intercept=True)
    reg_full.fit(tr_sc_norm.values, tr_sc['r_mean'].values)
    te_correction = reg_full.predict(te_sc_norm.values)
    correction_test_list.append(te_correction)

    fold_pred_oracle = oracle_oof[val_idx_sorted]
    fold_correction  = correction_oof[val_idx_sorted]
    fold_final       = np.clip(fold_pred_oracle + fold_correction, 0, None)
    fold_mae_oracle  = mean_absolute_error(y_true[val_idx_sorted], fold_pred_oracle)
    fold_mae_final   = mean_absolute_error(y_true[val_idx_sorted], fold_final)
    print(f'  Fold {fold_i+1}: oracle={fold_mae_oracle:.5f}, corrected={fold_mae_final:.5f}, delta={fold_mae_final-fold_mae_oracle:+.5f}')
    print(f'           beta: ' + ' '.join([f'{fn}={b:.4f}' for fn, b in zip(LOAD_FEATS, reg.coef_)]) + f' intercept={reg.intercept_:.4f}')

correction_test_avg = np.mean(correction_test_list, axis=0)

print(f'\n  Correction stats:')
print(f'    Train correction mean: {correction_oof.mean():.3f}')
print(f'    Test correction mean:  {correction_test_avg.mean():.3f}')
print(f'    Test > Train: {correction_test_avg.mean() > correction_oof.mean()} (should be True for domain shift correction)')

# Apply correction with different alpha
print(f'\n[5] Alpha sweep:')
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    pred = np.clip(oracle_oof + alpha * correction_oof, 0, None)
    delta = mae(pred) - mae(oracle_oof)
    print(f'  alpha={alpha:.1f}: OOF delta={delta:+.5f}, pred_mean={pred.mean():.3f}')

# Find best alpha
best_alpha = None
best_delta = 0.0
for alpha in np.arange(0.05, 1.05, 0.05):
    pred = np.clip(oracle_oof + alpha * correction_oof, 0, None)
    delta = mae(pred) - mae(oracle_oof)
    if delta < best_delta:
        best_delta = delta
        best_alpha = alpha

if best_alpha is not None:
    print(f'\n★ Best alpha={best_alpha:.2f}, OOF delta={best_delta:+.5f}')
    # Test correction
    te_sc_correction_row = np.array([correction_test_avg[te_row_sc_idx[i]]
                                      for i in range(len(test_raw))])
    test_final = np.clip(oracle_test + best_alpha * te_sc_correction_row, 0, None)
    print(f'  oracle_NEW test mean: {oracle_test.mean():.3f}')
    print(f'  corrected test mean: {test_final.mean():.3f} (delta={test_final.mean()-oracle_test.mean():+.3f})')

    # Save
    oof_final = np.clip(oracle_oof + best_alpha * correction_oof, 0, None)
    np.save('results/oracle_seq/oof_linCorrect.npy', oof_final)
    np.save('results/oracle_seq/test_linCorrect.npy', test_final)
    print(f'  Saved OOF and test predictions.')

    # Generate submission if OOF improved
    sub = pd.read_csv('sample_submission.csv')
    sub['predicted'] = test_final
    sub_file = f'submission_linCorrect_a{int(best_alpha*100)}_OOF{mae(oof_final):.5f}.csv'
    sub.to_csv(sub_file, index=False)
    print(f'  Submission: {sub_file}')
else:
    print(f'\nNo alpha improved OOF. Linear correction does not generalize.')
    # Still print what happens at alpha=0.3 to understand direction
    pred03 = np.clip(oracle_oof + 0.3 * correction_oof, 0, None)
    te_sc_correction_row = np.array([correction_test_avg[te_row_sc_idx[i]]
                                      for i in range(len(test_raw))])
    test03 = np.clip(oracle_test + 0.3 * te_sc_correction_row, 0, None)
    print(f'  At alpha=0.3: OOF delta={mae(pred03)-mae(oracle_oof):+.5f}, test pred_mean={test03.mean():.3f}')

print(f'\nDone. ({time.time()-t0:.0f}s)')
