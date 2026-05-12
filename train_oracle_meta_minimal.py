"""
Meta-Minimal Oracle — oracle_NEW OOF as ONLY primary feature.
Problem with previous meta-stacking: v30's 149 features dominate oracle_new_oof,
so the model effectively ignores oracle_new_oof and re-learns from v30 scratch
→ fold1=9.015 (WORSE than oracle_NEW's 8.810).

Fix: Remove ALL v30 features. Use ONLY:
1. oracle_new_oof  (primary, monotone_increasing=1)
2. 5 scenario-level load aggregates (for domain-shift correction)
3. row_position (step 0-25 within scenario)

With only 8 features and oracle_new_oof as the ONLY non-load feature,
the model MUST use oracle_new_oof as its primary split.
Monotone constraint ensures f(oracle_new) is monotone increasing.

If this works: fold-level MAEs ≈ oracle_NEW's fold-level MAEs (≈pass-through)
Gain: load features might push test predictions higher for high-load scenarios.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqD_meta_minimal.npy'
OUT_TEST = 'results/oracle_seq/test_D_meta_minimal.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Meta-Minimal Oracle — oracle_NEW + 5 load features ONLY')
print('  No v30 features: forces oracle_new_oof as primary predictor')
print('  Monotone constraint on oracle_new_oof')
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

# --- 2. Minimal feature set ---
print('\n[2] Minimal feature set (oracle_new + 5 scenario load + row_pos)...')
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

def build_minimal_feats(df, oracle_pred):
    """8 features only: oracle_pred + 5 load + row_pos + scenario_size."""
    grp = df.groupby(['layout_id','scenario_id'])

    # Scenario-level load (for domain-shift correction)
    sc_inflow_mean  = grp['order_inflow_15m'].transform('mean').values if 'order_inflow_15m' in df else np.zeros(len(df))
    sc_congestion   = grp['congestion_score'].transform('mean').values if 'congestion_score' in df else np.zeros(len(df))
    sc_pack_max     = grp['pack_utilization'].transform('max').values  if 'pack_utilization' in df else np.zeros(len(df))
    sc_robot_active = grp['robot_active'].transform('mean').values     if 'robot_active' in df else np.zeros(len(df))
    sc_fault        = grp['fault_count_15m'].transform('sum').values   if 'fault_count_15m' in df else np.zeros(len(df))

    row_pos = (df['row_in_sc'].values / 25.0).astype(np.float32)

    X = np.column_stack([
        oracle_pred.astype(np.float32),   # 0: oracle_new_oof (PRIMARY, monotone)
        sc_inflow_mean.astype(np.float32), # 1: load
        sc_congestion.astype(np.float32),  # 2: congestion
        sc_pack_max.astype(np.float32),    # 3: packing
        sc_robot_active.astype(np.float32),# 4: robot
        sc_fault.astype(np.float32),       # 5: faults
        row_pos,                           # 6: position
    ])
    return X

X_tr = build_minimal_feats(train_raw, oracle_oof)
X_te = build_minimal_feats(test_raw,  oracle_test)
print(f'  X_tr: {X_tr.shape}')

# Check domain shift
feat_names = ['oracle_new', 'sc_inflow_mean', 'sc_congestion', 'sc_pack_max',
              'sc_robot_active', 'sc_fault', 'row_pos']
print(f'\n  Domain shift:')
for i, fn in enumerate(feat_names):
    tr_m = X_tr[:, i].mean()
    te_m = X_te[:, i].mean()
    shift = (te_m - tr_m) / (abs(tr_m) + 1e-6) * 100
    if abs(shift) > 5:
        print(f'    {fn}: train={tr_m:.3f} test={te_m:.3f} shift={shift:+.1f}%')

# --- 3. Training with monotone constraint on oracle_new_oof ---
# Feature 0 (oracle_new_oof) must be monotone increasing
print('\n[3] Training variants with monotone constraint on oracle_new_oof...')

groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)

# Variant A: very strong reg
PARAMS_A = dict(
    objective='huber', alpha=0.9,
    n_estimators=5000, learning_rate=0.02,
    num_leaves=31, max_depth=6,
    min_child_samples=300,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=2.0,
    monotone_constraints=[1, 0, 0, 0, 0, 0, 0],  # oracle_new monotone increasing
    random_state=42, verbose=-1, n_jobs=-1,
)

print(f'\n  Variant A (monotone + mod reg):')
oof_A = np.zeros(len(train_raw))
test_preds_A = []
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)
    model = lgb.LGBMRegressor(**PARAMS_A)
    model.fit(X_tr[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr[val_idx_sorted], y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
    fold_pred = np.clip(model.predict(X_tr[val_idx_sorted]), 0, None)
    oof_A[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds_A.append(np.clip(model.predict(X_te), 0, None))
    oracle_fold_mae = mean_absolute_error(y_true[val_idx_sorted], oracle_oof[val_idx_sorted])
    print(f'  Fold {fold_i+1}: meta={fold_mae:.5f} oracle={oracle_fold_mae:.5f} delta={fold_mae-oracle_fold_mae:+.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model; gc.collect()

oofA_mae = mae(oof_A)
test_A = np.mean(test_preds_A, axis=0)
print(f'  Variant A OOF: {oofA_mae:.5f}  (oracle_NEW: {mae(oracle_oof):.5f}, delta={oofA_mae - mae(oracle_oof):+.5f})')

# Variant B: ultra-light (num_leaves=15)
PARAMS_B = dict(
    objective='huber', alpha=0.9,
    n_estimators=5000, learning_rate=0.02,
    num_leaves=15, max_depth=4,
    min_child_samples=500,
    subsample=0.9, colsample_bytree=0.9,
    reg_alpha=5.0, reg_lambda=5.0,
    monotone_constraints=[1, 0, 0, 0, 0, 0, 0],
    random_state=42, verbose=-1, n_jobs=-1,
)

print(f'\n  Variant B (monotone + heavy reg):')
oof_B = np.zeros(len(train_raw))
test_preds_B = []
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)
    model = lgb.LGBMRegressor(**PARAMS_B)
    model.fit(X_tr[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr[val_idx_sorted], y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(300, verbose=False), lgb.log_evaluation(0)])
    fold_pred = np.clip(model.predict(X_tr[val_idx_sorted]), 0, None)
    oof_B[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds_B.append(np.clip(model.predict(X_te), 0, None))
    oracle_fold_mae = mean_absolute_error(y_true[val_idx_sorted], oracle_oof[val_idx_sorted])
    print(f'  Fold {fold_i+1}: meta={fold_mae:.5f} oracle={oracle_fold_mae:.5f} delta={fold_mae-oracle_fold_mae:+.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model; gc.collect()

oofB_mae = mae(oof_B)
test_B = np.mean(test_preds_B, axis=0)
print(f'  Variant B OOF: {oofB_mae:.5f}  (oracle_NEW: {mae(oracle_oof):.5f}, delta={oofB_mae - mae(oracle_oof):+.5f})')

# --- 4. Analysis ---
print('\n[4] 비교 분석...')
print(f'\n  Variant A vs oracle_NEW:')
print(f'    OOF: {oofA_mae:.5f} vs {mae(oracle_oof):.5f}')
print(f'    corr(oracle_new, meta_A): {np.corrcoef(oracle_oof, oof_A)[0,1]:.4f}')
for w in [0.1, 0.2, 0.3]:
    b = (1-w)*oracle_oof + w*oof_A
    print(f'    blend w={w}: delta={mae(b)-mae(oracle_oof):+.5f}')

print(f'\n  Test prediction shift:')
print(f'    oracle_NEW test mean: {oracle_test.mean():.3f}')
print(f'    meta_A test mean:     {test_A.mean():.3f} (delta={test_A.mean()-oracle_test.mean():+.3f})')
print(f'    meta_B test mean:     {test_B.mean():.3f} (delta={test_B.mean()-oracle_test.mean():+.3f})')

# Save best
best_oof  = oof_A  if oofA_mae < oofB_mae else oof_B
best_test = test_A if oofA_mae < oofB_mae else test_B
best_label = 'A' if oofA_mae < oofB_mae else 'B'
np.save(OUT_OOF,  best_oof)
np.save(OUT_TEST, best_test)
print(f'\nSaved: {OUT_OOF} (variant {best_label})')
print(f'Total: {time.time()-t0:.0f}s')
print('Done.')
