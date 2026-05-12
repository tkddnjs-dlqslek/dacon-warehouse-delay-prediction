"""
Meta-Stacking Oracle — oracle_NEW OOF as Primary Feature
Key structural insight: all new oracle attempts (global_rank, trajectory, KNN, extended_lead v1/v2)
give fold2 MAE ≈ 9.80 vs oracle_NEW's 9.595.

Root cause: oracle_NEW uses mega33 DIRECTLY (0.64 weight, near in-sample for fold2).
New oracles use mega33 INDIRECTLY (lag1/lag2/sc_mean inputs) — this dilutes fold2's signal.

Solution: use oracle_NEW OOF predictions as PRIMARY input feature.
- oracle_NEW OOF is valid (computed out-of-fold for each row)
- Meta-model mainly learns f(oracle_new) ≈ identity + small corrections
- Load features (order_inflow, pack_util) provide domain-shift correction
- Very strong regularization → near-linear, low overfit risk

Expected behavior:
- fold2 MAE stays ≈ 9.595 (oracle_new_oof carries the fold2 signal)
- Load-feature corrections push test predictions higher for unseen high-load layouts
- OOF stays close to oracle_NEW (≈8.38) rather than degrading to 8.57+
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqD_meta_stacking.npy'
OUT_TEST = 'results/oracle_seq/test_D_meta_stacking.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); sys.exit(0)

t0 = time.time()
print('='*60)
print('Meta-Stacking Oracle — oracle_NEW as Primary Feature')
print('  oracle_NEW OOF → direct input (preserves fold2 signal)')
print('  Strong regularization → near-linear correction only')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# --- 1. oracle_NEW OOF and test reconstruction ---
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
print(f'  oracle_NEW pred_mean: {oracle_oof.mean():.3f}, true_mean: {y_true.mean():.3f}')

# Check oracle_new fold-level MAE (reference)
print(f'\n  oracle_NEW fold-level MAE:')
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    val_idx_sorted = np.sort(val_idx)
    fm = mean_absolute_error(y_true[val_idx_sorted], oracle_oof[val_idx_sorted])
    print(f'    Fold {fi+1}: {fm:.5f}')

# --- 2. v30 base features ---
print('\n[2] v30 feature 로드...')
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

# --- 3. Scenario-level load summary features ---
print('\n[3] Scenario-level load summary features...')

def compute_scenario_load_feats(df):
    """Simple per-scenario load aggregates for domain-shift correction."""
    df2 = df.copy()
    grp = df2.groupby(['layout_id','scenario_id'])

    # Key load features for domain shift correction
    feats = {}
    for col in ['order_inflow_15m', 'congestion_score', 'pack_utilization',
                'robot_active', 'robot_idle', 'fault_count_15m',
                'outbound_truck_wait_min', 'charge_queue_length']:
        if col in df2.columns:
            feats[f'sc_{col}_mean'] = grp[col].transform('mean')
            feats[f'sc_{col}_max']  = grp[col].transform('max')
        else:
            feats[f'sc_{col}_mean'] = 0.0
            feats[f'sc_{col}_max']  = 0.0

    # Load-pack interaction: high inflow + high pack_util = packing bottleneck
    if 'order_inflow_15m' in df2.columns and 'pack_utilization' in df2.columns:
        feats['sc_load_pack_interact'] = (
            grp['order_inflow_15m'].transform('mean') *
            grp['pack_utilization'].transform('mean')
        )
    else:
        feats['sc_load_pack_interact'] = 0.0

    # Efficiency vs load ratio (low congestion + high pack = fold 2 pattern)
    if 'congestion_score' in df2.columns and 'pack_utilization' in df2.columns:
        eps = 1e-6
        feats['sc_pack_congestion_ratio'] = (
            grp['pack_utilization'].transform('mean') /
            (grp['congestion_score'].transform('mean') + eps)
        )
    else:
        feats['sc_pack_congestion_ratio'] = 0.0

    feat_names = list(feats.keys())
    result = np.column_stack([pd.to_numeric(v, errors='coerce').fillna(0).values
                               for v in feats.values()]).astype(np.float32)
    return result, feat_names

load_tr, load_feat_names = compute_scenario_load_feats(train_raw)
load_te, _ = compute_scenario_load_feats(test_raw)
print(f'  Load features: {len(load_feat_names)}')

# Check domain shift for key features
print(f'\n  Domain shift (load features):')
for i, fn in enumerate(load_feat_names[:10]):
    tr_m = np.nanmean(load_tr[:, i])
    te_m = np.nanmean(load_te[:, i])
    if abs(te_m - tr_m) > 0.01 * abs(tr_m + 1e-6):
        print(f'    {fn}: train={tr_m:.3f}, test={te_m:.3f}, delta={te_m-tr_m:+.3f} ({(te_m-tr_m)/(abs(tr_m)+1e-6)*100:+.1f}%)')

# --- 4. Assemble meta-features ---
# Primary feature: oracle_new_oof (DIRECT — preserves fold2 signal)
# Secondary: v30 base (top features from mega33 importance, but include all for now)
# Tertiary: scenario-level load aggregates
print('\n[4] Meta-feature assembly...')

# Include oracle_new_oof as the FIRST and most important feature
# Also row_in_sc (position in scenario)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

oracle_oof_col_tr = oracle_oof.reshape(-1, 1).astype(np.float32)
oracle_oof_col_te = oracle_test.reshape(-1, 1).astype(np.float32)

row_pos_tr = (train_raw['row_in_sc'].values / 25.0).reshape(-1, 1).astype(np.float32)
row_pos_te = (test_raw['row_in_sc'].values / 25.0).reshape(-1, 1).astype(np.float32)

X_tr_meta = np.hstack([oracle_oof_col_tr, row_pos_tr, load_tr, X_tr_base])
X_te_meta = np.hstack([oracle_oof_col_te, row_pos_te, load_te, X_te_base])
print(f'  X_tr_meta: {X_tr_meta.shape} (oracle:1 + pos:1 + load:{len(load_feat_names)} + v30:{X_tr_base.shape[1]})')

meta_feat_names = ['oracle_new_oof', 'row_pos'] + load_feat_names + feat_cols

# --- 5. Two-variant training ---
# Variant A: ultra-strong reg (num_leaves=15) → near-linear
# Variant B: moderate reg (num_leaves=31) → can learn mild nonlinear corrections

print('\n[5] 5-fold GroupKFold 학습 (Variant A: ultra-strong reg)...')
PARAMS_A = dict(
    objective='huber', alpha=0.9,
    n_estimators=3000, learning_rate=0.03,
    num_leaves=15, max_depth=5,
    min_child_samples=500,
    subsample=0.8, colsample_bytree=0.5,
    reg_alpha=5.0, reg_lambda=5.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

gkf = GroupKFold(n_splits=5)
oof_A = np.zeros(len(train_raw))
test_preds_A = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    model = lgb.LGBMRegressor(**PARAMS_A)
    model.fit(X_tr_meta[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr_meta[val_idx_sorted], y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    fold_pred = np.clip(model.predict(X_tr_meta[val_idx_sorted]), 0, None)
    oof_A[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds_A.append(np.clip(model.predict(X_te_meta), 0, None))
    print(f'  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model; gc.collect()

oofA_mae = mean_absolute_error(y_true, oof_A)
test_A = np.mean(test_preds_A, axis=0)
print(f'\nVariant A OOF: {oofA_mae:.5f}  (oracle_NEW: {mae(oracle_oof):.5f})')
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    fm = mean_absolute_error(y_true[np.sort(val_idx)], oof_A[np.sort(val_idx)])
    print(f'  Fold {fi+1}: {fm:.5f}')

print('\n[6] 5-fold GroupKFold 학습 (Variant B: moderate reg)...')
PARAMS_B = dict(
    objective='huber', alpha=0.9,
    n_estimators=2000, learning_rate=0.05,
    num_leaves=31, max_depth=6,
    min_child_samples=200,
    subsample=0.7, colsample_bytree=0.6,
    reg_alpha=2.0, reg_lambda=2.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

oof_B = np.zeros(len(train_raw))
test_preds_B = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    model = lgb.LGBMRegressor(**PARAMS_B)
    model.fit(X_tr_meta[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr_meta[val_idx_sorted], y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    fold_pred = np.clip(model.predict(X_tr_meta[val_idx_sorted]), 0, None)
    oof_B[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds_B.append(np.clip(model.predict(X_te_meta), 0, None))
    print(f'  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model; gc.collect()

oofB_mae = mean_absolute_error(y_true, oof_B)
test_B = np.mean(test_preds_B, axis=0)
print(f'\nVariant B OOF: {oofB_mae:.5f}  (oracle_NEW: {mae(oracle_oof):.5f})')
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    fm = mean_absolute_error(y_true[np.sort(val_idx)], oof_B[np.sort(val_idx)])
    print(f'  Fold {fi+1}: {fm:.5f}')

# --- 7. Choose best variant and blends ---
print('\n[7] 최종 분석...')
best_oof_single = oof_A if oofA_mae < oofB_mae else oof_B
best_test_single = test_A if oofA_mae < oofB_mae else test_B
best_label = 'A' if oofA_mae < oofB_mae else 'B'
print(f'  Best variant: {best_label} (OOF={min(oofA_mae, oofB_mae):.5f})')

print(f'\n  Correlation with oracle_NEW:')
print(f'    corr(oracle_NEW, varA): {np.corrcoef(oracle_oof, oof_A)[0,1]:.4f}')
print(f'    corr(oracle_NEW, varB): {np.corrcoef(oracle_oof, oof_B)[0,1]:.4f}')

# Blend analysis with oracle_NEW
print(f'\n  Blend with oracle_NEW:')
for w in [0.1, 0.2, 0.3, 0.4, 0.5]:
    b_a = (1-w)*oracle_oof + w*oof_A
    b_b = (1-w)*oracle_oof + w*oof_B
    print(f'  w={w}: blend_A={mae(b_a):+.5f} delta={mae(b_a)-mae(oracle_oof):+.5f} | '
          f'blend_B={mae(b_b):+.5f} delta={mae(b_b)-mae(oracle_oof):+.5f}')

# Blend of A+B
print(f'\n  Blend A+B:')
for wa in [0.3, 0.5, 0.7]:
    b_ab = wa*oof_A + (1-wa)*oof_B
    print(f'  A={wa:.1f} B={1-wa:.1f}: OOF={mae(b_ab):.5f}')

# Save best variant
np.save(OUT_OOF,  best_oof_single)
np.save(OUT_TEST, best_test_single)
print(f'\nSaved: {OUT_OOF} (variant {best_label})')
print(f'       {OUT_TEST}')

# Test prediction analysis
print(f'\n  Test pred analysis:')
print(f'    oracle_NEW test mean: {oracle_test.mean():.3f}')
print(f'    varA test mean:       {test_A.mean():.3f} (delta={test_A.mean()-oracle_test.mean():+.3f})')
print(f'    varB test mean:       {test_B.mean():.3f} (delta={test_B.mean()-oracle_test.mean():+.3f})')

print(f'\nTotal time: {time.time()-t0:.0f}s')
print('Done.')
