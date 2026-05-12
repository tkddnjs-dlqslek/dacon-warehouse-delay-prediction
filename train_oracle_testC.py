"""
Generate oracle-C test predictions (oracle model + mega33 as lag at all positions)
Then blend with FIXED submission and check OOF improvement.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from scipy.optimize import minimize

print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

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

# True lag features
train_raw['lag1_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(1).fillna(global_mean))
train_raw['lag2_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(2).fillna(global_mean))

# Mega33 lags (for test and pseudo-lag eval)
test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega'] = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(1).fillna(global_mean))
test_raw['lag2_mega'] = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(2).fillna(global_mean))

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
row_sc_arr   = train_raw['row_in_sc'].values
mega_oof_arr = mega_oof_id

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base   = X_test_base[feat_cols].values
test_lag1_mega = test_raw['lag1_mega'].values
test_lag2_mega = test_raw['lag2_mega'].values
test_row_sc    = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_C    = np.load('results/oracle_seq/oof_seqC.npy')   # already computed
test_C_list = []

print("Training oracle model for test-C predictions...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx],
                  train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]
    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx],
                        train_lag2[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_true, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(9999)])

    # Test-C: use mega33_test shifted as lag (same as pseudo-lag but oracle model)
    X_test_C = make_X(X_test_base, test_lag1_mega, test_lag2_mega, test_row_sc)
    test_C_list.append(np.maximum(0, model.predict(X_test_C)))

    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: {elapsed:.0f}s", flush=True)

test_C_avg = np.mean(test_C_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/test_C.npy', test_C_avg)
print(f"test_C saved: shape={test_C_avg.shape}, mean={test_C_avg.mean():.4f}", flush=True)

# ────────────────────────────
# Blend analysis: oracle-C vs FIXED
# ────────────────────────────
print("\n=== BLEND ANALYSIS ===", flush=True)
mega_mae = np.mean(np.abs(mega_oof_id - y_true))

# Load FIXED components OOF
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')

# Need to align iter OOFs (they might be in layout,scenario order)
# Check shape
print(f"rank_oof: {rank_oof.shape}, iter1: {iter1_oof.shape}", flush=True)

# Reconstruct FIXED blend OOF
fw = {
    'mega33': 0.7636614598089654,
    'rank_adj': 0.1588758398901156,
    'iter_r1': 0.011855567572749024,
    'iter_r2': 0.03456830669223538,
    'iter_r3': 0.031038826035934514,
}

# Try to align iter OOFs — they might be in ID order or layout,scenario order
# Check rank_oof alignment
rank_mae_id = np.mean(np.abs(rank_oof - y_true))
print(f"rank_oof MAE (ID order check): {rank_mae_id:.4f}", flush=True)

# Build FIXED OOF
fixed_oof = (fw['mega33'] * mega_oof_id +
             fw['rank_adj'] * rank_oof +
             fw['iter_r1'] * iter1_oof +
             fw['iter_r2'] * iter2_oof +
             fw['iter_r3'] * iter3_oof)
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"FIXED OOF MAE (reconstructed): {fixed_mae:.4f}  (target: 8.3935)", flush=True)

# If FIXED OOF doesn't match, try layout,scenario order alignment
if abs(fixed_mae - 8.3935) > 0.005:
    print("Alignment mismatch — checking iter OOF order...", flush=True)
    # Check if iter OOFs are in layout,scenario order
    train_ls_df = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
    ls_order = [ls_to_pos[i] for i in train_raw['ID'].values]

    iter1_id = iter1_oof[ls_order]
    iter2_id = iter2_oof[ls_order]
    iter3_id = iter3_oof[ls_order]

    fixed_oof2 = (fw['mega33'] * mega_oof_id +
                  fw['rank_adj'] * rank_oof +
                  fw['iter_r1'] * iter1_id +
                  fw['iter_r2'] * iter2_id +
                  fw['iter_r3'] * iter3_id)
    fixed_mae2 = np.mean(np.abs(fixed_oof2 - y_true))
    print(f"FIXED OOF MAE (ls-order iter): {fixed_mae2:.4f}", flush=True)

    if abs(fixed_mae2 - 8.3935) < abs(fixed_mae - 8.3935):
        fixed_oof = fixed_oof2
        fixed_mae = fixed_mae2
        iter1_oof = iter1_id
        iter2_oof = iter2_id
        iter3_oof = iter3_id

print(f"\nFINAL FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

# Now optimize blend: oracle-C + FIXED components
# 5-way blend: mega33, rank, iter1, iter2, iter3, oracle-C
def oof_mae_from_weights(w, *oofs):
    blend = sum(wi * o for wi, o in zip(w, oofs))
    return np.mean(np.abs(blend - y_true))

# Grid search: oracle-C weight from 0 to 0.5
oofs = [mega_oof_id, rank_oof, iter1_oof, iter2_oof, iter3_oof, oof_C]
base_weights = [fw['mega33'], fw['rank_adj'], fw['iter_r1'], fw['iter_r2'], fw['iter_r3']]

best_mae_full = 9999
best_w_oracC = 0
for w_oracC in np.arange(0, 0.41, 0.02):
    # Scale existing weights proportionally
    remaining = 1 - w_oracC
    scaled = [bw * remaining for bw in base_weights]
    full_w = scaled + [w_oracC]
    mae = oof_mae_from_weights(full_w, *oofs)
    if mae < best_mae_full:
        best_mae_full = mae
        best_w_oracC = w_oracC

print(f"\nBest oracle-C weight added to FIXED: w={best_w_oracC:.2f}", flush=True)
print(f"Blend MAE: {best_mae_full:.4f}  (vs FIXED {fixed_mae:.4f}, delta={best_mae_full-fixed_mae:.4f})", flush=True)
print(f"vs mega33 alone: delta={best_mae_full-mega_mae:.4f}", flush=True)

# Also try replacing mega33 with oracle-C blend
for w_C in np.arange(0, 1.01, 0.05):
    w_C_combo = fw.copy()
    # blend: (1-w_C)*mega33 + w_C*oracle-C for the mega33 slot
    effective_mega = (1-w_C) * mega_oof_id + w_C * oof_C
    test_blend = (fw['rank_adj'] * rank_oof +
                  fw['iter_r1'] * iter1_oof +
                  fw['iter_r2'] * iter2_oof +
                  fw['iter_r3'] * iter3_oof +
                  (1 - fw['rank_adj'] - fw['iter_r1'] - fw['iter_r2'] - fw['iter_r3']) * effective_mega)
    # This doesn't add to 1... let me do it properly

# Proper 2-way blend: FIXED vs oracle-C
print("\n--- 2-way blend: FIXED_oof vs oracle-C ---", flush=True)
best_2way = 9999; best_w2 = 0
for w in np.arange(0, 0.51, 0.02):
    m = np.mean(np.abs(w*oof_C + (1-w)*fixed_oof - y_true))
    if m < best_2way:
        best_2way = m; best_w2 = w
print(f"Best w_oracleC={best_w2:.2f}: MAE={best_2way:.4f}  delta={best_2way-fixed_mae:.4f}", flush=True)

# Residual corr of oracle-C vs FIXED
corr_C_FIXED = np.corrcoef(oof_C - y_true, fixed_oof - y_true)[0,1]
print(f"residual_corr(oracle-C, FIXED): {corr_C_FIXED:.4f}", flush=True)

# Generate test submission if improvement is meaningful
if best_2way < fixed_mae - 0.001:
    print(f"\n→ OOF improvement {fixed_mae - best_2way:.4f} > 0.001, generating submission...", flush=True)

    # Load test components
    rank_test  = np.load('results/ranking/rank_adj_test.npy')
    iter1_test = np.load('results/iter_pseudo/round1_test.npy')
    iter2_test = np.load('results/iter_pseudo/round2_test.npy')
    iter3_test = np.load('results/iter_pseudo/round3_test.npy')

    # Check iter test alignment (same as OOF)
    test_ls_df = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
    te_ls_pos2 = {row['ID']:i for i,row in test_ls_df.iterrows()}
    te_order = [te_ls_pos2[i] for i in test_raw['ID'].values]

    # If iter was in ls order, align to ID order
    if abs(fixed_mae - 8.3935) > 0.001:  # Was using ls-order
        iter1_test = iter1_test[te_order]
        iter2_test = iter2_test[te_order]
        iter3_test = iter3_test[te_order]

    fixed_test = (fw['mega33'] * mega_test_id +
                  fw['rank_adj'] * rank_test +
                  fw['iter_r1'] * iter1_test +
                  fw['iter_r2'] * iter2_test +
                  fw['iter_r3'] * iter3_test)

    test_blend_final = np.maximum(0, best_w2 * test_C_avg + (1-best_w2) * fixed_test)

    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': test_blend_final})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    sub_df.to_csv('submission_oracle_C_blend.csv', index=False)
    print(f"Saved: submission_oracle_C_blend.csv", flush=True)
    print(f"OOF: {best_2way:.4f}  (vs FIXED {fixed_mae:.4f}  delta={best_2way-fixed_mae:.4f})", flush=True)
else:
    print(f"\n→ Insufficient improvement ({fixed_mae - best_2way:.4f} < 0.001), NOT generating submission.", flush=True)

with open('results/oracle_seq/blend_summary.json','w',encoding='utf-8') as f:
    json.dump({
        'oracle_C_mae': float(np.mean(np.abs(oof_C - y_true))),
        'oracle_C_corr_mega33': float(np.corrcoef(oof_C - y_true, mega_oof_id - y_true)[0,1]),
        'fixed_mae_reconstructed': float(fixed_mae),
        'best_2way_w': float(best_w2),
        'best_2way_mae': float(best_2way),
        'delta_vs_fixed': float(best_2way - fixed_mae),
        'corr_C_FIXED': float(corr_C_FIXED),
        'best_full_blend_mae': float(best_mae_full),
        'best_full_w_oracC': float(best_w_oracC),
    }, f, indent=2)
print("Done.", flush=True)
