"""
Residual meta-correction on the neural base blend.
Neural base OOF=8.37119 → can LGB learn residual corrections?
Key: conservative regularization, no te_id2 on oracle test (they're row-id ordered).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
test_raw['row_in_sc'] = test_raw.groupby(['layout_id','scenario_id']).cumcount()

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)
with open('results/mlp_deep_final.pkl','rb') as f: dm = pickle.load(f)
with open('results/mlp_deep_s2_final.pkl','rb') as f: ds2 = pickle.load(f)
with open('results/mlp_deep_gelu_final.pkl','rb') as f: dg = pickle.load(f)

# Neural base weights (proven)
w_base = {'mega37':0.4424,'rank_adj':0.0747,'oracle_log_v2':0.1540,
          'oracle_xgb_combined':0.1406,'oracle_xgb_v31':0.0589,
          'mlp_deep':0.0506,'mlp_s2':0.0539,'mlp_gelu':0.0198}
total = sum(w_base.values()); w_base = {k:v/total for k,v in w_base.items()}

# Build neural base predictions
base_oof_parts = {
    'mega37':  d37['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'mlp_deep': dm['oof'][id2], 'mlp_s2': ds2['oof'][id2], 'mlp_gelu': dg['oof'][id2],
    'oracle_xgb':          np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_xgb_rem':      np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
base_test_parts = {
    'mega37':  d37['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),      # row-id order, no te_id2
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'mlp_deep': dm['test'][te_id2], 'mlp_s2': ds2['test'][te_id2], 'mlp_gelu': dg['test'][te_id2],
    'oracle_xgb':          np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_xgb_rem':      np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

base_oof  = np.clip(sum(w_base.get(k,0)*base_oof_parts[k]  for k in base_oof_parts), 0, None)
base_test = np.clip(sum(w_base.get(k,0)*base_test_parts[k] for k in base_test_parts), 0, None)
base_mae  = np.mean(np.abs(base_oof - y_true))
print(f"Neural base OOF: {base_mae:.5f}")

# Also with cascade gate (current best single prediction)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]
mlp_deep_oof  = dm['oof'][id2]; mlp_deep_test = dm['test'][te_id2]

# Reproduce neural+scipy_gate best (p1=0.112, w1=0.0261, p2=0.265, w2=0.0384)
m1 = (clf_oof>0.112).astype(float); m1t = (clf_test>0.112).astype(float)
m2 = (clf_oof>0.265).astype(float); m2t = (clf_test>0.265).astype(float)
m3 = (clf_oof>0.45).astype(float);  m3t = (clf_test>0.45).astype(float)
b1 = (1-m1*0.0261)*base_oof + m1*0.0261*rh_oof
b2 = (1-m2*0.0384)*b1 + m2*0.0384*rm_oof
b3 = (1-m3*0.090)*b2 + m3*0.090*mlp_deep_oof  # triple gate
best_blend_oof  = b3
b1t = (1-m1t*0.0261)*base_test + m1t*0.0261*rh_test
b2t = (1-m2t*0.0384)*b1t + m2t*0.0384*rm_test
b3t = (1-m3t*0.090)*b2t + m3t*0.090*mlp_deep_test
best_blend_test = b3t
best_blend_mae  = np.mean(np.abs(best_blend_oof - y_true))
print(f"Neural+triple_gate OOF: {best_blend_mae:.5f}")

# Residual = y_true - best_blend
resid_tr = y_true - best_blend_oof
print(f"Residual stats: mean={resid_tr.mean():.4f}  std={resid_tr.std():.4f}")

# Meta features: component preds + position + differences + clf_prob
agreement_oof = np.std(np.stack([
    base_oof_parts['oracle_log_v2'], base_oof_parts['oracle_xgb_combined'],
    base_oof_parts['oracle_xgb_v31'], base_oof_parts['oracle_xgb'],
    base_oof_parts['oracle_xgb_rem'],
], axis=1), axis=1)

agreement_test = np.std(np.stack([
    base_test_parts['oracle_log_v2'], base_test_parts['oracle_xgb_combined'],
    base_test_parts['oracle_xgb_v31'], base_test_parts['oracle_xgb'],
    base_test_parts['oracle_xgb_rem'],
], axis=1), axis=1)

def build_meta(base_oof_vals, oracle_xgb, oracle_xgb_rem, oracle_log_v2,
               mlp_deep_vals, pos_arr, clf_prob, agree):
    return np.column_stack([
        base_oof_vals,
        oracle_xgb,
        oracle_xgb_rem,
        oracle_log_v2,
        mlp_deep_vals,
        pos_arr,
        pos_arr / 24.0,
        clf_prob,
        clf_prob ** 2,
        base_oof_vals - oracle_xgb,
        base_oof_vals - oracle_log_v2,
        mlp_deep_vals - base_oof_vals,
        agree,
        np.log1p(np.maximum(0, base_oof_vals)),
        np.log1p(np.maximum(0, oracle_xgb)),
        (base_oof_vals > 50).astype(float),
        (base_oof_vals > 80).astype(float),
    ])

X_tr = build_meta(
    best_blend_oof, base_oof_parts['oracle_xgb'],
    base_oof_parts['oracle_xgb_rem'], base_oof_parts['oracle_log_v2'],
    mlp_deep_oof, train_raw['row_in_sc'].values, clf_oof, agreement_oof
).astype(np.float32)

X_te = build_meta(
    best_blend_test, base_test_parts['oracle_xgb'],
    base_test_parts['oracle_xgb_rem'], base_test_parts['oracle_log_v2'],
    mlp_deep_test, test_raw['row_in_sc'].values, clf_test, agreement_test
).astype(np.float32)

print(f"Meta features: {X_tr.shape[1]}")

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
folds = list(gkf.split(np.arange(len(y_true)), groups=groups))

print("\n[Meta-residual correction]", flush=True)
prev_best = 8.36746
best_meta_mae = 999
best_meta_oof = None; best_meta_test = None

for n_est, lr, min_cs, reg in [(500, 0.05, 1000, 3.0), (300, 0.05, 2000, 5.0),
                                 (200, 0.05, 500, 5.0), (200, 0.05, 5000, 2.0)]:
    corr_oof  = np.zeros(len(y_true))
    corr_test = np.zeros(len(test_raw))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=n_est, learning_rate=lr,
            num_leaves=15, max_depth=4, min_child_samples=min_cs,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=reg, reg_lambda=reg,
            random_state=42, verbose=-1, n_jobs=-1
        )
        m.fit(X_tr[tr_idx], resid_tr[tr_idx],
              eval_set=[(X_tr[val_idx], resid_tr[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        corr_oof[val_idx] = m.predict(X_tr[val_idx])
        corr_test += m.predict(X_te) / 5
    final_oof  = np.maximum(0, best_blend_oof  + corr_oof)
    final_test = np.maximum(0, best_blend_test + corr_test)
    mm = mean_absolute_error(y_true, final_oof)
    print(f"  n_est={n_est} lr={lr} min_cs={min_cs} reg={reg}: OOF={mm:.5f}  delta={mm-best_blend_mae:+.5f}", flush=True)
    if mm < best_meta_mae:
        best_meta_mae = mm; best_meta_oof = final_oof; best_meta_test = final_test

print(f"\nBest meta-residual OOF: {best_meta_mae:.5f}  prev={prev_best:.5f}")

if best_meta_mae < prev_best - 0.0001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_meta_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_neural_meta_resid_OOF{best_meta_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"No improvement over {prev_best:.5f}")
print("Done.")
