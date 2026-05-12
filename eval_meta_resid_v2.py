"""
Meta-residual v2: broader hyperparameter search + ensemble of corrections.
Starting from neural+triple_gate base (OOF=8.36724).
Goal: find most conservative config that still improves OOF, for reliable LB.
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

w_base = {'mega37':0.4424,'rank_adj':0.0747,'oracle_log_v2':0.1540,
          'oracle_xgb_combined':0.1406,'oracle_xgb_v31':0.0589,
          'mlp_deep':0.0506,'mlp_s2':0.0539,'mlp_gelu':0.0198}
total = sum(w_base.values()); w_base = {k:v/total for k,v in w_base.items()}

base_oof_parts = {
    'mega37':  d37['meta_avg_oof'][id2], 'mega34': d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'mlp_deep': dm['oof'][id2], 'mlp_s2': ds2['oof'][id2], 'mlp_gelu': dg['oof'][id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_xgb_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
base_test_parts = {
    'mega37':  d37['meta_avg_test'][te_id2], 'mega34': d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'mlp_deep': dm['test'][te_id2], 'mlp_s2': ds2['test'][te_id2], 'mlp_gelu': dg['test'][te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_xgb_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

base_oof  = np.clip(sum(w_base.get(k,0)*base_oof_parts[k]  for k in base_oof_parts), 0, None)
base_test = np.clip(sum(w_base.get(k,0)*base_test_parts[k] for k in base_test_parts), 0, None)

# Triple gate
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]
mlp_deep_oof  = dm['oof'][id2]; mlp_deep_test = dm['test'][te_id2]

# Reproduce triple gate (p0.11w0.025+p0.27w0.040+p0.45w0.090 → OOF=8.36724)
m1=(clf_oof>0.11).astype(float); m1t=(clf_test>0.11).astype(float)
m2=(clf_oof>0.27).astype(float); m2t=(clf_test>0.27).astype(float)
m3=(clf_oof>0.45).astype(float); m3t=(clf_test>0.45).astype(float)
b1=(1-m1*0.025)*base_oof+m1*0.025*rh_oof; b1t=(1-m1t*0.025)*base_test+m1t*0.025*rh_te
b2=(1-m2*0.040)*b1+m2*0.040*rm_oof; b2t=(1-m2t*0.040)*b1t+m2t*0.040*rm_te
b3=(1-m3*0.090)*b2+m3*0.090*mlp_deep_oof; b3t=(1-m3t*0.090)*b2t+m3t*0.090*mlp_deep_test
best_blend_oof = b3; best_blend_test = b3t
best_blend_mae = np.mean(np.abs(best_blend_oof - y_true))
print(f"Triple gate OOF: {best_blend_mae:.5f}")

resid_tr = y_true - best_blend_oof

# Meta features
def build_meta(base_pred, oracle_xgb, oracle_rem, oracle_lv2, mlp_d, pos, clf_p):
    agreement = np.std(np.stack([oracle_xgb, oracle_rem, oracle_lv2, mlp_d], axis=1), axis=1)
    return np.column_stack([
        base_pred, oracle_xgb, oracle_rem, oracle_lv2, mlp_d,
        pos, pos/24.0,
        clf_p, clf_p**2, np.sqrt(np.maximum(0, clf_p)),
        base_pred - oracle_xgb, base_pred - oracle_lv2,
        mlp_d - base_pred,
        agreement,
        np.log1p(np.maximum(0, base_pred)),
        np.log1p(np.maximum(0, oracle_xgb)),
        (base_pred > 50).astype(float), (base_pred > 80).astype(float),
        (base_pred > 30).astype(float),
        # High residual signal: where is current prediction vs oracle?
        base_pred / np.maximum(1, oracle_xgb),  # ratio
    ])

X_tr = build_meta(best_blend_oof, base_oof_parts['oracle_xgb'], base_oof_parts['oracle_xgb_rem'],
                  base_oof_parts['oracle_log_v2'], mlp_deep_oof,
                  train_raw['row_in_sc'].values, clf_oof).astype(np.float32)
X_te = build_meta(best_blend_test, base_test_parts['oracle_xgb'], base_test_parts['oracle_xgb_rem'],
                  base_test_parts['oracle_log_v2'], mlp_deep_test,
                  test_raw['row_in_sc'].values, clf_test).astype(np.float32)

print(f"Meta features: {X_tr.shape[1]}")

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
folds = list(gkf.split(np.arange(len(y_true)), groups=groups))

print("\n[Broad hyperparameter search]", flush=True)
prev_best = 8.36746
results = []

configs = [
    # (n_est, lr, min_cs, reg, nl, tag)
    (500, 0.05, 1000, 3.0, 15, 'base'),
    (500, 0.05, 2000, 3.0, 15, 'mc2k'),
    (500, 0.05, 3000, 3.0, 15, 'mc3k'),
    (500, 0.05, 5000, 3.0, 15, 'mc5k'),
    (500, 0.05, 1000, 5.0, 15, 'reg5'),
    (500, 0.05, 1000, 8.0, 15, 'reg8'),
    (300, 0.03, 1000, 3.0, 15, 'lr3'),
    (500, 0.05, 1000, 3.0, 7,  'nl7'),   # shallower
    (1000, 0.03, 1000, 3.0, 15, 'n1k'),
]

meta_predictions = {}
for n_est, lr, min_cs, reg, nl, tag in configs:
    corr_oof  = np.zeros(len(y_true))
    corr_test = np.zeros(len(test_raw))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=n_est, learning_rate=lr,
            num_leaves=nl, max_depth=4, min_child_samples=min_cs,
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
    print(f"  [{tag}] n_est={n_est} lr={lr} min_cs={min_cs} reg={reg} nl={nl}: OOF={mm:.5f}  delta={mm-best_blend_mae:+.5f}", flush=True)
    results.append((mm, tag, final_oof, final_test, corr_oof, corr_test))
    meta_predictions[tag] = (corr_oof, corr_test)

results.sort(key=lambda x: x[0])
best_mae, best_tag, best_oof, best_test, _, _ = results[0]
print(f"\nBest single config: [{best_tag}] OOF={best_mae:.5f}")

# Ensemble of corrections (average multiple configs)
print(f"\n[Ensemble of corrections]", flush=True)
for n_configs in [2, 3, 4, 5]:
    top = results[:n_configs]
    ens_corr_oof  = sum(r[4] for r in top) / n_configs
    ens_corr_test = sum(r[5] for r in top) / n_configs
    ens_oof  = np.maximum(0, best_blend_oof  + ens_corr_oof)
    ens_test = np.maximum(0, best_blend_test + ens_corr_test)
    mm = mean_absolute_error(y_true, ens_oof)
    tags = [r[1] for r in top]
    print(f"  Top-{n_configs} ensemble {tags}: OOF={mm:.5f}  delta={mm-best_blend_mae:+.5f}", flush=True)
    if mm < best_mae:
        best_mae = mm; best_oof = ens_oof; best_test = ens_test; best_tag = f'ens_top{n_configs}'

# Conservative blend: linear interpolation between no-correction and best correction
print(f"\n[Conservative blend: blend_ratio of correction]")
for alpha in [0.3, 0.5, 0.7, 1.0]:
    blended_oof  = np.maximum(0, best_blend_oof  + alpha * results[0][4])
    blended_test = np.maximum(0, best_blend_test + alpha * results[0][5])
    mm = mean_absolute_error(y_true, blended_oof)
    print(f"  alpha={alpha}: OOF={mm:.5f}")

print(f"\n=== FINAL: best={best_mae:.5f}  prev={prev_best:.5f}  delta={best_mae-prev_best:+.5f} ===")

if best_mae < prev_best - 0.0001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_v2_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")

# Always save the most conservative config (min_cs=5000)
mc5k = next(r for r in results if r[1] == 'mc5k')
mc5k_mae = mc5k[0]
mc5k_test = mc5k[3]
print(f"\nConservative [mc5k] OOF={mc5k_mae:.5f}")
if mc5k_mae < prev_best - 0.0001:
    sub = np.maximum(0, mc5k_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_conservative_OOF{mc5k_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED conservative: {fname} ***")
print("Done.")
