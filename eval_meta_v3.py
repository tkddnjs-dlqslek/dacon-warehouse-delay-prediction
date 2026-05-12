"""
Meta-residual v3: reproduce 8.35122 approach (5-oracle agreement) + broader sweep.
Key diff from eval_meta_push: agreement computed from 5 oracle variants only (not including mlp_d).
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

bop = {
    'mega37':  d37['meta_avg_oof'][id2], 'mega34': d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'mlp_deep': dm['oof'][id2], 'mlp_s2': ds2['oof'][id2], 'mlp_gelu': dg['oof'][id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_xgb_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
btp = {
    'mega37':  d37['meta_avg_test'][te_id2], 'mega34': d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'mlp_deep': dm['test'][te_id2], 'mlp_s2': ds2['test'][te_id2], 'mlp_gelu': dg['test'][te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_xgb_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

base_oof  = np.clip(sum(w_base.get(k,0)*bop[k]  for k in bop), 0, None)
base_test = np.clip(sum(w_base.get(k,0)*btp[k] for k in btp), 0, None)

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]
mlp_deep_oof  = dm['oof'][id2]; mlp_deep_test = dm['test'][te_id2]

m1=(clf_oof>0.112).astype(float); m1t=(clf_test>0.112).astype(float)
m2=(clf_oof>0.265).astype(float); m2t=(clf_test>0.265).astype(float)
m3=(clf_oof>0.45).astype(float);  m3t=(clf_test>0.45).astype(float)
b1=(1-m1*0.0261)*base_oof+m1*0.0261*rh_oof; b1t=(1-m1t*0.0261)*base_test+m1t*0.0261*rh_te
b2=(1-m2*0.0384)*b1+m2*0.0384*rm_oof; b2t=(1-m2t*0.0384)*b1t+m2t*0.0384*rm_te
b3=(1-m3*0.090)*b2+m3*0.090*mlp_deep_oof; b3t=(1-m3t*0.090)*b2t+m3t*0.090*mlp_deep_test
best_blend_oof = b3; best_blend_test = b3t
best_blend_mae = np.mean(np.abs(best_blend_oof - y_true))
print(f"Base+triple_gate OOF: {best_blend_mae:.5f}")
resid_tr = y_true - best_blend_oof

# 5-oracle agreement (same as eval_neural_meta_resid.py)
agree_oof  = np.std(np.stack([bop['oracle_log_v2'], bop['oracle_xgb_combined'],
    bop['oracle_xgb_v31'], bop['oracle_xgb'], bop['oracle_xgb_rem']], axis=1), axis=1)
agree_test = np.std(np.stack([btp['oracle_log_v2'], btp['oracle_xgb_combined'],
    btp['oracle_xgb_v31'], btp['oracle_xgb'], btp['oracle_xgb_rem']], axis=1), axis=1)

def build_meta(base_pred, oracle_xgb, oracle_rem, oracle_lv2, mlp_d, pos, clf_p, agree):
    return np.column_stack([
        base_pred, oracle_xgb, oracle_rem, oracle_lv2, mlp_d,
        pos, pos/24.0,
        clf_p, clf_p**2,
        base_pred - oracle_xgb, base_pred - oracle_lv2,
        mlp_d - base_pred, agree,
        np.log1p(np.maximum(0, base_pred)),
        np.log1p(np.maximum(0, oracle_xgb)),
        (base_pred > 50).astype(float), (base_pred > 80).astype(float),
    ])

X_tr = build_meta(best_blend_oof, bop['oracle_xgb'], bop['oracle_xgb_rem'],
                  bop['oracle_log_v2'], mlp_deep_oof,
                  train_raw['row_in_sc'].values, clf_oof, agree_oof).astype(np.float32)
X_te = build_meta(best_blend_test, btp['oracle_xgb'], btp['oracle_xgb_rem'],
                  btp['oracle_log_v2'], mlp_deep_test,
                  test_raw['row_in_sc'].values, clf_test, agree_test).astype(np.float32)

print(f"Meta features: {X_tr.shape[1]}")

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
folds = list(gkf.split(np.arange(len(y_true)), groups=groups))

prev_best = 8.35122
print(f"\n[v3 sweep with 5-oracle agreement] prev_best={prev_best:.5f}", flush=True)

configs = [
    (500,  0.05, 1000, 1.0, 15, 'reg1'),
    (500,  0.05, 1000, 2.0, 15, 'reg2'),
    (500,  0.05, 1000, 3.0, 15, 'reg3'),   # should reproduce 8.35122
    (500,  0.05, 1000, 5.0, 15, 'reg5'),
    (500,  0.05, 1000, 8.0, 15, 'reg8'),
    (500,  0.05, 500,  2.0, 15, 'mc500_r2'),
    (500,  0.05, 500,  3.0, 15, 'mc500_r3'),
    (800,  0.03, 1000, 3.0, 15, 'lr3_n800'),
    (500,  0.05, 1000, 3.0, 31, 'nl31'),
    (500,  0.05, 2000, 3.0, 15, 'mc2k'),
    (500,  0.05, 500,  1.5, 15, 'mc500_r1.5'),
]

all_results = {}
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
    mm = mean_absolute_error(y_true, final_oof)
    print(f"  [{tag:12s}] OOF={mm:.5f}  delta={mm-best_blend_mae:+.5f}", flush=True)
    all_results[tag] = (mm, corr_oof, corr_test)

sorted_tags = sorted(all_results.keys(), key=lambda t: all_results[t][0])
print(f"\n[Top 5 by OOF]")
for t in sorted_tags[:5]:
    print(f"  {t}: {all_results[t][0]:.5f}")

print(f"\n[Ensemble of top-N]", flush=True)
best_ens_mae = prev_best; best_ens_oof = None; best_ens_test = None; best_ens_tag = None

for n in [2, 3, 4, 5, 6]:
    top_tags = sorted_tags[:n]
    avg_oof  = np.mean([all_results[t][1] for t in top_tags], axis=0)
    avg_test = np.mean([all_results[t][2] for t in top_tags], axis=0)
    ens_oof  = np.maximum(0, best_blend_oof  + avg_oof)
    ens_test = np.maximum(0, best_blend_test + avg_test)
    mm = mean_absolute_error(y_true, ens_oof)
    print(f"  Top-{n} {top_tags}: OOF={mm:.5f}  delta={mm-best_blend_mae:+.5f}", flush=True)
    if mm < best_ens_mae:
        best_ens_mae = mm; best_ens_oof = ens_oof; best_ens_test = ens_test
        best_ens_tag = f'v3_ens_top{n}'
        print(f"  ★ NEW BEST: {best_ens_mae:.5f}")

# Best single
best_single_tag = sorted_tags[0]
best_single_mae = all_results[best_single_tag][0]
if best_single_mae < prev_best:
    print(f"\n★ Best single: [{best_single_tag}] OOF={best_single_mae:.5f} < prev={prev_best:.5f}")

print(f"\n=== SUMMARY ===")
print(f"  best_single: {best_single_mae:.5f} [{best_single_tag}]")
print(f"  best_ens:    {best_ens_mae:.5f} [{best_ens_tag}]")

sample_sub = pd.read_csv('sample_submission.csv')
saved = []

if best_ens_mae < prev_best - 0.0002:
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values,
                           'avg_delay_minutes_next_30m': best_ens_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_v3_ens_OOF{best_ens_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False); saved.append(fname)
    print(f"*** SAVED: {fname} ***")

if best_single_mae < prev_best - 0.0002:
    best_s_test = np.maximum(0, best_blend_test + all_results[best_single_tag][2])
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values,
                           'avg_delay_minutes_next_30m': best_s_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_v3_single_OOF{best_single_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False); saved.append(fname)
    print(f"*** SAVED single: {fname} ***")

if not saved:
    print("No improvement beyond prev_best-0.0002")
print("Done.")
