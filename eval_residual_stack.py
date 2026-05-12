"""
Residual stacking: train a meta-model on the residuals of the best blend.
Uses GroupKFold to avoid leakage. Meta-features: all OOF predictions + layout info.
Goal: capture systematic per-layout prediction errors.
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
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)

# Current best scipy blend base (no cascade)
w_scipy = np.array([0.4997, 0.1019, 0.0, 0.0, 0.0, 0.1338, 0.1741, 0.0991])
w_scipy /= w_scipy.sum()
keys8 = ['mega33','rank_adj','iter_r1','iter_r2','iter_r3','oracle_xgb','oracle_lv2','oracle_rem']
C8_oof = np.column_stack([
    d33['meta_avg_oof'][id2],
    np.load('results/ranking/rank_adj_oof.npy')[id2],
    np.load('results/iter_pseudo/round1_oof.npy')[id2],
    np.load('results/iter_pseudo/round2_oof.npy')[id2],
    np.load('results/iter_pseudo/round3_oof.npy')[id2],
    np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
])
C8_test = np.column_stack([
    d33['meta_avg_test'][te_id2],
    np.load('results/ranking/rank_adj_test.npy')[te_id2],
    np.load('results/iter_pseudo/round1_test.npy')[te_id2],
    np.load('results/iter_pseudo/round2_test.npy')[te_id2],
    np.load('results/iter_pseudo/round3_test.npy')[te_id2],
    np.load('results/oracle_seq/test_C_xgb.npy'),
    np.load('results/oracle_seq/test_C_log_v2.npy'),
    np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
])
scipy_oof  = np.clip(C8_oof  @ w_scipy, 0, None)
scipy_test = np.clip(C8_test @ w_scipy, 0, None)
scipy_mae  = np.mean(np.abs(scipy_oof - y_true))
print(f"Scipy base OOF: {scipy_mae:.5f}")

# Cascade dual gate
m11 = (np.load('results/cascade/clf_oof.npy')[id2]  > 0.11).astype(float)
m25 = (np.load('results/cascade/clf_oof.npy')[id2]  > 0.25).astype(float)
m11_te = (np.load('results/cascade/clf_test.npy')[te_id2] > 0.11).astype(float)
m25_te = (np.load('results/cascade/clf_test.npy')[te_id2] > 0.25).astype(float)
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

casc_oof  = (1-m11*0.03)*scipy_oof  + m11*0.03*lgb_rh_oof
casc_test = (1-m11_te*0.03)*scipy_test + m11_te*0.03*lgb_rh_test
casc_oof  = (1-m25*0.03)*casc_oof  + m25*0.03*lgb_rm_oof
casc_test = (1-m25_te*0.03)*casc_test + m25_te*0.03*lgb_rm_test
casc_mae  = np.mean(np.abs(casc_oof - y_true))
print(f"Scipy+cascade OOF: {casc_mae:.5f}  (this is current best)")

# Load ALL oracle as meta-features
oracle_names = [
    'xgb', 'log_v2', 'xgb_remaining', 'cb', 'lgb_dual', 'lgb_latepos',
    'lgb_remaining', 'lgb_remaining_v3', 'xgb_v31', 'xgb_v31_sc',
    'xgb_combined', 'lgb_stack', 'ranklag',
]
meta_oof_parts  = [casc_oof.reshape(-1,1), scipy_oof.reshape(-1,1),
                   d34['meta_avg_oof'][id2].reshape(-1,1),
                   d37['meta_avg_oof'][id2].reshape(-1,1)]
meta_test_parts = [casc_test.reshape(-1,1), scipy_test.reshape(-1,1),
                   d34['meta_avg_test'][te_id2].reshape(-1,1),
                   d37['meta_avg_test'][te_id2].reshape(-1,1)]

for name in oracle_names:
    try:
        o = np.load(f'results/oracle_seq/oof_seqC_{name}.npy').reshape(-1,1)
        t = np.load(f'results/oracle_seq/test_C_{name}.npy').reshape(-1,1)
        meta_oof_parts.append(o); meta_test_parts.append(t)
    except FileNotFoundError:
        pass

# Layout info
layout_info = pd.read_csv('layout_info.csv')
num_cols = ['aisle_width_avg','intersection_count','one_way_ratio','pack_station_count',
            'charger_count','layout_compactness','zone_dispersion','robot_total',
            'building_age_years','floor_area_sqm']
layout_info = layout_info[['layout_id'] + num_cols].fillna(0)
layout_tr = train_raw[['layout_id']].merge(layout_info, on='layout_id', how='left')[num_cols].values.astype(np.float32)
layout_te = test_raw[['layout_id']].merge(layout_info, on='layout_id', how='left')[num_cols].values.astype(np.float32)

X_meta_tr = np.hstack(meta_oof_parts  + [layout_tr])
X_meta_te = np.hstack(meta_test_parts + [layout_te])
print(f"\nMeta features: {X_meta_tr.shape[1]}")

# GroupKFold stacking
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.arange(len(y_true)), groups=groups))

print("\n[LGB stack]", flush=True)
prev_best = 8.37851
best_stack_mae = 999
best_stack_oof = None; best_stack_test = None

for lr, n_est, nl in [(0.05, 300, 31), (0.03, 500, 31), (0.02, 800, 15)]:
    stack_oof  = np.zeros(len(y_true))
    stack_test = np.zeros(len(test_raw))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=n_est, learning_rate=lr,
            num_leaves=nl, max_depth=4, min_child_samples=200,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
            random_state=42, verbose=-1, n_jobs=-1
        )
        m.fit(X_meta_tr[tr_idx], y_true[tr_idx],
              eval_set=[(X_meta_tr[val_idx], y_true[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        stack_oof[val_idx] = m.predict(X_meta_tr[val_idx])
        stack_test += m.predict(X_meta_te) / 5
    stack_oof  = np.clip(stack_oof, 0, None)
    stack_test = np.clip(stack_test, 0, None)
    mm = mean_absolute_error(y_true, stack_oof)
    print(f"  lr={lr} n_est={n_est} nl={nl}: OOF={mm:.5f}  delta={mm-casc_mae:+.5f}", flush=True)
    if mm < best_stack_mae:
        best_stack_mae = mm; best_stack_oof = stack_oof; best_stack_test = stack_test

print(f"\nBest stack OOF: {best_stack_mae:.5f}  delta={best_stack_mae-casc_mae:+.5f}")

if best_stack_mae < prev_best - 0.0001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_stack_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_residual_stack_OOF{best_stack_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"No improvement over {prev_best:.5f}")
print("Done.")
