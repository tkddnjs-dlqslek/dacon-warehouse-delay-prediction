"""
Residual meta-blender: learns correction to best4 linear blend.
Meta predicts (y_true - best4_oof), applied as best4_test + correction.
Maintains best4's calibration, only learns per-row adjustments.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_tr = (fw['mega33']*d['meta_avg_oof'][id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_te = (fw['mega33']*d['meta_avg_test'][te_id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

xgb_tr = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_tr = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_tr = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')[te_id2]
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')[te_id2]
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')[te_id2]

best4_tr = 0.64*fixed_tr + 0.12*xgb_tr + 0.16*lv2_tr + 0.08*rem_tr
best4_te = 0.64*fixed_te + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te

# Residual target: y_true - best4
resid_tr = y_true - best4_tr
print(f"Residual stats: mean={resid_tr.mean():.4f}  std={resid_tr.std():.4f}  abs_mean={np.abs(resid_tr).mean():.4f}", flush=True)

def make_meta(fixed, xgb, lv2, rem, pos_arr):
    agreement = np.std(np.stack([fixed, xgb, lv2, rem], axis=1), axis=1)
    return np.column_stack([
        fixed, xgb, lv2, rem,
        pos_arr,
        pos_arr / 24.0,
        fixed - xgb,
        fixed - rem,
        xgb - lv2,
        agreement,
        np.log1p(np.maximum(0, fixed)),
        np.log1p(np.maximum(0, rem)),
    ])

X_tr = make_meta(fixed_tr, xgb_tr, lv2_tr, rem_tr, train_raw['row_in_sc'].values).astype(np.float32)
X_te = make_meta(fixed_te, xgb_te, lv2_te, rem_te, test_raw['row_in_sc'].values).astype(np.float32)

PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.03,
    num_leaves=31, max_depth=5, min_child_samples=500,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=2.0, reg_lambda=2.0,
    random_state=42, verbose=-1, n_jobs=4,
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
correction_oof  = np.zeros(len(y_true))
correction_test = np.zeros(len(test_raw))

print("Training residual meta-blender...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(y_true)), groups=groups)):
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr[tr_idx], resid_tr[tr_idx],
              eval_set=[(X_tr[val_idx], resid_tr[val_idx])],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
    correction_oof[val_idx] = model.predict(X_tr[val_idx])
    correction_test += model.predict(X_te) / 5
    # Evaluate on validation
    corrected_val = np.maximum(0, best4_tr[val_idx] + correction_oof[val_idx])
    mae = np.mean(np.abs(corrected_val - y_true[val_idx]))
    print(f"Fold {fold_i+1}: corrected MAE={mae:.5f}  correction_mean={correction_oof[val_idx].mean():.3f}  it={model.best_iteration_}", flush=True)

# Final predictions
final_oof  = np.maximum(0, best4_tr + correction_oof)
final_test = np.maximum(0, best4_te + correction_test)

oof_mae   = np.mean(np.abs(final_oof - y_true))
best4_mae = np.mean(np.abs(best4_tr - y_true))
print(f"\nResidual meta OOF: {oof_mae:.5f}  (best4: {best4_mae:.5f})  delta={oof_mae-best4_mae:+.5f}", flush=True)
print(f"Train corr_mean: {correction_oof.mean():.4f}  Test corr_mean: {correction_test.mean():.4f}")
print(f"Final test mean: {final_test.mean():.4f}  (best4 test mean: {best4_te.mean():.4f})")

os.makedirs('results/meta', exist_ok=True)
np.save('results/meta/meta_resid_oof.npy', final_oof)
np.save('results/meta/meta_resid_test.npy', final_test)

THRESHOLD = 8.3820
if oof_mae < THRESHOLD:
    sample_sub = pd.read_csv('sample_submission.csv')
    test_raw_orig = pd.read_csv('test.csv')
    test_raw_orig['_row_id'] = test_raw_orig['ID'].str.replace('TEST_','').astype(int)
    test_raw_orig = test_raw_orig.sort_values('_row_id').reset_index(drop=True)
    sub_df = pd.DataFrame({'ID': test_raw_orig['ID'].values, 'avg_delay_minutes_next_30m': final_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_resid_OOF{oof_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"OOF {oof_mae:.5f} >= threshold {THRESHOLD}. Not saved.")
print("Done.")
