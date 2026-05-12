"""
Non-linear position-aware meta-blender.
Input: [fixed, xgb, lv2, rem, row_in_sc] → predict y_true.
Uses GroupKFold(layout_id) + LGB early stopping to prevent overfitting.
Kill if fold1 OOF > 8.38 (must beat current best4=8.3825).
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

# Load FIXED component (ls order → _row_id order)
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

# Oracle OOF predictions (_row_id order)
xgb_tr  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_tr  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_tr  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te  = np.load('results/oracle_seq/test_C_xgb.npy')[te_id2]
lv2_te  = np.load('results/oracle_seq/test_C_log_v2.npy')[te_id2]
rem_te  = np.load('results/oracle_seq/test_C_xgb_remaining.npy')[te_id2]

best4_tr = 0.64*fixed_tr + 0.12*xgb_tr + 0.16*lv2_tr + 0.08*rem_tr
best4_te = 0.64*fixed_te + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te

# Meta-features
def make_meta(fixed, xgb, lv2, rem, pos_arr):
    agreement = np.std(np.stack([fixed, xgb, lv2, rem], axis=1), axis=1)
    return np.column_stack([
        fixed, xgb, lv2, rem,
        pos_arr,
        pos_arr ** 2,
        pos_arr / 24.0,          # normalized position
        fixed - xgb,             # fixed vs xgb gap
        fixed - rem,             # fixed vs rem gap
        xgb - lv2,               # oracle gap
        agreement,               # disagreement level
        np.log1p(np.maximum(0, fixed)),
        np.log1p(np.maximum(0, rem)),
    ])

X_tr = make_meta(fixed_tr, xgb_tr, lv2_tr, rem_tr, train_raw['row_in_sc'].values).astype(np.float32)
X_te = make_meta(fixed_te, xgb_te, lv2_te, rem_te, test_raw['row_in_sc'].values).astype(np.float32)

PARAMS = dict(
    objective='mae', n_estimators=2000, learning_rate=0.05,
    num_leaves=31, max_depth=5, min_child_samples=200,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=4,
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof = np.zeros(len(y_true))
test_preds = []

print("Training meta-blender...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(y_true)), groups=groups)):
    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr[val_idx], y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[val_idx] = np.maximum(0, model.predict(X_tr[val_idx]))
    test_preds.append(np.maximum(0, model.predict(X_te)))
    mae = np.mean(np.abs(oof[val_idx] - y_true[val_idx]))
    it = model.best_iteration_
    print(f"Fold {fold_i+1}: meta={mae:.5f}  it={it}", flush=True)

test_avg = np.mean(test_preds, axis=0)
oof_mae  = np.mean(np.abs(oof - y_true))
best4_mae = np.mean(np.abs(best4_tr - y_true))
print(f"\nMeta-blender OOF: {oof_mae:.5f}  (best4: {best4_mae:.5f})  delta={oof_mae-best4_mae:+.5f}", flush=True)

os.makedirs('results/meta', exist_ok=True)
np.save('results/meta/meta_blend_oof.npy', oof)
np.save('results/meta/meta_blend_test.npy', test_avg)
print("Saved.", flush=True)

# Submit if better than current best
THRESHOLD = 8.3820
if oof_mae < THRESHOLD:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': test_avg})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_meta_OOF{oof_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"OOF {oof_mae:.5f} >= threshold {THRESHOLD:.5f}. Not saved.")
print("Done.")
