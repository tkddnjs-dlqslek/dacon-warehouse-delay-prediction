import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

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

# oracle_NEW baseline (w_fixed=0.64)
oracle_oof  = np.clip(0.64*fixed_oof  + 0.12*xgb_o_oof  + 0.16*lv2_o_oof  + 0.08*rem_o_oof,  0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW baseline OOF: {base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

# Varying FIXED weight from 0.64 to 1.00
# The remaining components (xgb=0.12/0.36, lv2=0.16/0.36, rem=0.08/0.36 of the 0.36 slack)
# Original proportions within the seq components: xgb:lv2:rem = 0.12:0.16:0.08 = 1/3 : 4/9 : 2/9
# But let's also try different FIXED+seq proportional rescaling:
print('\n--- Vary FIXED weight (seq components proportionally scaled) ---')
print(f'{"w_fixed":>8}  {"OOF":>9}  {"delta_OOF":>10}  {"test_mean":>10}  {"delta_test":>10}')
for w_fixed in [0.60, 0.64, 0.65, 0.67, 0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.90, 1.00]:
    w_rem = 1.0 - w_fixed
    # proportional rescale of seq: xgb=0.12, lv2=0.16, rem=0.08 (sum=0.36)
    w_xgb = 0.12 * w_rem / 0.36
    w_lv2 = 0.16 * w_rem / 0.36
    w_rem2 = 0.08 * w_rem / 0.36
    oof_  = np.clip(w_fixed*fixed_oof  + w_xgb*xgb_o_oof  + w_lv2*lv2_o_oof  + w_rem2*rem_o_oof,  0, None)
    test_ = np.clip(w_fixed*fixed_test + w_xgb*xgb_o_test + w_lv2*lv2_o_test + w_rem2*rem_o_test, 0, None)
    oof_mae = mae(oof_)
    print(f'{w_fixed:>8.2f}  {oof_mae:>9.5f}  {oof_mae-base_oof:>+10.5f}  {test_.mean():>10.3f}  {test_.mean()-oracle_test.mean():>+10.3f}')

# Also try: FIXED alone vs seq alone
print('\n--- FIXED-only vs seq-only ---')
oof_f  = np.clip(fixed_oof, 0, None)
test_f = np.clip(fixed_test, 0, None)
oof_s  = np.clip((0.12*xgb_o_oof+0.16*lv2_o_oof+0.08*rem_o_oof)/0.36, 0, None)
test_s = np.clip((0.12*xgb_o_test+0.16*lv2_o_test+0.08*rem_o_test)/0.36, 0, None)
print(f'  FIXED only:   OOF={mae(oof_f):.5f}  delta={mae(oof_f)-base_oof:+.5f}  test_mean={test_f.mean():.3f}')
print(f'  Seq only:     OOF={mae(oof_s):.5f}  delta={mae(oof_s)-base_oof:+.5f}  test_mean={test_s.mean():.3f}')

# Per-fold analysis for key candidates
print('\n--- Fold-level analysis for key w_fixed values ---')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_splits = [(tr, np.sort(val)) for tr, val in gkf.split(np.arange(len(train_raw)), groups=groups)]

for w_fixed in [0.64, 0.70, 0.75, 0.80]:
    w_rem = 1.0 - w_fixed
    w_xgb = 0.12 * w_rem / 0.36
    w_lv2 = 0.16 * w_rem / 0.36
    w_rem2 = 0.08 * w_rem / 0.36
    oof_ = np.clip(w_fixed*fixed_oof + w_xgb*xgb_o_oof + w_lv2*lv2_o_oof + w_rem2*rem_o_oof, 0, None)
    fold_maes = [mean_absolute_error(y_true[vs], oof_[vs]) for _, vs in fold_splits]
    print(f'  w_fixed={w_fixed:.2f}: OOF={mae(oof_):.5f}  folds={[f"{m:.3f}" for m in fold_maes]}')

# Best candidate: generate submission
print('\n--- Generate submission for best candidate ---')
best_w = 0.70
w_rem = 1.0 - best_w
w_xgb = 0.12 * w_rem / 0.36
w_lv2 = 0.16 * w_rem / 0.36
w_rem2 = 0.08 * w_rem / 0.36
best_test = np.clip(best_w*fixed_test + w_xgb*xgb_o_test + w_lv2*lv2_o_test + w_rem2*rem_o_test, 0, None)
best_oof  = np.clip(best_w*fixed_oof  + w_xgb*xgb_o_oof  + w_lv2*lv2_o_oof  + w_rem2*rem_o_oof,  0, None)
print(f'  w_fixed=0.70: OOF={mae(best_oof):.5f}  test_mean={best_test.mean():.3f}')
sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = best_test
oof_val = mae(best_oof)
fname = f'submission_highFixed070_OOF{oof_val:.5f}.csv'
sub.to_csv(fname, index=False)
print(f'  Saved: {fname}')

# Also generate w=0.75
best_w2 = 0.75
w_rem2b = 1.0 - best_w2
w_xgb2 = 0.12 * w_rem2b / 0.36
w_lv22 = 0.16 * w_rem2b / 0.36
w_rem2c = 0.08 * w_rem2b / 0.36
best_test2 = np.clip(best_w2*fixed_test + w_xgb2*xgb_o_test + w_lv22*lv2_o_test + w_rem2c*rem_o_test, 0, None)
best_oof2  = np.clip(best_w2*fixed_oof  + w_xgb2*xgb_o_oof  + w_lv22*lv2_o_oof  + w_rem2c*rem_o_oof,  0, None)
print(f'  w_fixed=0.75: OOF={mae(best_oof2):.5f}  test_mean={best_test2.mean():.3f}')
sub2 = pd.read_csv('sample_submission.csv')
sub2['avg_delay_minutes_next_30m'] = best_test2
oof_val2 = mae(best_oof2)
fname2 = f'submission_highFixed075_OOF{oof_val2:.5f}.csv'
sub2.to_csv(fname2, index=False)
print(f'  Saved: {fname2}')

# FIXED only submission
print(f'\n  FIXED only: OOF={mae(oof_f):.5f}  test_mean={test_f.mean():.3f}')
sub3 = pd.read_csv('sample_submission.csv')
sub3['avg_delay_minutes_next_30m'] = test_f
oof_val3 = mae(oof_f)
fname3 = f'submission_FIXED_only_OOF{oof_val3:.5f}.csv'
sub3.to_csv(fname3, index=False)
print(f'  Saved: {fname3}')

print('\nDone.')
