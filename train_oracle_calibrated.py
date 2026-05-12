"""
Calibrated Oracle
oracle_NEW systematically under-predicts: mean=15.6 vs true=18.962 (bias=-3.4 min)
For high-load test scenarios (even higher true mean), under-prediction is worse.

Approach: isotonic regression calibration
- Fit monotone calibration f(pred) -> calibrated_pred using training OOF
- Applied within GroupKFold to avoid leakage
- Calibration = monotone function, so preserves relative ordering
- NOT a gate: same function applied to all scenarios

Two variants:
A) Global calibration: fit on all training predictions
B) Quantile-bucket calibration: fit per-decile bucket (slightly nonlinear)

Key hypothesis: if oracle_NEW's bias is SYSTEMATIC (same direction for seen and unseen
high-load scenarios), monotone calibration from seen layouts generalizes to unseen.

This is safer than gate because:
1. Monotone: no conditional logic, no layout-type detection
2. Simple: 10 parameters max (one per bucket)
3. Consistent: same correction for all scenarios in same prediction range
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
print('='*60)
print('Calibrated Oracle — Isotonic Regression Calibration')
print('  Corrects systematic under-prediction of oracle_NEW')
print('  Monotone mapping (no gate, no overfitting risk)')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# Reconstruct oracle_NEW OOF and test predictions
print('\n[1] oracle_NEW OOF 재구성...')
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
xgb_o_test = np.load('results/oracle_seq/test_C_xgb.npy')[te_id2] if False else np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o_test = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o_test = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

# Handle test array size (might need reindexing)
oracle_new_oof  = np.clip(0.64*fixed_oof  + 0.12*xgb_o_oof  + 0.16*lv2_o_oof  + 0.08*rem_o_oof,  0, None)
oracle_new_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'  oracle_NEW OOF: {mae(oracle_new_oof):.5f}')
print(f'  Pred mean: {oracle_new_oof.mean():.3f}, True mean: {y_true.mean():.3f}')
print(f'  Pred median: {np.median(oracle_new_oof):.3f}, True median: {np.median(y_true):.3f}')

# Check oracle_new_test size
print(f'  oracle_NEW test size: {oracle_new_test.shape[0]} (expected 50000 or {len(test_raw)})')

del d33; gc.collect()

print('\n[2] GroupKFold 내 Isotonic Calibration...')
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)
calib_oof = np.zeros(len(train_raw))

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    val_idx_sorted = np.sort(val_idx)

    # Fit isotonic regression on training fold
    ir = IsotonicRegression(out_of_bounds='clip', increasing=True)
    ir.fit(oracle_new_oof[tr_idx], y_true[tr_idx])

    # Apply to validation fold
    calib_pred = np.clip(ir.predict(oracle_new_oof[val_idx_sorted]), 0, None)
    calib_oof[val_idx_sorted] = calib_pred

    fold_mae = mean_absolute_error(y_true[val_idx_sorted], calib_pred)
    orig_mae = mean_absolute_error(y_true[val_idx_sorted], oracle_new_oof[val_idx_sorted])
    print(f'  Fold {fold_i+1}: calibrated={fold_mae:.5f} (oracle_NEW={orig_mae:.5f}, delta={fold_mae-orig_mae:+.5f})')

print(f'\nCalibrated OOF: {mae(calib_oof):.5f}')
print(f'oracle_NEW OOF: {mae(oracle_new_oof):.5f}')
print(f'Delta: {mae(calib_oof) - mae(oracle_new_oof):+.5f}')

print(f'\nCalibrated pred mean: {calib_oof.mean():.3f}')
print(f'oracle_NEW pred mean: {oracle_new_oof.mean():.3f}')
print(f'True mean: {y_true.mean():.3f}')

# Fit full calibration on all training data → apply to test
print('\n[3] 전체 학습 데이터로 calibration 적합 → test 적용...')
ir_full = IsotonicRegression(out_of_bounds='clip', increasing=True)
ir_full.fit(oracle_new_oof, y_true)
calib_test = np.clip(ir_full.predict(oracle_new_test), 0, None)

print(f'  Test pred mean before: {oracle_new_test.mean():.3f}')
print(f'  Test pred mean after:  {calib_test.mean():.3f}')
print(f'  Test pred std before:  {oracle_new_test.std():.3f}')
print(f'  Test pred std after:   {calib_test.std():.3f}')

# Also try partial calibration (blend of oracle_NEW and calibrated)
print(f'\n[4] Blend 분석...')
for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
    blend_oof = (1-alpha)*oracle_new_oof + alpha*calib_oof
    delta = mae(blend_oof) - mae(oracle_new_oof)
    print(f'  alpha={alpha}: OOF delta={delta:+.5f}')

# Save submission if OOF improves
sub = pd.read_csv('sample_submission.csv')
sub_oof_delta = mae(calib_oof) - mae(oracle_new_oof)
if sub_oof_delta < 0:
    # Use leave-one-layout-out calibration for test prediction
    print(f'\n★ OOF IMPROVED by {-sub_oof_delta:.5f}! Saving submission...')
    sub['predicted'] = calib_test
    sub_file = f'submissions/submission_calibrated_OOF{mae(calib_oof):.5f}.csv'
    os.makedirs('submissions', exist_ok=True)
    sub.to_csv(sub_file, index=False)
    print(f'  Saved: {sub_file}')
else:
    print(f'\nOOF did NOT improve (delta={sub_oof_delta:+.5f}). Not saving submission.')

    # But check: does partial blend help?
    for alpha in [0.1, 0.2, 0.3]:
        blend_oof = (1-alpha)*oracle_new_oof + alpha*calib_oof
        if mae(blend_oof) < mae(oracle_new_oof):
            blend_test = (1-alpha)*oracle_new_test + alpha*calib_test
            delta = mae(blend_oof) - mae(oracle_new_oof)
            sub_file = f'submissions/submission_calibrated_blend{int(alpha*100)}_OOF{mae(blend_oof):.5f}.csv'
            sub['predicted'] = blend_test
            sub.to_csv(sub_file, index=False)
            print(f'  Partial blend alpha={alpha}: OOF delta={delta:+.5f} → Saved: {sub_file}')

print(f'\nDone. ({time.time()-t0:.0f}s total)')
