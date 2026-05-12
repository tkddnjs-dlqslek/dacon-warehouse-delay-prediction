"""
Targeted correction for 3 test layouts beyond training pack_mean range.
Training max pack_mean = 0.8522 (WH_217, residual=+19.32 even with direct training access).
Test beyond-range: WH_201(0.936), WH_246(0.911), WH_283(0.975).

This correction has ZERO OOF impact (no training layout triggers the condition).
Pure extrapolation gamble: apply +19.32 (conservative) or extrapolated correction.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle

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
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

# Also oracle_oof for verification
fixed_oof  = (fw['mega33']*d33['meta_avg_oof'][id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_oof = np.clip(0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'oracle_NEW OOF: {mae(oracle_oof):.5f}  test_mean={oracle_test.mean():.3f}')

# Training layout pack stats
tr_ly_pack = train_raw.groupby('layout_id')['pack_utilization'].mean()
te_ly_pack = test_raw.groupby('layout_id')['pack_utilization'].mean()
train_pack_max = tr_ly_pack.max()

# Training pack residuals for reference
train_raw['oracle_pred'] = oracle_oof
tr_ly_resid = (train_raw.groupby('layout_id')['avg_delay_minutes_next_30m'].mean()
             - train_raw.groupby('layout_id')['oracle_pred'].mean())
tr_ly_df = pd.DataFrame({'pack_mean': tr_ly_pack, 'residual': tr_ly_resid}).reset_index()
tr_ly_df.columns = ['layout_id', 'pack_mean', 'residual']

# WH_217: boundary layout
wh217_pack = tr_ly_pack['WH_217']
wh217_resid = tr_ly_resid['WH_217']
print(f'\nBoundary training layout: WH_217 pack={wh217_pack:.4f}  residual={wh217_resid:+.2f}')

# Other training layouts near boundary (pack > 0.80)
hi_pack_tr = tr_ly_df[tr_ly_df['pack_mean'] > 0.80].sort_values('pack_mean', ascending=False)
print('\nTraining layouts pack > 0.80:')
for _, r in hi_pack_tr.iterrows():
    print(f'  {r["layout_id"]}: pack={r["pack_mean"]:.4f}  residual={r["residual"]:+.2f}')
mean_hi_resid = hi_pack_tr['residual'].mean()
print(f'  Mean residual (pack>0.80): {mean_hi_resid:.2f}')

# Test layouts beyond training range
beyond = te_ly_pack[te_ly_pack > train_pack_max]
print(f'\nTest layouts beyond training pack max ({train_pack_max:.4f}):')
for lid, pack in beyond.sort_values(ascending=False).items():
    print(f'  {lid}: pack={pack:.4f}  oracle_pred_mean={oracle_test[test_raw["layout_id"]==lid].mean():.2f}')

# Linear fit: residual = beta * pack + intercept (for training hi-pack layouts)
beta = np.cov(tr_ly_df['pack_mean'], tr_ly_df['residual'])[0,1] / tr_ly_df['pack_mean'].var()
intercept = tr_ly_df['residual'].mean() - beta * tr_ly_df['pack_mean'].mean()
print(f'\nLayout-level regression: residual = {beta:.2f} * pack + {intercept:.2f}')
print(f'  At pack=0.852 (WH_217): {beta*0.852+intercept:.2f}  (true: {wh217_resid:.2f})')
for lid, pack in beyond.sort_values(ascending=False).items():
    reg_correction = beta * pack + intercept
    print(f'  At pack={pack:.4f} ({lid}): regression correction = {reg_correction:.2f}')

# Scenarios: different targeted correction approaches
print('\n=== Targeted correction submissions ===')
print('(Only layouts with pack_mean > training_max)')

def make_targeted_sub(test_preds, correction_fn, name, beyond_lids):
    """Apply correction only to rows of beyond-range layouts."""
    preds = test_preds.copy()
    total_rows = 0
    for lid in beyond_lids:
        mask = test_raw['layout_id'] == lid
        te_pack = te_ly_pack[lid]
        corr = correction_fn(te_pack, test_preds[mask].mean())
        preds[mask] = np.clip(preds[mask] + corr, 0, None)
        n = mask.sum()
        total_rows += n
        print(f'  {lid}: pack={te_pack:.4f}  original={test_preds[mask].mean():.2f}  correction={corr:+.2f}  new={preds[mask].mean():.2f}')
    print(f'  Total corrected rows: {total_rows} ({100*total_rows/len(preds):.1f}%)')
    print(f'  oracle_test_mean: {test_preds.mean():.3f}  corrected_mean: {preds.mean():.3f}  delta={preds.mean()-test_preds.mean():+.3f}')
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = preds
    # OOF unchanged (no training layouts in correction range)
    fname = f'submission_{name}_OOF{mae(oracle_oof):.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'  Saved: {fname}')
    return preds

beyond_lids = list(beyond.sort_values(ascending=False).index)

# Approach 1: Conservative — use WH_217's training residual (+19.32)
print('\nApproach 1: Conservative (correction = WH_217 training residual = +19.32)')
make_targeted_sub(oracle_test.copy(), lambda pack, pred: wh217_resid, 'pack_conservative', beyond_lids)

# Approach 2: Regression extrapolation
print('\nApproach 2: Regression extrapolation (linear fit beyond training max)')
make_targeted_sub(oracle_test.copy(), lambda pack, pred: beta * pack + intercept, 'pack_regression', beyond_lids)

# Approach 3: Moderate — correct to match WH_217's TRUE y value (extrapolated)
# WH_217 true_y = 42.53, oracle = 23.22, ratio = 1.832
# Apply same ratio to beyond-range layouts
print('\nApproach 3: Multiplicative scaling (oracle × 1.832, matching WH_217 over/under ratio)')
wh217_oracle_mean = train_raw.loc[train_raw['layout_id']=='WH_217', 'oracle_pred'].mean()
wh217_y_mean = train_raw.loc[train_raw['layout_id']=='WH_217', 'avg_delay_minutes_next_30m'].mean()
ratio = wh217_y_mean / wh217_oracle_mean
print(f'  WH_217 ratio (true/oracle): {ratio:.3f}')
make_targeted_sub(oracle_test.copy(), lambda pack, pred: pred * (ratio - 1), 'pack_multiplicative', beyond_lids)

# Approach 4: High-pack mean residual (all layouts with pack > 0.80)
print(f'\nApproach 4: Mean residual of all high-pack layouts (pack>0.80): {mean_hi_resid:.2f}')
make_targeted_sub(oracle_test.copy(), lambda pack, pred: mean_hi_resid, 'pack_hipack_mean', beyond_lids)

print('\nDone.')
