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

oof_ly = np.load('results/oracle_seq/oof_seqD_layout_stats.npy')
test_ly = np.load('results/oracle_seq/test_D_layout_stats.npy')

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
oracle_oof  = np.clip(0.64*fixed_oof  + 0.12*xgb_o_oof  + 0.16*lv2_o_oof  + 0.08*rem_o_oof,  0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'oracle_NEW OOF:    {mae(oracle_oof):.5f}')
print(f'layout_stats OOF:  {mae(oof_ly):.5f}  delta={mae(oof_ly)-mae(oracle_oof):+.5f}')
corr = float(np.corrcoef(oracle_oof, oof_ly)[0,1])
print(f'corr(oracle_NEW, layout_stats): {corr:.4f}')

print('\nFold-level MAE:')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    print(f'  Fold {fi+1}: layout_stats={mean_absolute_error(y_true[vs], oof_ly[vs]):.5f}  oracle_NEW={mean_absolute_error(y_true[vs], oracle_oof[vs]):.5f}  delta={mean_absolute_error(y_true[vs], oof_ly[vs])-mean_absolute_error(y_true[vs], oracle_oof[vs]):+.5f}')

print('\nBlend with oracle_NEW:')
for w in [0.05, 0.10, 0.15, 0.20, 0.30]:
    b = (1-w)*oracle_oof + w*oof_ly
    print(f'  w={w:.2f}: OOF={mae(b):.5f}  delta={mae(b)-mae(oracle_oof):+.5f}')

print(f'\nTest pred:')
print(f'  oracle_NEW test mean: {oracle_test.mean():.3f}')
print(f'  layout_stats test mean: {test_ly.mean():.3f} (delta={test_ly.mean()-oracle_test.mean():+.3f})')

test_raw['oracle_pred'] = oracle_test
test_raw['layout_pred'] = test_ly
ly_cmp = test_raw.groupby('layout_id').agg(
    oracle_mean=('oracle_pred','mean'), layout_mean=('layout_pred','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
ly_cmp['delta'] = ly_cmp['layout_mean'] - ly_cmp['oracle_mean']
print('\nHigh-inflow test layouts (top 15):')
for _, row in ly_cmp.sort_values('inflow_mean', ascending=False).head(15).iterrows():
    print(f"  {row['layout_id']}: oracle={row['oracle_mean']:.2f} layout_stats={row['layout_mean']:.2f} delta={row['delta']:+.2f} inflow={row['inflow_mean']:.1f}")
print('\nLow-inflow test layouts (bottom 10):')
for _, row in ly_cmp.sort_values('inflow_mean').head(10).iterrows():
    print(f"  {row['layout_id']}: oracle={row['oracle_mean']:.2f} layout_stats={row['layout_mean']:.2f} delta={row['delta']:+.2f} inflow={row['inflow_mean']:.1f}")
print('Done.')
