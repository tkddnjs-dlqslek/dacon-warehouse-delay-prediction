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

xgb_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')

oracle_oof  = np.clip(0.64*fixed_oof + 0.12*xgb_o_oof + 0.16*lv2_o_oof + 0.08*rem_o_oof, 0, None)

mae = lambda p, idx=None: float(np.mean(np.abs(np.clip(p,0,None)[idx] - y_true[idx]))) if idx is not None else float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# Compute per-layout inflow stats
train_raw['ly_inflow_mean'] = train_raw.groupby('layout_id')['order_inflow_15m'].transform('mean')

# Inflow percentiles for training layouts
ly_inflow = train_raw.groupby('layout_id')['order_inflow_15m'].mean()
print('Training layout inflow distribution:')
for p in [25, 50, 75, 90, 95]:
    print(f'  p{p}: {np.percentile(ly_inflow, p):.1f}')
print(f'  max: {ly_inflow.max():.1f}')
print(f'\nTest layout inflow stats:')
te_ly_inflow = test_raw.groupby('layout_id')['order_inflow_15m'].mean()
print(f'  mean={te_ly_inflow.mean():.1f}, max={te_ly_inflow.max():.1f}')
print(f'  fraction of test layouts above train max ({ly_inflow.max():.1f}): {(te_ly_inflow > ly_inflow.max()).mean():.1%}')
print(f'  fraction of test layouts above train p90 ({np.percentile(ly_inflow,90):.1f}): {(te_ly_inflow > np.percentile(ly_inflow,90)).mean():.1%}')

# KEY ANALYSIS: For high-inflow training layouts, which w_fixed gives best MAE?
print('\n--- w_fixed sweep by training inflow quantile ---')
inflow_thresholds = [50, 60, 70, 80, 90, 95]

for w_fixed in [0.60, 0.64, 0.70, 0.75, 0.80, 0.90, 1.00]:
    w_rem = 1.0 - w_fixed
    w_xgb = 0.12 * w_rem / 0.36
    w_lv2 = 0.16 * w_rem / 0.36
    w_rem2 = 0.08 * w_rem / 0.36
    oof_ = np.clip(w_fixed*fixed_oof + w_xgb*xgb_o_oof + w_lv2*lv2_o_oof + w_rem2*rem_o_oof, 0, None)
    row = [f'{w_fixed:.2f}']
    row.append(f'{mae(oof_):.5f}')
    for p in [80, 90]:
        thresh = np.percentile(train_raw['ly_inflow_mean'], p)
        hi_idx = np.where(train_raw['ly_inflow_mean'] >= thresh)[0]
        row.append(f'{mae(oof_, hi_idx):.5f}')
    print(f'  w_fixed={"/".join(row[:1])}  ALL={row[1]}  top20%={row[2]}  top10%={row[3]}')

# GroupKFold validation layout analysis
print('\n--- Validation layout MAE split by inflow level ---')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    val_ly_inflow = train_raw['ly_inflow_mean'].values[vs]
    hi_idx_local = vs[val_ly_inflow >= np.percentile(train_raw['ly_inflow_mean'], 75)]
    lo_idx_local = vs[val_ly_inflow < np.percentile(train_raw['ly_inflow_mean'], 25)]

    oof_64 = np.clip(0.64*fixed_oof + 0.12*xgb_o_oof + 0.16*lv2_o_oof + 0.08*rem_o_oof, 0, None)
    oof_80 = np.clip(0.80*fixed_oof + 0.10*xgb_o_oof + 0.13*lv2_o_oof + 0.07*rem_o_oof, 0, None)

    if len(hi_idx_local) > 0:
        print(f'  Fold {fi+1}: hi_mae(w=0.64)={mae(oof_64, hi_idx_local):.5f}  hi_mae(w=0.80)={mae(oof_80, hi_idx_local):.5f}  lo_mae(w=0.64)={mae(oof_64, lo_idx_local):.5f}  lo_mae(w=0.80)={mae(oof_80, lo_idx_local):.5f}')
    else:
        print(f'  Fold {fi+1}: no high-inflow validation layouts')

# Per-layout MAE analysis for oracle_NEW
print('\n--- Per-layout analysis: inflow vs oracle residual ---')
train_raw['pred'] = oracle_oof
train_raw['residual'] = y_true - oracle_oof
ly_stats = train_raw.groupby('layout_id').agg(
    inflow_mean=('order_inflow_15m','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('pred','mean'),
    residual_mean=('residual','mean'),
    n=('residual','count')
).reset_index()
ly_stats['abs_residual'] = ly_stats['residual_mean'].abs()

corr_inflow_res = np.corrcoef(ly_stats['inflow_mean'], ly_stats['residual_mean'])[0,1]
print(f'  corr(layout_inflow, layout_residual): {corr_inflow_res:.4f}')
print(f'  (positive = higher inflow → oracle underpredicts)')

print('\n  High-inflow training layouts (top 15 by inflow):')
for _, row in ly_stats.sort_values('inflow_mean', ascending=False).head(15).iterrows():
    print(f"    {row['layout_id']}: inflow={row['inflow_mean']:.1f}  y_mean={row['y_mean']:.2f}  pred_mean={row['pred_mean']:.2f}  resid={row['residual_mean']:+.2f}")

print('\n  Low-inflow training layouts (bottom 10 by inflow):')
for _, row in ly_stats.sort_values('inflow_mean').head(10).iterrows():
    print(f"    {row['layout_id']}: inflow={row['inflow_mean']:.1f}  y_mean={row['y_mean']:.2f}  pred_mean={row['pred_mean']:.2f}  resid={row['residual_mean']:+.2f}")

# Check if residual correlates with inflow for validation-only rows
print('\n--- Validation-only layout residuals ---')
all_val_residuals = []
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    # For validation, oracle_oof was computed via the model that did NOT see these layouts
    # But fixed_oof uses mega33 which DID see these layouts
    # So residuals may be different from "true" out-of-sample
    pass

# Actually, oracle_NEW uses mega33 DIRECTLY (not retrained), so fold "validation" is in-sample for mega33
# The true OOF residual is confounded
# Instead, let's look at: for layouts in fold i validation, what's the relationship between
# their ly_inflow and the residual?

ly_stats_val = []
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    val_df = train_raw.iloc[vs].copy()
    val_df['oof_pred'] = oracle_oof[vs]
    val_df['residual'] = y_true[vs] - oracle_oof[vs]
    ly_gr = val_df.groupby('layout_id').agg(
        inflow_mean=('order_inflow_15m','mean'),
        residual_mean=('residual','mean')
    ).reset_index()
    ly_gr['fold'] = fi + 1
    ly_stats_val.append(ly_gr)

ly_val_all = pd.concat(ly_stats_val, ignore_index=True)
corr = np.corrcoef(ly_val_all['inflow_mean'], ly_val_all['residual_mean'])[0,1]
print(f'  corr(layout_inflow, OOF_residual): {corr:.4f}')

# Regression: residual ~ inflow (layout-level)
from numpy.polynomial import polynomial as P
x = ly_val_all['inflow_mean'].values
y = ly_val_all['residual_mean'].values
m, b = np.polyfit(x, y, 1)
print(f'  Linear fit: residual = {m:.4f} × inflow + {b:.4f}')
print(f'  At train mean inflow (94.6): predicted residual = {m*94.6+b:.3f}')
print(f'  At test mean inflow (131.7): predicted residual = {m*131.7+b:.3f}')
print(f'  Expected test correction needed: +{m*(131.7-94.6):.3f}')
print('Done.')
