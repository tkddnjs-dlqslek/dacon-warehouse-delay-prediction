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
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
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
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_oof  = np.clip(0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

train_raw['oracle_pred'] = oracle_oof
tr_ly = train_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('oracle_pred','mean')
).reset_index()
tr_ly['residual'] = tr_ly['y_mean'] - tr_ly['pred_mean']

te_ly = test_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
test_raw['oracle_pred'] = oracle_test
te_pred = test_raw.groupby('layout_id')['oracle_pred'].mean().reset_index()
te_pred.columns = ['layout_id','pred_mean']
te_ly = te_ly.merge(te_pred, on='layout_id')

# High-pack only regression (pack > 0.65)
hi = tr_ly[tr_ly['pack_mean'] > 0.65].copy()
print(f'High-pack training layouts (pack>0.65): n={len(hi)}')
for _, r in hi.sort_values('pack_mean', ascending=False).iterrows():
    print(f'  {r["layout_id"]}: pack={r["pack_mean"]:.4f}  inflow={r["inflow_mean"]:.1f}  y={r["y_mean"]:.2f}  pred={r["pred_mean"]:.2f}  resid={r["residual"]:+.2f}')

# Fit high-pack only regression
if len(hi) > 3:
    x = hi['pack_mean'].values
    y = hi['residual'].values
    m, b = np.polyfit(x, y, 1)
    print(f'\nHigh-pack regression: residual = {m:.2f} * pack + {b:.2f}')
    print(f'  At pack=0.852 (WH_217): {m*0.852+b:.2f}  (true: +19.32)')
    for lid in ['WH_283','WH_201','WH_246']:
        pack = te_ly[te_ly['layout_id']==lid]['pack_mean'].values[0]
        pred = te_ly[te_ly['layout_id']==lid]['pred_mean'].values[0]
        correction = m * pack + b
        print(f'  At pack={pack:.4f} ({lid}): correction = {correction:+.2f}  new_pred = {pred+correction:.2f}')

# More detailed: what's the relationship between pack_mean^2 and residual?
print('\n=== Non-linear pack→residual analysis ===')
x_lin  = tr_ly['pack_mean'].values
x_quad = tr_ly['pack_mean'].values ** 2
y = tr_ly['residual'].values

# Quadratic fit
m2, m1, b = np.polyfit(x_lin, y, 2)
print(f'Quadratic fit: residual = {m2:.2f} × pack² + {m1:.2f} × pack + {b:.2f}')
print(f'  At pack=0.852: {m2*0.852**2 + m1*0.852 + b:.2f}  (true: +19.32)')
for lid in ['WH_283','WH_201','WH_246']:
    pack = te_ly[te_ly['layout_id']==lid]['pack_mean'].values[0]
    pred = te_ly[te_ly['layout_id']==lid]['pred_mean'].values[0]
    correction = m2*pack**2 + m1*pack + b
    print(f'  At pack={pack:.4f} ({lid}): correction = {correction:+.2f}  new_pred = {pred+correction:.2f}')

# Cubic fit (to capture exponential-like behavior)
m3, m2b, m1b, b2 = np.polyfit(x_lin, y, 3)
print(f'\nCubic fit: residual = {m3:.2f}×pack³ + {m2b:.2f}×pack² + {m1b:.2f}×pack + {b2:.2f}')
print(f'  At pack=0.852: {m3*0.852**3 + m2b*0.852**2 + m1b*0.852 + b2:.2f}  (true: +19.32)')
for lid in ['WH_283','WH_201','WH_246']:
    pack = te_ly[te_ly['layout_id']==lid]['pack_mean'].values[0]
    pred = te_ly[te_ly['layout_id']==lid]['pred_mean'].values[0]
    correction = m3*pack**3 + m2b*pack**2 + m1b*pack + b2
    print(f'  At pack={pack:.4f} ({lid}): correction = {correction:+.2f}  new_pred = {pred+correction:.2f}')

# Risk/reward analysis
print('\n=== Risk/Reward analysis ===')
mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'Base OOF: {base_oof:.5f}  (unchanged for all targeted corrections)')
print(f'OOF cannot validate these corrections — beyond training range.')

print('\nExpected LB impact (2.4% of test = 1200 rows):')
for correction, name in [(10,'regression'),(15.52,'hipack_mean'),(19.32,'conservative'),(24,'multiplicative')]:
    if_right = f'{9.7527 - correction*1200/50000:.4f}'  # if correction is exactly right
    print(f'  Correction={correction:5.1f}: if exactly right → LB ~{if_right}  if missed by {correction} units → LB ~{9.7527+correction*1200/50000:.4f}')

# The scenario-level pack analysis for WH_201
print('\n=== WH_201 scenario-level analysis ===')
wh201 = test_raw[test_raw['layout_id']=='WH_201'].copy()
wh201['oracle_pred'] = oracle_test[wh201.index]
sc = wh201.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    pack_max=('pack_utilization','max'),
    inflow_mean=('order_inflow_15m','mean'),
    pred_mean=('oracle_pred','mean'),
    pred_max=('oracle_pred','max'),
).reset_index()
print(sc.sort_values('pred_mean', ascending=False).to_string())

# Similar training scenario for comparison (WH_217 scenarios)
print('\n=== WH_217 (most similar training layout) scenario-level analysis ===')
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
wh217['oracle_pred'] = oracle_oof[wh217.index]
sc217 = wh217.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('oracle_pred','mean'),
).reset_index()
sc217['residual'] = sc217['y_mean'] - sc217['pred_mean']
print(sc217.sort_values('y_mean', ascending=False).head(10).to_string())
print(f'\nWH_217 scenario stats: y_mean range = [{sc217["y_mean"].min():.1f}, {sc217["y_mean"].max():.1f}]')
print(f'                       pred_mean range = [{sc217["pred_mean"].min():.1f}, {sc217["pred_mean"].max():.1f}]')
print(f'                       residual range = [{sc217["residual"].min():.1f}, {sc217["residual"].max():.1f}]')

print('\nDone.')
