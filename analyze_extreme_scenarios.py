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
test_raw['oracle_pred'] = oracle_test

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

print(f'oracle_NEW OOF: {mae(oracle_oof):.5f}')

# WH_217: training reference layout
print('\n=== WH_217 (training reference, pack=0.852) ===')
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
sc217 = wh217.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    pack_max=('pack_utilization','max'),
    inflow_mean=('order_inflow_15m','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    y_max=('avg_delay_minutes_next_30m','max'),
    pred_mean=('oracle_pred','mean'),
    n=('oracle_pred','count')
).reset_index()
sc217['residual'] = sc217['y_mean'] - sc217['pred_mean']
print(f'Total scenarios: {len(sc217)}, total rows: {len(wh217)}')
print(f'Layout mean y: {sc217["y_mean"].mean():.2f}  pred: {sc217["pred_mean"].mean():.2f}  resid: {sc217["residual"].mean():+.2f}')

# Break down by pack category
for pack_thresh, label in [(0.99, 'pack=1.0'), (0.90, 'pack 0.9-1.0'), (0.70, 'pack 0.7-0.9'), (0.0, 'pack <0.7')]:
    if pack_thresh == 0.99:
        sc_sub = sc217[sc217['pack_mean'] >= 0.99]
    elif pack_thresh == 0.90:
        sc_sub = sc217[(sc217['pack_mean'] >= 0.90) & (sc217['pack_mean'] < 0.99)]
    elif pack_thresh == 0.70:
        sc_sub = sc217[(sc217['pack_mean'] >= 0.70) & (sc217['pack_mean'] < 0.90)]
    else:
        sc_sub = sc217[sc217['pack_mean'] < 0.70]
    if len(sc_sub) > 0:
        print(f'  {label}: n_sc={len(sc_sub)}  y_mean={sc_sub["y_mean"].mean():.2f}  pred={sc_sub["pred_mean"].mean():.2f}  resid={sc_sub["residual"].mean():+.2f}  y_max_top={sc_sub["y_mean"].max():.2f}')

# Now analyze the 3 extreme test layouts
for wh_id in ['WH_201', 'WH_246', 'WH_283']:
    df = test_raw[test_raw['layout_id']==wh_id].copy()
    sc = df.groupby('scenario_id').agg(
        pack_mean=('pack_utilization','mean'),
        pack_max=('pack_utilization','max'),
        inflow_mean=('order_inflow_15m','mean'),
        pred_mean=('oracle_pred','mean'),
        pred_max=('oracle_pred','max'),
        n=('oracle_pred','count')
    ).reset_index()
    print(f'\n=== {wh_id} (test, pack_mean={df["pack_utilization"].mean():.4f}) ===')
    print(f'Total scenarios: {len(sc)}, total rows: {len(df)}')
    print(f'Layout oracle mean: {df["oracle_pred"].mean():.2f}')
    for pack_thresh, label in [(0.99, 'pack=1.0'), (0.90, 'pack 0.9-1.0'), (0.70, 'pack 0.7-0.9'), (0.0, 'pack <0.7')]:
        if pack_thresh == 0.99:
            sc_sub = sc[sc['pack_mean'] >= 0.99]
        elif pack_thresh == 0.90:
            sc_sub = sc[(sc['pack_mean'] >= 0.90) & (sc['pack_mean'] < 0.99)]
        elif pack_thresh == 0.70:
            sc_sub = sc[(sc['pack_mean'] >= 0.70) & (sc['pack_mean'] < 0.90)]
        else:
            sc_sub = sc[sc['pack_mean'] < 0.70]
        if len(sc_sub) > 0:
            print(f'  {label}: n_sc={len(sc_sub)}  pred={sc_sub["pred_mean"].mean():.2f}  pred_max_row={sc_sub["pred_max"].max():.2f}')

# Extrapolate: for WH_217's pack=1.0 scenarios, what's the relationship between inflow and y?
print('\n=== WH_217 pack=1.0 scenario: inflow vs TRUE y ===')
sc217_hi = sc217[sc217['pack_mean'] >= 0.99].sort_values('y_mean', ascending=False)
for _, r in sc217_hi.iterrows():
    print(f'  inflow={r["inflow_mean"]:.1f}  y_mean={r["y_mean"]:.1f}  pred={r["pred_mean"]:.1f}  resid={r["residual"]:+.1f}')

# Estimate correction for WH_201's pack=1.0 scenarios
print('\n=== Extrapolated corrections for WH_201 pack=1.0 scenarios ===')
wh201 = test_raw[test_raw['layout_id']=='WH_201'].copy()
sc201_hi = wh201[wh201.groupby('scenario_id')['pack_utilization'].transform('mean') >= 0.99]
sc201_hi_stats = sc201_hi.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    pred_mean=('oracle_pred','mean'),
).reset_index()

# Fit WH_217 pack=1.0: y ~ inflow
x_ref = sc217_hi['inflow_mean'].values
y_ref = sc217_hi['y_mean'].values
m_ref, b_ref = np.polyfit(x_ref, y_ref, 1)
print(f'WH_217 pack=1.0 regression: y = {m_ref:.3f} × inflow + {b_ref:.2f}')
for _, r in sc201_hi_stats.sort_values('pred_mean', ascending=False).iterrows():
    extrapolated_y = m_ref * r['inflow_mean'] + b_ref
    print(f'  SC {r["scenario_id"][-4:]}: inflow={r["inflow_mean"]:.1f}  oracle_pred={r["pred_mean"]:.2f}  extrapolated_y={extrapolated_y:.2f}  correction={extrapolated_y-r["pred_mean"]:+.2f}')

# Generate optimistic correction submission
print('\n=== Generating scenario-aware targeted correction ===')
preds = oracle_test.copy()
for wh_id in ['WH_201', 'WH_246', 'WH_283']:
    df = test_raw[test_raw['layout_id']==wh_id].copy()
    sc_stats = df.groupby('scenario_id')['pack_utilization'].mean()
    sc_oracle = df.groupby('scenario_id')['oracle_pred'].mean()
    n_sc = len(sc_stats)
    n_hi_sc = (sc_stats >= 0.99).sum()

    # WH_217 pack=1.0 mean values
    wh217_hi_y = sc217_hi['y_mean'].mean()
    wh217_hi_pred = sc217_hi['pred_mean'].mean()

    layout_correction_estimate = (n_hi_sc / n_sc) * (wh217_hi_y - wh217_hi_pred)
    for_all_rows = layout_correction_estimate
    mask = test_raw['layout_id'] == wh_id
    preds[mask] = np.clip(oracle_test[mask] + for_all_rows, 0, None)
    print(f'{wh_id}: n_sc={n_sc}  n_pack1={n_hi_sc}  estimated_correction={for_all_rows:+.2f}  original={oracle_test[mask].mean():.2f}  new={preds[mask].mean():.2f}')

sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = preds
oof_val = mae(oracle_oof)
fname = f'submission_pack_scenario_aware_OOF{oof_val:.5f}.csv'
sub.to_csv(fname, index=False)
print(f'test_mean: {preds.mean():.3f}  delta: {preds.mean()-oracle_test.mean():+.3f}')
print(f'Saved: {fname}')

print('\nDone.')
