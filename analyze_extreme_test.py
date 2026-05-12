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
oracle_oof  = np.clip(0.64*fixed_oof + 0.12*xgb_o_oof + 0.16*lv2_o_oof + 0.08*rem_o_oof, 0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)

# Compute layout stats
def ly_agg(df):
    return df.groupby('layout_id').agg(
        inflow_mean=('order_inflow_15m','mean'),
        inflow_max=('order_inflow_15m','max'),
        cong_mean=('congestion_score','mean'),
        pack_mean=('pack_utilization','mean'),
        pack_max=('pack_utilization','max'),
        robot_mean=('robot_active','mean'),
        fault_mean=('fault_count_15m','mean'),
        blocked_mean=('blocked_path_15m','mean'),
        idle_mean=('robot_idle','mean'),
        n=('order_inflow_15m','count')
    ).reset_index()

tr_ly = ly_agg(train_raw)
te_ly = ly_agg(test_raw)

# Training max values for key features
print('=== Training max values (extrapolation boundary) ===')
for col in ['inflow_mean','inflow_max','pack_mean','pack_max','cong_mean','robot_mean']:
    print(f'  {col}: train_max={tr_ly[col].max():.4f}  test_max={te_ly[col].max():.4f}  ratio={te_ly[col].max()/tr_ly[col].max():.2f}x')

# Feature ranges: training vs test
print('\n=== Feature range comparison ===')
for col in ['inflow_mean','pack_mean','cong_mean','robot_mean']:
    tr_vals = tr_ly[col].values
    te_vals = te_ly[col].values
    n_beyond = (te_vals > tr_vals.max()).sum()
    n_below = (te_vals < tr_vals.min()).sum()
    print(f'  {col}: train=[{tr_vals.min():.3f},{tr_vals.max():.3f}] test=[{te_vals.min():.3f},{te_vals.max():.3f}]  beyond_train_max={n_beyond}  below_train_min={n_below}')

# Identify test layouts beyond ALL training feature ranges
print('\n=== Test layouts beyond training range in multiple features ===')
test_raw['oracle_pred'] = oracle_test
te_pred = test_raw.groupby('layout_id')['oracle_pred'].mean().reset_index()
te_ly = te_ly.merge(te_pred, on='layout_id')
te_ly['beyond_count'] = 0
for col, train_col in [('inflow_mean','inflow_mean'),('pack_mean','pack_mean'),('cong_mean','cong_mean')]:
    te_ly['beyond_count'] += (te_ly[col] > tr_ly[train_col].max()).astype(int)
print(te_ly.sort_values('beyond_count', ascending=False).head(20)[
    ['layout_id','inflow_mean','pack_mean','cong_mean','robot_mean','oracle_pred','beyond_count']
].to_string())

# Training layout residuals
train_raw['oracle_pred'] = oracle_oof
tr_pred = train_raw.groupby('layout_id')['oracle_pred'].mean().reset_index()
tr_y = train_raw.groupby('layout_id')['avg_delay_minutes_next_30m'].mean().reset_index()
tr_ly = tr_ly.merge(tr_pred, on='layout_id').merge(tr_y, on='layout_id')
tr_ly['residual'] = tr_ly['avg_delay_minutes_next_30m'] - tr_ly['oracle_pred']

print('\n=== Training layouts with pack_mean near test extremes ===')
print(f'Training layouts with pack_mean > 0.70:')
hi_pack_tr = tr_ly[tr_ly['pack_mean'] > 0.70].sort_values('pack_mean', ascending=False)
print(hi_pack_tr[['layout_id','pack_mean','inflow_mean','cong_mean','avg_delay_minutes_next_30m','oracle_pred','residual']].to_string())

# What's the oracle_pred for WH_201 (pack=0.936) broken down by scenario?
print('\n=== WH_201 (pack=0.936) scenario analysis ===')
wh201 = test_raw[test_raw['layout_id']=='WH_201'].copy()
wh201['oracle_pred'] = oracle_test[wh201.index]
sc_stats = wh201.groupby('scenario_id').agg(
    n=('oracle_pred','count'),
    pred_mean=('oracle_pred','mean'),
    pred_max=('oracle_pred','max'),
    inflow_mean=('order_inflow_15m','mean'),
    pack_mean=('pack_utilization','mean'),
).reset_index()
print(sc_stats.sort_values('pred_mean', ascending=False).head(10).to_string())
print(f'WH_201 overall: n={len(wh201)}, oracle_mean={wh201["oracle_pred"].mean():.2f}, pack_max={wh201["pack_utilization"].max():.4f}')

# Compare WH_201 to most similar training layout (WH_217: pack=0.852, resid=+19.32)
print('\n=== Most similar training layout to WH_201 ===')
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
wh217['oracle_pred'] = oracle_oof[wh217.index]
print(f'WH_217: n={len(wh217)}, y_mean={wh217["avg_delay_minutes_next_30m"].mean():.2f}, oracle_mean={wh217["oracle_pred"].mean():.2f}, resid={wh217["avg_delay_minutes_next_30m"].mean()-wh217["oracle_pred"].mean():+.2f}')
print(f'  pack_mean={wh217["pack_utilization"].mean():.4f}, inflow_mean={wh217["order_inflow_15m"].mean():.1f}, cong={wh217["congestion_score"].mean():.2f}')

# Scenario-level statistics for WH_217
sc217 = wh217.groupby('scenario_id').agg(
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('oracle_pred','mean'),
    resid_mean=lambda x: x.mean() - wh217.groupby('scenario_id')['oracle_pred'].transform('mean').mean(),
).reset_index()

# High-pack scenario distribution in test
print('\n=== Distribution of pack_max > 0.95 in test layouts ===')
for wh, grp in test_raw.groupby('layout_id'):
    if grp['pack_utilization'].max() > 0.95:
        print(f"  {wh}: pack_max={grp['pack_utilization'].max():.4f}  pack_mean={grp['pack_utilization'].mean():.4f}  inflow_mean={grp['order_inflow_15m'].mean():.1f}  oracle_pred={oracle_test[grp.index].mean():.2f}")

# Key question: for training layouts with pack_mean > 0.8, what are typical TRUE y values?
print('\n=== High-pack training layouts (pack>0.75): true delay stats ===')
hi_pack = tr_ly[tr_ly['pack_mean'] > 0.75].sort_values('pack_mean', ascending=False)
for _, r in hi_pack.iterrows():
    print(f"  {r['layout_id']}: pack={r['pack_mean']:.3f}  inflow={r['inflow_mean']:.1f}  y_mean={r['avg_delay_minutes_next_30m']:.2f}  pred={r['oracle_pred']:.2f}  resid={r['residual']:+.2f}")

print('\nDone.')
