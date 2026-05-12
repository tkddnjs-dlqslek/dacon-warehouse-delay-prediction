import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold

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

# Layout-level statistics
LOAD_COLS = ['order_inflow_15m', 'congestion_score', 'pack_utilization',
             'robot_active', 'fault_count_15m', 'robot_idle',
             'blocked_path_15m', 'outbound_truck_wait_min']
agg_fns = {}
for col in LOAD_COLS:
    if col in train_raw.columns:
        agg_fns[f'ly_{col}_mean'] = (col, 'mean')
        agg_fns[f'ly_{col}_std']  = (col, 'std')
        agg_fns[f'ly_{col}_max']  = (col, 'max')
        agg_fns[f'ly_{col}_p90']  = (col, lambda x: x.quantile(0.9))

ly_stats = train_raw.groupby('layout_id').agg(
    inflow_mean=('order_inflow_15m', 'mean'),
    inflow_max=('order_inflow_15m', 'max'),
    inflow_std=('order_inflow_15m', 'std'),
    congestion_mean=('congestion_score', 'mean'),
    congestion_max=('congestion_score', 'max'),
    congestion_std=('congestion_score', 'std'),
    pack_mean=('pack_utilization', 'mean'),
    pack_max=('pack_utilization', 'max'),
    pack_std=('pack_utilization', 'std'),
    robot_mean=('robot_active', 'mean'),
    robot_max=('robot_active', 'max'),
    fault_mean=('fault_count_15m', 'mean'),
    blocked_mean=('blocked_path_15m', 'mean'),
    idle_mean=('robot_idle', 'mean'),
    y_mean=('avg_delay_minutes_next_30m', 'mean'),
    y_max=('avg_delay_minutes_next_30m', 'max'),
    n=('avg_delay_minutes_next_30m', 'count')
).reset_index()
train_raw['oracle_pred'] = oracle_oof
pred_by_layout = train_raw.groupby('layout_id')['oracle_pred'].mean().reset_index()
pred_by_layout.columns = ['layout_id','pred_mean']
ly_stats = ly_stats.merge(pred_by_layout, on='layout_id')
ly_stats['residual'] = ly_stats['y_mean'] - ly_stats['pred_mean']

# Feature correlation with residuals
feat_cols = ['inflow_mean', 'inflow_max', 'inflow_std', 'congestion_mean', 'congestion_max',
             'congestion_std', 'pack_mean', 'pack_max', 'pack_std',
             'robot_mean', 'robot_max', 'fault_mean', 'blocked_mean', 'idle_mean',
             'y_mean', 'pred_mean']
print('Layout-level feature correlations with residual:')
for col in feat_cols:
    c = np.corrcoef(ly_stats[col], ly_stats['residual'])[0,1]
    if abs(c) > 0.1:
        print(f'  {col:30s}: corr={c:+.4f}')

# Interaction features
ly_stats['inflow_x_congestion'] = ly_stats['inflow_mean'] * ly_stats['congestion_mean']
ly_stats['inflow_x_pack'] = ly_stats['inflow_mean'] * ly_stats['pack_mean']
ly_stats['inflow_x_fault'] = ly_stats['inflow_mean'] * ly_stats['fault_mean']
ly_stats['congestion_x_pack'] = ly_stats['congestion_mean'] * ly_stats['pack_mean']
ly_stats['inflow_x_blocked'] = ly_stats['inflow_mean'] * ly_stats['blocked_mean']
ly_stats['cong_per_inflow'] = ly_stats['congestion_mean'] / (ly_stats['inflow_mean'] + 1)
ly_stats['pack_per_inflow'] = ly_stats['pack_mean'] / (ly_stats['inflow_mean'] + 0.01)

print('\nInteraction feature correlations with residual:')
for col in ['inflow_x_congestion','inflow_x_pack','inflow_x_fault','congestion_x_pack','inflow_x_blocked','cong_per_inflow','pack_per_inflow']:
    c = np.corrcoef(ly_stats[col], ly_stats['residual'])[0,1]
    print(f'  {col:30s}: corr={c:+.4f}')

# Focus on high-residual layouts
print('\n--- High-residual layouts (top 10 by abs_residual) ---')
ly_stats['abs_residual'] = ly_stats['residual'].abs()
for _, row in ly_stats.sort_values('abs_residual', ascending=False).head(15).iterrows():
    print(f"  {row['layout_id']}: resid={row['residual']:+.2f}  inflow={row['inflow_mean']:.1f}  cong={row['congestion_mean']:.2f}  pack={row['pack_mean']:.3f}  robot={row['robot_mean']:.1f}  blocked={row['blocked_mean']:.3f}")

# Comparison: WH_115 vs WH_294
print('\n--- Key comparison: WH_115 (resid=+17.62) vs WH_294 (resid=+0.12) ---')
for wh in ['WH_115', 'WH_294']:
    r = ly_stats[ly_stats['layout_id']==wh].iloc[0]
    print(f"  {wh}: inflow={r['inflow_mean']:.1f}  cong={r['congestion_mean']:.2f}  pack={r['pack_mean']:.3f}  robot={r['robot_mean']:.1f}  fault={r['fault_mean']:.3f}  blocked={r['blocked_mean']:.3f}  idle={r['idle_mean']:.1f}  y_mean={r['y_mean']:.2f}  pred={r['pred_mean']:.2f}  resid={r['residual']:+.2f}")

# Also WH_140, WH_270
for wh in ['WH_140', 'WH_270', 'WH_066']:
    if wh in ly_stats['layout_id'].values:
        r = ly_stats[ly_stats['layout_id']==wh].iloc[0]
        print(f"  {wh}: inflow={r['inflow_mean']:.1f}  cong={r['congestion_mean']:.2f}  pack={r['pack_mean']:.3f}  robot={r['robot_mean']:.1f}  fault={r['fault_mean']:.3f}  blocked={r['blocked_mean']:.3f}  idle={r['idle_mean']:.1f}  y_mean={r['y_mean']:.2f}  pred={r['pred_mean']:.2f}  resid={r['residual']:+.2f}")

# Check test layouts
print('\n--- Test layout stats ---')
te_ly = test_raw.groupby('layout_id').agg(
    inflow_mean=('order_inflow_15m', 'mean'),
    congestion_mean=('congestion_score', 'mean'),
    pack_mean=('pack_utilization', 'mean'),
    robot_mean=('robot_active', 'mean'),
    fault_mean=('fault_count_15m', 'mean'),
    blocked_mean=('blocked_path_15m', 'mean'),
    idle_mean=('robot_idle', 'mean'),
).reset_index()
te_ly['inflow_x_pack'] = te_ly['inflow_mean'] * te_ly['pack_mean']

test_ls2 = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos2 = {row['ID']:i for i,row in test_ls2.iterrows()}
te_id2 = [te_ls_pos2[i] for i in test_raw['ID'].values]
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
xgb_o_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o_test = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o_test = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)
test_raw['oracle_pred'] = oracle_test
te_pred = test_raw.groupby('layout_id')['oracle_pred'].mean().reset_index()
te_pred.columns = ['layout_id','oracle_pred']
te_ly = te_ly.merge(te_pred, on='layout_id')

print('\nTop 15 test layouts by inflow:')
for _, r in te_ly.sort_values('inflow_mean', ascending=False).head(15).iterrows():
    print(f"  {r['layout_id']}: inflow={r['inflow_mean']:.1f}  cong={r['congestion_mean']:.2f}  pack={r['pack_mean']:.3f}  robot={r['robot_mean']:.1f}  fault={r['fault_mean']:.3f}  blocked={r['blocked_mean']:.3f}  oracle_pred={r['oracle_pred']:.2f}")

# Which test layouts look like WH_115 (high inflow + high pack/fault) ?
print('\nTest layouts similar to high-residual training pattern:')
print('  (high pack_mean × inflow_mean)')
te_ly['inflow_x_congestion'] = te_ly['inflow_mean'] * te_ly['congestion_mean']
te_ly['inflow_x_pack'] = te_ly['inflow_mean'] * te_ly['pack_mean']
for _, r in te_ly.sort_values('inflow_x_congestion', ascending=False).head(15).iterrows():
    # Find closest training analog
    closest = ly_stats.iloc[np.argmin(np.abs(ly_stats['inflow_x_congestion'] - r['inflow_x_congestion']))]
    print(f"  {r['layout_id']}: inflow={r['inflow_mean']:.1f}  cong={r['congestion_mean']:.2f}  pack={r['pack_mean']:.3f}  inflow×cong={r['inflow_x_congestion']:.1f}  oracle_pred={r['oracle_pred']:.2f}  closest_train={closest['layout_id']}(resid={closest['residual']:+.2f})")

print('\nDone.')
