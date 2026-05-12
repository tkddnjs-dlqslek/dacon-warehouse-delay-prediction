import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, glob

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

mae = lambda p, idx=None: float(np.mean(np.abs(np.clip(p,0,None)[idx]-y_true[idx]))) if idx is not None else float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# Load all FIXED components
mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]

fixed_oof  = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fixed_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test

xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_oof  = np.clip(0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

print(f'oracle_NEW: OOF={mae(oracle_oof):.5f}  test_mean={oracle_test.mean():.3f}')

# Per-component analysis
print('\n=== FIXED component analysis ===')
comps = {
    'mega33_oof': mega33_oof, 'rank_adj_oof': rank_oof,
    'iter_r1': r1_oof, 'iter_r2': r2_oof, 'iter_r3': r3_oof
}
comps_test = {
    'mega33': mega33_test, 'rank_adj': rank_test,
    'iter_r1': r1_test, 'iter_r2': r2_test, 'iter_r3': r3_test
}
for name, p in comps.items():
    tc = comps_test.get(name.replace('_oof',''), None)
    tm = f'{np.clip(tc,0,None).mean():.3f}' if tc is not None else 'N/A'
    corr = np.corrcoef(oracle_oof, np.clip(p,0,None))[0,1]
    print(f'  {name:20s}: OOF={mae(np.clip(p,0,None)):.5f}  test_mean={tm:>8s}  corr_with_oracle={corr:.4f}')

# Check if iter_pseudo rounds change over iterations
print('\n=== Iter_pseudo improvement over rounds ===')
for i, (rn, rt) in enumerate([(r1_oof,r1_test),(r2_oof,r2_test),(r3_oof,r3_test)], 1):
    corr = np.corrcoef(oracle_oof, np.clip(rn,0,None))[0,1]
    print(f'  Round {i}: OOF={mae(np.clip(rn,0,None)):.5f}  test_mean={np.clip(rt,0,None).mean():.3f}  corr={corr:.4f}')

# Can we do round 4 iter_pseudo?
print('\n=== Iter_pseudo round 4 candidate analysis ===')
# Round 3 test predictions → use as pseudo labels for test
# Then retrain on train+test with test pseudo labels
# This is what iter_pseudo already does
# Check if round 3 test_mean > round 2 test_mean
print(f'  r1 test_mean: {np.clip(r1_test,0,None).mean():.3f}')
print(f'  r2 test_mean: {np.clip(r2_test,0,None).mean():.3f}')
print(f'  r3 test_mean: {np.clip(r3_test,0,None).mean():.3f}')
print(f'  oracle_NEW test_mean: {oracle_test.mean():.3f}')
r1r = np.corrcoef(r1_oof, r2_oof)[0,1]
r2r = np.corrcoef(r2_oof, r3_oof)[0,1]
r1r3 = np.corrcoef(r1_oof, r3_oof)[0,1]
print(f'  corr(r1,r2)={r1r:.4f}  corr(r2,r3)={r2r:.4f}  corr(r1,r3)={r1r3:.4f}')

# Blend FIXED with different iter_pseudo weights
print('\n=== iter_pseudo weight sweep in FIXED ===')
for extra_r3_w in [-0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.15]:
    # Increase r3 weight, decrease mega33 proportionally
    w_m = fw['mega33'] - extra_r3_w
    w_r3 = fw['iter_r3'] + extra_r3_w
    f_oof = w_m*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + w_r3*r3_oof
    f_test = w_m*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + w_r3*r3_test
    o_oof = np.clip(0.64*f_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
    o_test = np.clip(0.64*f_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)
    print(f'  extra_r3={extra_r3_w:+.2f}: OOF={mae(o_oof):.5f}  delta={mae(o_oof)-mae(oracle_oof):+.6f}  test_mean={o_test.mean():.3f}')

# Look at r3 specifically: what does it predict differently?
print('\n=== r3 vs mega33 on high-pack layouts ===')
train_raw['r3_oof'] = r3_oof
train_raw['mega33_oof'] = mega33_oof
train_raw['oracle_pred'] = oracle_oof
ly_stats = train_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    r3_mean=('r3_oof','mean'),
    mega33_mean=('mega33_oof','mean'),
    oracle_mean=('oracle_pred','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean')
).reset_index()
ly_stats['residual'] = ly_stats['y_mean'] - ly_stats['oracle_mean']
hi_pack = ly_stats[ly_stats['pack_mean'] > 0.75].sort_values('pack_mean', ascending=False)
print(f'{"layout_id":12s}  {"pack":6s}  {"r3_mean":9s}  {"mega33":9s}  {"r3-mega33":10s}  {"y_mean":8s}  {"resid":8s}')
for _, r in hi_pack.iterrows():
    print(f'{r["layout_id"]:12s}  {r["pack_mean"]:.4f}  {r["r3_mean"]:9.3f}  {r["mega33_mean"]:9.3f}  {r["r3_mean"]-r["mega33_mean"]:+10.3f}  {r["y_mean"]:8.2f}  {r["residual"]:+8.2f}')

# R3 test high-pack layout predictions
test_raw['r3_test'] = r3_test
test_raw['mega33_test'] = mega33_test
test_raw['oracle_pred'] = oracle_test
te_stats = test_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    r3_mean=('r3_test','mean'),
    mega33_mean=('mega33_test','mean'),
    oracle_mean=('oracle_pred','mean')
).reset_index()
te_hi = te_stats[te_stats['pack_mean'] > 0.75].sort_values('pack_mean', ascending=False)
print('\nTest high-pack layouts:')
for _, r in te_hi.iterrows():
    print(f'{r["layout_id"]:12s}  pack={r["pack_mean"]:.4f}  r3_mean={r["r3_mean"]:.3f}  mega33={r["mega33_mean"]:.3f}  r3-mega33={r["r3_mean"]-r["mega33_mean"]:+.3f}  oracle={r["oracle_mean"]:.2f}')

print('\nDone.')
