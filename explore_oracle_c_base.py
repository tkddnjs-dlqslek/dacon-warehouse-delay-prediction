import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from scipy.stats import pearsonr

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

# Load oracle components
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')

oracle5_o = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)
oracle5_t = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)

# Load oracle_NEW reference
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Load oracle_C_as_base
c_base_df = pd.read_csv('FINAL_NEW_oracle_C_as_base_OOF_oracle.csv')
c_base_df = c_base_df.set_index('ID').reindex(id_order).reset_index()
c_base_t  = c_base_df['avg_delay_minutes_next_30m'].values
print(f"oracle_C_as_base: seen={c_base_t[seen_mask].mean():.3f}  unseen={c_base_t[unseen_mask].mean():.3f}")
print(f"  r(oracle_NEW)={pearsonr(c_base_t, oracle_new_t)[0]:.4f}  r(oracle_5way)={pearsonr(c_base_t, oracle5_t)[0]:.4f}")

# Check correlations with individual oracle components
for name, arr in [('xgb', xgb_t), ('lv2', lv2_t), ('rem', rem_t), ('xgbc', xgbc_t), ('mono', mono_t)]:
    r_c, _ = pearsonr(c_base_t, np.clip(arr, 0, None))
    print(f"  r(c_base, {name}) = {r_c:.4f}  diff_seen={np.clip(arr,0,None)[seen_mask].mean() - c_base_t[seen_mask].mean():.3f}")

print()

# ============================================================
print("="*70)
print("Individual oracle OOF values")
print("="*70)
for name, oo in [('xgb', xgb_o), ('lv2', lv2_o), ('rem', rem_o), ('xgbc', xgbc_o), ('mono', mono_o), ('5way', oracle5_o)]:
    print(f"  {name}: OOF={mae_fn(np.clip(oo,0,None)):.5f}")

print()

# ============================================================
print("="*70)
print("Per-layout oracle5_oof residual analysis (train set)")
print("="*70)
oracle5_oof = oracle5_o

train_layouts_list = sorted(train_raw['layout_id'].unique())
n_layouts = len(train_layouts_list)
print(f"Total training layouts: {n_layouts}")

layout_stats = []
for lay in train_layouts_list:
    mask = (train_raw['layout_id'] == lay).values
    y_lay = y_true[mask]
    p_lay = np.clip(oracle5_oof[mask], 0, None)
    mae = np.mean(np.abs(p_lay - y_lay))
    resid = p_lay.mean() - y_lay.mean()
    layout_stats.append((lay, mask.sum(), y_lay.mean(), p_lay.mean(), mae, resid))

layout_stats.sort(key=lambda x: x[5], reverse=True)
print(f"\nLayouts sorted by residual (oracle5_oof pred_mean - y_mean):")
print(f"  {'layout':15s} {'n':>5} {'y_mean':>8} {'pred_mean':>10} {'MAE':>8} {'resid':>8}")
for lay, n, ym, pm, mae, resid in layout_stats[:15]:
    print(f"  {lay:15s} {n:5d} {ym:8.3f} {pm:10.3f} {mae:8.4f} {resid:+8.3f}")
print("  ...")
for lay, n, ym, pm, mae, resid in layout_stats[-15:]:
    print(f"  {lay:15s} {n:5d} {ym:8.3f} {pm:10.3f} {mae:8.4f} {resid:+8.3f}")

resids = np.array([x[5] for x in layout_stats])
print(f"\nLayout residual distribution:")
print(f"  mean={resids.mean():.4f}  std={resids.std():.4f}")
print(f"  pct5={np.percentile(resids,5):.3f}  pct25={np.percentile(resids,25):.3f}  median={np.median(resids):.3f}  pct75={np.percentile(resids,75):.3f}  pct95={np.percentile(resids,95):.3f}")
print(f"  layouts underpred (resid<-1): {sum(1 for r in resids if r<-1)}")
print(f"  layouts overpred (resid>+1): {sum(1 for r in resids if r>+1)}")

print()

# ============================================================
print("="*70)
print("Unseen test layout statistics")
print("="*70)
unseen_layouts_test = test_raw[unseen_mask]['layout_id'].unique()
test_raw2 = pd.read_csv('test.csv')
test_raw2['_row_id'] = test_raw2['ID'].str.replace('TEST_','').astype(int)
test_raw2 = test_raw2.sort_values('_row_id').reset_index(drop=True)

unseen_layout_stats = []
for lay in sorted(unseen_layouts_test):
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    oN_m = oracle_new_t[mask].mean()
    o5_m = oracle5_t[mask].mean()
    inflow_m = test_raw2.loc[mask, 'order_inflow_15m'].mean() if 'order_inflow_15m' in test_raw2.columns else float('nan')
    unseen_layout_stats.append((lay, mask.sum(), oN_m, o5_m, inflow_m))

unseen_layout_stats.sort(key=lambda x: x[2], reverse=True)
print(f"\nTop-10 highest oracle_NEW unseen layouts:")
print(f"  {'layout':15s} {'n':>5} {'oN_mean':>9} {'o5_mean':>9} {'inflow_mean':>12}")
for lay, n, oN, o5, inf in unseen_layout_stats[:10]:
    print(f"  {lay:15s} {n:5d} {oN:9.3f} {o5:9.3f} {inf:12.3f}")
print(f"\nBottom-10 lowest oracle_NEW unseen layouts:")
for lay, n, oN, o5, inf in unseen_layout_stats[-10:]:
    print(f"  {lay:15s} {n:5d} {oN:9.3f} {o5:9.3f} {inf:12.3f}")

# Per-layout: what determines delay level? Check correlation with inflow
if 'order_inflow_15m' in test_raw2.columns:
    lay_inflows = np.array([x[4] for x in unseen_layout_stats])
    lay_oN     = np.array([x[2] for x in unseen_layout_stats])
    r_inf_oN, _ = pearsonr(lay_inflows, lay_oN)
    print(f"\nr(inflow, oracle_NEW prediction) across unseen layouts = {r_inf_oN:.4f}")

    unseen_inf = test_raw2.loc[unseen_mask, 'order_inflow_15m']
    seen_inf   = test_raw2.loc[seen_mask, 'order_inflow_15m']
    print(f"\nInflow distribution:")
    print(f"  Unseen test: mean={unseen_inf.mean():.2f}  median={unseen_inf.median():.2f}  std={unseen_inf.std():.2f}")
    print(f"  Seen test:   mean={seen_inf.mean():.2f}  median={seen_inf.median():.2f}  std={seen_inf.std():.2f}")

print("\nDone.")
