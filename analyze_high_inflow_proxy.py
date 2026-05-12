import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Rebuild oracle_NEW OOF
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)

# Fill NaN for inflow
inflow_train_raw = train_raw['order_inflow_15m'].values
inflow_test_raw  = test_raw['order_inflow_15m'].values
train_med = np.nanmedian(inflow_train_raw)
inflow_train = np.where(np.isnan(inflow_train_raw), train_med, inflow_train_raw)
inflow_test  = np.where(np.isnan(inflow_test_raw), train_med, inflow_test_raw)

# ============================================================
# Per-layout: compute layout-level OOF MAE and inflow mean
# ============================================================
print("="*70)
print("Per-layout oracle_NEW OOF calibration vs layout inflow")
print("="*70)
layout_ids = train_raw['layout_id'].values
from collections import defaultdict
lay_stats = defaultdict(lambda: {'n':0, 'y_sum':0, 'pred_sum':0, 'mae_sum':0, 'inflow_sum':0, 'inflow_valid':0})
for i, (lid, yi, poi, infi) in enumerate(zip(layout_ids, y_true, fw4_oo, inflow_train)):
    lay_stats[lid]['n'] += 1
    lay_stats[lid]['y_sum'] += yi
    lay_stats[lid]['pred_sum'] += poi
    lay_stats[lid]['mae_sum'] += abs(poi - yi)
    lay_stats[lid]['inflow_sum'] += infi
    lay_stats[lid]['inflow_valid'] += 1

lay_df = []
for lid, s in lay_stats.items():
    n = s['n']
    ym = s['y_sum']/n
    pm = s['pred_sum']/n
    mae = s['mae_sum']/n
    inflow_m = s['inflow_sum']/s['inflow_valid'] if s['inflow_valid'] > 0 else 0
    lay_df.append({'layout_id':lid, 'n':n, 'y_mean':ym, 'pred_mean':pm, 'mae':mae,
                   'resid':pm-ym, 'inflow_mean':inflow_m})
lay_df = pd.DataFrame(lay_df)

# Bin layouts by average inflow
inflow_quartiles = np.percentile(lay_df['inflow_mean'], [25, 50, 75])
print(f"\nLayout inflow quartiles: Q1={inflow_quartiles[0]:.1f}  Q2={inflow_quartiles[1]:.1f}  Q3={inflow_quartiles[2]:.1f}")

for q_lo, q_hi, label in [(0, inflow_quartiles[0], 'Q1 low'),
                            (inflow_quartiles[0], inflow_quartiles[1], 'Q2 mid-low'),
                            (inflow_quartiles[1], inflow_quartiles[2], 'Q3 mid-high'),
                            (inflow_quartiles[2], 9999, 'Q4 high')]:
    mask = (lay_df['inflow_mean'] >= q_lo) & (lay_df['inflow_mean'] < q_hi)
    if mask.sum() > 0:
        print(f"  {label:12s}: n_layouts={mask.sum():4d}  y_mean={lay_df[mask]['y_mean'].mean():.3f}  "
              f"resid={lay_df[mask]['resid'].mean():+.3f}  mae={lay_df[mask]['mae'].mean():.3f}  "
              f"inflow={lay_df[mask]['inflow_mean'].mean():.1f}")

# ============================================================
# Key analysis: HIGH-INFLOW seen layouts as proxy for unseen
# ============================================================
print("\n" + "="*70)
print("HIGH-INFLOW seen layouts as proxy for unseen layouts")
print("Unseen test has mean inflow=161. Find training layouts with similar inflow.")
print("="*70)

# Identify high-inflow training layouts (>120 mean inflow, similar to unseen test)
high_inflow_layouts = lay_df[lay_df['inflow_mean'] > 120]['layout_id'].values
low_inflow_layouts  = lay_df[lay_df['inflow_mean'] <= 120]['layout_id'].values
print(f"\nHigh inflow layouts (mean>120): {len(high_inflow_layouts)}")
print(f"Low inflow layouts (mean<=120): {len(low_inflow_layouts)}")

hi_mask_train = np.isin(layout_ids, high_inflow_layouts)
lo_mask_train = np.isin(layout_ids, low_inflow_layouts)
print(f"\nHigh-inflow layout rows: {hi_mask_train.sum()} ({hi_mask_train.mean()*100:.1f}%)")
print(f"Low-inflow layout rows: {lo_mask_train.sum()}")

print(f"\n  High-inflow training rows (proxy unseen):")
print(f"    y_mean = {y_true[hi_mask_train].mean():.3f}")
print(f"    pred_mean = {fw4_oo[hi_mask_train].mean():.3f}")
print(f"    residual = {(fw4_oo[hi_mask_train]-y_true[hi_mask_train]).mean():+.3f}")
mae_hi = np.mean(np.abs(np.clip(fw4_oo[hi_mask_train],0,None) - y_true[hi_mask_train]))
print(f"    MAE = {mae_hi:.5f}")
print(f"  Low-inflow training rows:")
mae_lo = np.mean(np.abs(np.clip(fw4_oo[lo_mask_train],0,None) - y_true[lo_mask_train]))
print(f"    residual = {(fw4_oo[lo_mask_train]-y_true[lo_mask_train]).mean():+.3f}  MAE = {mae_lo:.5f}")

# Thresholds sweep
print(f"\n  Inflow threshold analysis:")
print(f"  {'threshold':12s} {'n_layouts':>10} {'n_rows':>8} {'y_mean':>8} {'pred_mean':>10} {'residual':>9} {'MAE':>8}")
for thr in [80, 100, 120, 140, 160, 180, 200]:
    hi_lays = lay_df[lay_df['inflow_mean'] > thr]['layout_id'].values
    hi_mask = np.isin(layout_ids, hi_lays)
    if hi_mask.sum() > 0:
        ym = y_true[hi_mask].mean()
        pm = fw4_oo[hi_mask].mean()
        rm = pm - ym
        mae_val = np.mean(np.abs(np.clip(fw4_oo[hi_mask],0,None) - y_true[hi_mask]))
        print(f"  >{thr:4d}: n_layouts={len(hi_lays):6d}  n_rows={hi_mask.sum():8d}  y={ym:8.3f}  pred={pm:10.3f}  resid={rm:+9.3f}  MAE={mae_val:.5f}")

# ============================================================
# Most important: compare test unseen prediction distribution
# to high-inflow training prediction distribution
# ============================================================
print("\n" + "="*70)
print("KEY: Test unseen vs high-inflow training OOF prediction comparison")
print("="*70)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
p_hi_train = fw4_oo[hi_mask_train]
y_hi_train = y_true[hi_mask_train]

print("\n  HIGH-INFLOW TRAINING (layout mean inflow > 120):")
print(f"  {'bucket':12s} {'n':>7} {'y_mean':>8} {'pred':>8} {'resid':>8} {'mae':>8}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_hi_train >= lo) & (p_hi_train < hi)
    if mask.sum() > 0:
        ym = y_hi_train[mask].mean()
        pm = p_hi_train[mask].mean()
        rm = pm - ym
        mae = np.mean(np.abs(pm - y_hi_train[mask]))
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y={ym:8.3f}  pred={pm:8.3f}  resid={rm:+8.3f}  mae={mae:8.3f}")

print("\n  TEST UNSEEN:")
p_t_unseen = oracle_new_t[unseen_mask]
print(f"  {'bucket':12s} {'n':>7} {'pred':>8}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_t_unseen >= lo) & (p_t_unseen < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  pred={p_t_unseen[mask].mean():.3f}")

# ============================================================
# Estimate optimal correction for unseen test
# ============================================================
print("\n" + "="*70)
print("ESTIMATED optimal Δ for unseen test based on proxy analysis")
print("="*70)
# Key metric: high-inflow training layouts (proxy unseen):
# overall residual on prediction-bucketed basis
hi_resid_overall = (fw4_oo[hi_mask_train] - y_true[hi_mask_train]).mean()
hi_mae_overall = np.mean(np.abs(fw4_oo[hi_mask_train] - y_true[hi_mask_train]))
all_resid = (fw4_oo - y_true).mean()
print(f"\n  Overall training residual: {all_resid:.3f}")
print(f"  High-inflow training residual: {hi_resid_overall:.3f}")
print(f"  High-inflow training MAE: {hi_mae_overall:.3f}")
print(f"  Implied optimal Δ for unseen: ~{-hi_resid_overall:.2f}")
print()
print("  This is the BEST ESTIMATE of how much to correct oracle_NEW for unseen layouts")
print("  (assuming unseen test ≈ high-inflow training layouts)")

print("\nDone.")
