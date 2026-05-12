import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
test_layouts  = set(test_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask     = ~unseen_mask

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof=np.clip(d33['meta_oofs']['cb'][id2],0,None); cb_test=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]; rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
fx=wm*mega+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
bb_o=np.clip(wf*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
bb_t=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t,0,None)
bb_o=np.clip((1-w_cb)*bb_o+w_cb*cb_oof,0,None)
bb_t=np.clip((1-w_cb)*bb_t+w_cb*cb_test,0,None)
fw4_o=np.clip(0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
fw4_t=np.clip(0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t,0,None)
dual_o=fw4_o.copy()
sfw=np.sort(fw4_o); sft=np.sort(fw4_t)
dual_o[fw4_o>=sfw[-2000]]=(1-0.15)*fw4_o[fw4_o>=sfw[-2000]]+0.15*rh_o[fw4_o>=sfw[-2000]]
dual_o[fw4_o>=sfw[-5500]]=(1-0.08)*dual_o[fw4_o>=sfw[-5500]]+0.08*slhm_o[fw4_o>=sfw[-5500]]
dual_o=np.clip(dual_o,0,None)
dual_t=fw4_t.copy()
dual_t[fw4_t>=sft[-2000]]=(1-0.15)*fw4_t[fw4_t>=sft[-2000]]+0.15*rh_t[fw4_t>=sft[-2000]]
dual_t[fw4_t>=sft[-5500]]=(1-0.08)*dual_t[fw4_t>=sft[-5500]]+0.08*slhm_t[fw4_t>=sft[-5500]]
dual_t=np.clip(dual_t,0,None)
dual_mae=lambda p=None: float(np.mean(np.abs(np.clip(p if p is not None else dual_o,0,None)-y_true)))
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# rh triple
sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o_full=dual_o.copy()
rh_trip_o_full[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o_full=np.clip(rh_trip_o_full,0,None)
trip_mae=mae_fn(rh_trip_o_full)
print(f"Base dual MAE: {mae_fn(dual_o):.5f}  rh_triple MAE: {trip_mae:.5f}")

# ============================================================
print("\n" + "="*70)
print("Part 1: High-inflow seen layout analysis (proxy for unseen)")
print("="*70)

inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values

# GroupKFold setup
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true), dtype=int)
for fi, (_, vi) in enumerate(gkf.split(train_raw, y_true, groups)):
    fold_ids[vi] = fi

# For each seen layout, compute its mean inflow and the model's bias
layout_stats = {}
for layout in train_raw['layout_id'].unique():
    mask = train_raw['layout_id'].values == layout
    layout_inflow = np.nanmean(train_inflow[mask])
    layout_residual = float(np.nanmean((y_true - dual_o)[mask]))
    layout_n = mask.sum()
    layout_y = float(np.nanmean(y_true[mask]))
    layout_stats[layout] = {
        'mean_inflow': layout_inflow,
        'mean_residual': layout_residual,
        'n': layout_n,
        'mean_y': layout_y
    }

# Sort by mean_inflow and show top 20
sorted_layouts = sorted(layout_stats.items(), key=lambda x: x[1]['mean_inflow'], reverse=True)
print(f"\nTop 20 seen layouts by mean inflow:")
print(f"  {'Layout':12s}  {'mean_inflow':>11}  {'mean_y':>7}  {'pred_mean':>9}  {'residual':>9}  {'n':>5}")
for lid, stats in sorted_layouts[:20]:
    mask = train_raw['layout_id'].values == lid
    pred_m = float(np.nanmean(dual_o[mask]))
    print(f"  {lid:12s}  {stats['mean_inflow']:>11.1f}  {stats['mean_y']:>7.2f}  {pred_m:>9.2f}  {stats['mean_residual']:>+9.3f}  {stats['n']:>5}")

# Inflow vs residual correlation across layouts
inflows = np.array([v['mean_inflow'] for v in layout_stats.values()])
residuals = np.array([v['mean_residual'] for v in layout_stats.values()])
corr = np.corrcoef(inflows, residuals)[0,1]
print(f"\nCorrelation (layout mean_inflow vs layout mean_residual): {corr:.4f}")

# ============================================================
print("\n" + "="*70)
print("Part 2: GroupKFold LOO — what is the model bias for")
print("       high-inflow layouts when trained without them?")
print("="*70)

# For each fold, check: what is the residual for high-inflow layouts in the held-out fold?
# This tells us the model's extrapolation error for high-inflow layouts
print(f"\n{'Fold':>5}  {'layouts':>7}  {'hi_inflow_layouts':>17}  {'hi_resid':>8}  {'lo_resid':>8}")
hi_threshold = np.percentile(inflows, 80)  # top 20% inflow layouts
print(f"  (high inflow threshold: {hi_threshold:.1f})")

fold_hi_residuals = []
for fi in range(5):
    va_mask = fold_ids == fi
    va_layouts = set(train_raw['layout_id'].values[va_mask])
    hi_layouts = {l for l in va_layouts if layout_stats[l]['mean_inflow'] >= hi_threshold}
    lo_layouts = va_layouts - hi_layouts

    hi_va = va_mask & np.isin(train_raw['layout_id'].values, list(hi_layouts))
    lo_va = va_mask & np.isin(train_raw['layout_id'].values, list(lo_layouts))

    hi_r = float(np.nanmean((y_true-dual_o)[hi_va])) if hi_va.sum()>0 else float('nan')
    lo_r = float(np.nanmean((y_true-dual_o)[lo_va])) if lo_va.sum()>0 else float('nan')
    fold_hi_residuals.append(hi_r)
    print(f"  Fold {fi}: n={va_mask.sum():7}  hi_layouts={len(hi_layouts):17}  hi_resid={hi_r:>+8.3f}  lo_resid={lo_r:>+8.3f}")

mean_hi_resid = float(np.nanmean(fold_hi_residuals))
print(f"\nMean high-inflow layout residual (across folds): {mean_hi_resid:+.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Per-layout LOO bias for SEEN test layouts")
print("="*70)

# Seen test layouts: 50 layouts present in both train and test
seen_test_layouts = set(test_raw['layout_id'].values[seen_mask])
print(f"Seen test layouts count: {len(seen_test_layouts)}")
print(f"\n{'Layout':12s}  {'train_inflow':>12}  {'train_resid':>11}  {'test_pred':>9}  {'test_inflow':>11}")
seen_test_stats = []
for lid in sorted(seen_test_layouts)[:20]:
    tr_mask = train_raw['layout_id'].values == lid
    te_mask = (test_raw['layout_id'].values == lid) & seen_mask
    tr_inf = np.nanmean(train_inflow[tr_mask])
    tr_res = float(np.nanmean((y_true-dual_o)[tr_mask]))
    te_inf = np.nanmean(test_raw[inflow_col].values[te_mask]) if te_mask.sum()>0 else float('nan')
    te_pred = float(np.nanmean(dual_t[te_mask])) if te_mask.sum()>0 else float('nan')
    seen_test_stats.append((lid, tr_inf, tr_res, te_pred, te_inf))
    print(f"  {lid:12s}  {tr_inf:>12.1f}  {tr_res:>+11.3f}  {te_pred:>9.3f}  {te_inf:>11.1f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Estimate optimal alpha for unseen using extrapolation")
print("="*70)

# From training data, what is the relationship between
# (hold-out layout's mean_inflow) and (hold-out layout's mean_residual)?
# This should match what we'd expect for unseen test layouts

x_data = inflows  # layout mean inflows
y_data = residuals  # layout mean residuals

# Fit: residual = a * log1p(mean_inflow) + b
log_x = np.log1p(x_data)
valid = ~(np.isnan(x_data) | np.isnan(y_data))
coefs_layout = np.polyfit(log_x[valid], y_data[valid], 1)
a_lay, b_lay = coefs_layout
print(f"\nLayout-level fit: resid ≈ {a_lay:.4f} * log1p(mean_inflow) + {b_lay:.4f}")

# Train inflow range
print(f"Training layouts mean_inflow range: [{inflows.min():.1f}, {inflows.max():.1f}]")

# Predict for unseen test layout inflow (need to estimate mean_inflow per unseen layout)
unseen_test_layouts = set(test_raw['layout_id'].values[unseen_mask])
test_inflow = test_raw[inflow_col].values
print(f"\nPer unseen-test-layout mean inflow (top 10):")
unseen_layout_stats = []
for lid in sorted(unseen_test_layouts):
    te_mask = test_raw['layout_id'].values == lid
    te_inf = np.nanmean(test_inflow[te_mask])
    unseen_layout_stats.append((lid, te_inf))
unseen_layout_stats.sort(key=lambda x: x[1], reverse=True)
for lid, inf in unseen_layout_stats[:15]:
    expected_resid = a_lay*np.log1p(inf)+b_lay
    print(f"  {lid:12s}  mean_inflow={inf:6.1f}  expected_resid={expected_resid:+.3f}")

# Overall unseen expected residual
unseen_inflows = np.array([s[1] for s in unseen_layout_stats if not np.isnan(s[1])])
expected_resids = a_lay*np.log1p(unseen_inflows)+b_lay
print(f"\nExpected unseen layout residual: {expected_resids.mean():.3f}")
print(f"  Min: {expected_resids.min():.3f}  Max: {expected_resids.max():.3f}")

# Corresponding alpha: expected_resid = alpha * bin_resid_unseen
bin_resid_unseen_mean = 4.006  # from earlier analysis
implied_alpha = expected_resids.mean() / bin_resid_unseen_mean
print(f"\nBin residual for unseen rows (mean): {bin_resid_unseen_mean:.3f}")
print(f"Layout-level expected residual: {expected_resids.mean():.3f}")
print(f"Implied alpha (expected/bin): {implied_alpha:.3f}")
print(f"  → Recommended: apply alpha={implied_alpha:.2f} for unseen rows")

# ============================================================
print("\n" + "="*70)
print("Part 5: Save recommended submission with layout-level implied alpha")
print("="*70)

# Build asymmetric calibration
bins=[0,10,20,30,40,50,60,70,80,100,120,150,200,250,300,500,1000]
residual = y_true - dual_o
test_inflow_arr = test_raw[inflow_col].values
bin_residuals={}
for i in range(len(bins)-1):
    lo,hi=bins[i],bins[i+1]
    mask=(train_inflow>=lo)&(train_inflow<hi)
    if mask.sum()==0: continue
    bin_residuals[(lo,hi)]=float(np.nanmean(residual[mask]))
def inflow_to_resid(arr):
    r=np.zeros(len(arr))
    for (lo,hi),mr in bin_residuals.items(): r[(arr>=lo)&(arr<hi)]=mr
    return r
test_resid_arr = inflow_to_resid(test_inflow_arr)

# Apply recommended alpha for unseen only
alpha_u = implied_alpha
alpha_arr = np.where(unseen_mask, alpha_u, 0.0)
ct_rec = np.clip(rh_trip_t + alpha_arr*test_resid_arr, 0, None)
print(f"\nRecommended alpha_u={alpha_u:.3f}:")
print(f"  test={ct_rec.mean():.3f}  seen={ct_rec[seen_mask].mean():.3f}  unseen={ct_rec[unseen_mask].mean():.3f}")

sub_tmpl = pd.read_csv('sample_submission.csv')
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_rec
fname = f"FINAL_triple_asymLayoutAlpha_u{alpha_u:.2f}_OOF{trip_mae:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"Saved: {fname}")

# Also save a range
for alpha_u in [0.5, 0.75, implied_alpha, 1.0, 1.25, 1.5, 2.0]:
    alpha_arr = np.where(unseen_mask, alpha_u, 0.0)
    ct = np.clip(rh_trip_t + alpha_arr*test_resid_arr, 0, None)
    print(f"  alpha_u={alpha_u:.2f}: test={ct.mean():.3f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
