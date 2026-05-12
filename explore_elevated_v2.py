import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

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

sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)
dual_mae = mae_fn(dual_o)
print(f"Base: dual={dual_mae:.5f}  triple={trip_mae:.5f}")

inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values; test_inflow=test_raw[inflow_col].values
residual=y_true-dual_o

# ---- Build bin residuals using PANDAS (NaN-safe) ----
bins=[0,10,20,30,40,50,60,70,80,100,120,150,200,250,300,500,1000]
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
test_resid=inflow_to_resid(test_inflow)

# NaN-safe layout inflow (use pandas)
layout_train_inflow_mean = train_raw.groupby('layout_id')[inflow_col].mean()  # pd.Series, NaN-skip
layout_train_inflow_std  = train_raw.groupby('layout_id')[inflow_col].std()
layout_train_residual    = pd.Series(residual, index=train_raw.index).groupby(train_raw['layout_id']).mean()

test_seen = test_raw[seen_mask].copy().reset_index(drop=True)
test_inflow_seen = test_inflow[seen_mask]
test_seen['inflow'] = test_inflow_seen
layout_test_inflow_mean = test_seen.groupby('layout_id')['inflow'].mean()

elevated_lids = ['WH_056', 'WH_091', 'WH_104', 'WH_147', 'WH_150', 'WH_243', 'WH_250', 'WH_299']

# ============================================================
print("\n" + "="*70)
print("Part A: Layout-level inflow correlation with residuals (NaN-safe)")
print("="*70)

ls_df = pd.DataFrame({
    'inflow': layout_train_inflow_mean,
    'residual': layout_train_residual,
}).dropna()
inflows = ls_df['inflow'].values
resids = ls_df['residual'].values
from scipy import stats as sp
r, p = sp.pearsonr(inflows, resids)
r2, p2 = sp.pearsonr(np.log1p(inflows), resids)
print(f"N layouts with valid inflow: {len(ls_df)}")
print(f"Pearson(inflow, residual) = {r:.4f} (p={p:.4f})")
print(f"Pearson(log1p(inflow), residual) = {r2:.4f} (p={p2:.4f})")

coef = np.polyfit(np.log1p(inflows), resids, 1)
print(f"\nLog-linear fit: resid = {coef[0]:.4f}*log1p(inflow) + {coef[1]:.4f}")
for tgt in [50, 80, 100, 130, 160, 172, 200, 250, 300]:
    print(f"  inflow={tgt:3d}: resid_hat = {coef[0]*np.log1p(tgt)+coef[1]:+.3f}")

# ============================================================
print("\n" + "="*70)
print("Part B: Detailed per-layout analysis for elevated test layouts")
print("="*70)

def bin_resid_at(inflow_val):
    for (lo,hi),mr in bin_residuals.items():
        if inflow_val >= lo and inflow_val < hi:
            return mr
    return 0.0

print(f"\n{'Layout':10s}  {'TrainInf':>8s}  {'TestInf':>8s}  {'Elev':>6s}  "
      f"{'TrResid':>8s}  {'BrTrain':>8s}  {'LayOff':>8s}  {'BrTest':>8s}  {'EstTest':>8s}")
print("-"*90)

for lid in elevated_lids:
    tr_inf = layout_train_inflow_mean.get(lid, np.nan)
    te_inf = layout_test_inflow_mean.get(lid, np.nan)
    elev = te_inf - tr_inf if not (np.isnan(te_inf) or np.isnan(tr_inf)) else np.nan
    tr_res = layout_train_residual.get(lid, np.nan)
    br_train = bin_resid_at(tr_inf) if not np.isnan(tr_inf) else np.nan
    lay_off = tr_res - br_train if not np.isnan(br_train) else np.nan
    br_test = bin_resid_at(te_inf) if not np.isnan(te_inf) else np.nan
    est_test = br_test + lay_off if not np.isnan(lay_off) else np.nan
    print(f"  {lid:10s}  {tr_inf:>8.1f}  {te_inf:>8.1f}  {elev:>+6.1f}  "
          f"  {tr_res:>+7.3f}  {br_train:>+8.3f}  {lay_off:>+8.3f}  {br_test:>+8.3f}  {est_test:>+8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part C: Cross-inflow residual analysis using same-layout pairs")
print("="*70)
# Can we use scenarios with different inflow WITHIN the same layout to estimate the inflow effect?
# This is a within-layout analysis: for layouts with high inflow variance,
# how does residual change with inflow?
print(f"\nLayouts with highest within-layout inflow std:")
layout_inflow_stats = train_raw.groupby('layout_id').agg(
    inflow_mean=(inflow_col, 'mean'),
    inflow_std=(inflow_col, 'std'),
    inflow_max=(inflow_col, 'max'),
    inflow_min=(inflow_col, 'min'),
    n=('ID', 'count'),
).dropna(subset=['inflow_mean'])

train_raw_with_pred = train_raw.copy()
train_raw_with_pred['dual_resid'] = residual

# For each layout, compute correlation between scenario inflow and residual
print(f"\n{'Layout':10s}  {'Inflow_mean':>11s}  {'Inflow_std':>10s}  {'Inflow_range':>12s}  {'Within_r':>8s}")
print("-"*65)
within_layout_r = {}
for lid in layout_inflow_stats.nlargest(20, 'inflow_std').index:
    m = train_raw['layout_id'] == lid
    li = train_inflow[m.values]
    lr = residual[m.values]
    valid = ~np.isnan(li)
    if valid.sum() < 10: continue
    r_w, _ = sp.pearsonr(li[valid], lr[valid])
    within_layout_r[lid] = r_w
    print(f"  {lid:10s}  {li[valid].mean():>11.1f}  {li[valid].std():>10.1f}  "
          f"[{li[valid].min():.0f},{li[valid].max():.0f}]{' ':>4s}  {r_w:>+8.4f}")

mean_within_r = np.mean(list(within_layout_r.values()))
print(f"\nMean within-layout r(inflow, residual) = {mean_within_r:.4f}")
print(f"(Positive = higher inflow → model underpredicts more)")

# ============================================================
print("\n" + "="*70)
print("Part D: Quantile analysis — do elevated layouts have larger high-end errors?")
print("="*70)
# Check if the elevated test layouts are in a regime where predictions are high/low
for lid in elevated_lids:
    m = train_raw['layout_id'] == lid
    dual_layout = dual_o[m.values]
    y_layout = y_true[m.values]
    resid_layout = residual[m.values]
    print(f"\n  {lid}:")
    print(f"    Train y_true: mean={y_layout.mean():.2f}  p25={np.percentile(y_layout,25):.2f}  p75={np.percentile(y_layout,75):.2f}  p95={np.percentile(y_layout,95):.2f}")
    print(f"    Train dual_o: mean={dual_layout.mean():.2f}  p75={np.percentile(dual_layout,75):.2f}  p95={np.percentile(dual_layout,95):.2f}")
    print(f"    Residual: mean={resid_layout.mean():.3f}  std={resid_layout.std():.3f}")
    te_m_idx = np.where(seen_mask)[0]
    te_lid_rows = [i for i in te_m_idx if test_raw.loc[i,'layout_id']==lid]
    if len(te_lid_rows) > 0:
        te_pred = rh_trip_t[te_lid_rows]
        te_if = test_inflow[te_lid_rows]
        print(f"    Test rh_trip: mean={te_pred.mean():.2f}  p75={np.percentile(te_pred,75):.2f}  p95={np.percentile(te_pred,95):.2f}")
        print(f"    Test inflow: mean={np.nanmean(te_if):.1f}  p95={np.nanpercentile(te_if,95):.1f}")

# ============================================================
print("\n" + "="*70)
print("Part E: Calibration variant with layout-specific offset")
print("="*70)
# Variant: for elevated seen layouts, use estimated correction = bin_at_test + layout_offset
# instead of generic test_resid
sub_tmpl = pd.read_csv('sample_submission.csv')

alpha_layout_opt = np.zeros(len(test_raw))
alpha_layout_opt[unseen_mask] = 1.5
for lid in elevated_lids:
    te_inf = layout_test_inflow_mean.get(lid, np.nan)
    tr_res = layout_train_residual.get(lid, np.nan)
    tr_inf = layout_train_inflow_mean.get(lid, np.nan)
    if np.isnan(te_inf) or np.isnan(tr_res): continue
    br_train = bin_resid_at(tr_inf)
    lay_off = tr_res - br_train
    br_test = bin_resid_at(te_inf)
    est_test_resid = br_test + lay_off
    # alpha such that alpha * test_resid = est_test_resid
    # For each row in this layout: alpha = est_test_resid / row_test_resid
    lid_rows = [i for i in np.where(seen_mask)[0] if test_raw.loc[i,'layout_id']==lid]
    for i in lid_rows:
        row_resid = test_resid[i]
        if row_resid != 0:
            alpha_layout_opt[i] = est_test_resid / row_resid
        else:
            alpha_layout_opt[i] = 0.0

ct_lo = np.clip(rh_trip_t + alpha_layout_opt*test_resid, 0, None)
print(f"\nLayout-offset calibration: seen={ct_lo[seen_mask].mean():.3f}  unseen={ct_lo[unseen_mask].mean():.3f}")

# Also try simpler: use just est_test_resid directly as additive (alpha=1.0 for elevated, not via test_resid)
alpha_simple = np.zeros(len(test_raw))
alpha_simple[unseen_mask] = 1.5
for lid in elevated_lids:
    lid_rows = [i for i in np.where(seen_mask)[0] if test_raw.loc[i,'layout_id']==lid]
    for i in lid_rows:
        alpha_simple[i] = 0.5  # gentle correction

ct_simple = np.clip(rh_trip_t + alpha_simple*test_resid, 0, None)
print(f"Simple a=0.5 for 8 elevated: seen={ct_simple[seen_mask].mean():.3f}  unseen={ct_simple[unseen_mask].mean():.3f}")

# Print the alpha table for elevated layouts
print(f"\nEstimated test corrections for elevated layouts:")
print(f"{'Layout':10s}  {'LayOff':>8s}  {'BrTest':>8s}  {'EstTestRes':>10s}  {'ImplAlpha':>10s}")
for lid in elevated_lids:
    te_inf = layout_test_inflow_mean.get(lid, np.nan)
    tr_res = layout_train_residual.get(lid, np.nan)
    tr_inf = layout_train_inflow_mean.get(lid, np.nan)
    br_train = bin_resid_at(tr_inf) if not np.isnan(tr_inf) else np.nan
    lay_off = tr_res - br_train if not np.isnan(br_train) else np.nan
    br_test = bin_resid_at(te_inf) if not np.isnan(te_inf) else np.nan
    est_test = br_test + lay_off if not np.isnan(lay_off) else np.nan
    # mean test_resid for this layout
    lid_rows = [i for i in np.where(seen_mask)[0] if test_raw.loc[i,'layout_id']==lid]
    mean_row_resid = float(np.nanmean(test_resid[lid_rows])) if lid_rows else np.nan
    impl_alpha = est_test / mean_row_resid if (mean_row_resid and not np.isnan(est_test)) else np.nan
    print(f"  {lid:10s}  {lay_off:>+8.3f}  {br_test:>+8.3f}  {est_test:>+10.3f}  {impl_alpha:>10.3f}")

# Save the layout-offset calibration
fname_lo = f"FINAL_NEW_layoutOffset_u15_elevOpt_OOF{trip_mae:.5f}.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_lo
sub.to_csv(fname_lo, index=False)
print(f"\nSaved: {fname_lo}")
print(f"  seen={ct_lo[seen_mask].mean():.3f}  unseen={ct_lo[unseen_mask].mean():.3f}")
print("\nDone.")
