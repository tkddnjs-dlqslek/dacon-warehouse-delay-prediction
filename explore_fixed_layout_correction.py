import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from scipy import stats as sp

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

# Layout-level stats
layout_train_inflow_mean = train_raw.groupby('layout_id')[inflow_col].mean()
layout_train_residual    = pd.Series(residual).groupby(train_raw['layout_id'].values).mean()

test_seen = test_raw[seen_mask].copy().reset_index(drop=True)
test_inflow_seen = test_inflow[seen_mask]
test_seen['inflow'] = test_inflow_seen
layout_test_inflow_mean = test_seen.groupby('layout_id')['inflow'].mean()

# Cross-layout log-linear fit for residual
ls_df = pd.DataFrame({
    'inflow': layout_train_inflow_mean,
    'residual': layout_train_residual,
}).dropna()
inflows = ls_df['inflow'].values
resids = ls_df['residual'].values
coef = np.polyfit(np.log1p(inflows), resids, 1)
print(f"Log-linear: resid = {coef[0]:.4f}*log1p(inflow) + {coef[1]:.4f}")

# Bin residuals (for comparison)
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

def bin_resid_at(v):
    for (lo,hi),mr in bin_residuals.items():
        if v >= lo and v < hi: return mr
    return 0.0

# ============================================================
print("\n" + "="*70)
print("Part 1: Unseen layout inflow distribution and fixed corrections")
print("="*70)

# Get unseen layout mean inflows from test data
test_unseen = test_raw[unseen_mask].copy().reset_index(drop=True)
test_inflow_unseen = test_inflow[unseen_mask]
test_unseen['inflow'] = test_inflow_unseen
layout_unseen_inflow = test_unseen.groupby('layout_id')['inflow'].mean()

print(f"\nUnseen layouts (n={layout_unseen_inflow.shape[0]}):")
print(f"  Mean inflow: {layout_unseen_inflow.mean():.1f}  std={layout_unseen_inflow.std():.1f}")
print(f"  Range: [{layout_unseen_inflow.min():.1f}, {layout_unseen_inflow.max():.1f}]")
print(f"\nPer-unseen-layout fixed corrections (log-linear model):")
print(f"  {'Layout':12s}  {'MeanInflow':>10s}  {'FixedCorr':>9s}  {'BinCorr':>8s}")
print("-"*55)
unseen_lids_sorted = layout_unseen_inflow.sort_values().index
for lid in unseen_lids_sorted:
    mif = layout_unseen_inflow[lid]
    fixed_c = coef[0]*np.log1p(mif) + coef[1]
    bin_c = bin_resid_at(mif)
    print(f"  {lid:12s}  {mif:>10.1f}  {fixed_c:>+9.3f}  {bin_c:>+8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 2: Fixed correction approaches for unseen")
print("="*70)
# Approach A: Per-layout fixed offset based on log-linear model applied at layout mean inflow
# (Within-layout variation ignored since r=0.065)

# Compute per-row fixed correction for unseen based on layout mean inflow
fixed_correction_unseen = np.zeros(len(test_raw))
for lid in layout_unseen_inflow.index:
    mif = layout_unseen_inflow[lid]
    corr = max(0, coef[0]*np.log1p(mif) + coef[1])
    lid_rows = np.where((test_raw['layout_id'].values==lid) & unseen_mask)[0]
    fixed_correction_unseen[lid_rows] = corr

# Approach B: Per-layout bin correction at layout mean inflow
bin_correction_unseen = np.zeros(len(test_raw))
for lid in layout_unseen_inflow.index:
    mif = layout_unseen_inflow[lid]
    corr = bin_resid_at(mif)
    lid_rows = np.where((test_raw['layout_id'].values==lid) & unseen_mask)[0]
    bin_correction_unseen[lid_rows] = corr

# Current bin approach (per-row inflow)
per_row_bin = test_resid.copy()

print(f"\nUnseen correction statistics:")
print(f"  Fixed (log-linear at layout mean): mean={fixed_correction_unseen[unseen_mask].mean():.3f}  std={fixed_correction_unseen[unseen_mask].std():.3f}")
print(f"  Bin at layout mean:                mean={bin_correction_unseen[unseen_mask].mean():.3f}  std={bin_correction_unseen[unseen_mask].std():.3f}")
print(f"  Bin at per-row inflow:             mean={per_row_bin[unseen_mask].mean():.3f}  std={per_row_bin[unseen_mask].std():.3f}")

# Correlation between fixed and per-row bin
r_fixed_bin, _ = sp.pearsonr(fixed_correction_unseen[unseen_mask], per_row_bin[unseen_mask])
print(f"\n  r(fixed_corr, per_row_bin) for unseen = {r_fixed_bin:.4f}")
print(f"  (low r means different mechanisms)")

# ============================================================
print("\n" + "="*70)
print("Part 3: Candidate submissions using fixed corrections")
print("="*70)

sub_tmpl = pd.read_csv('sample_submission.csv')

configs = []
# Fixed log-linear correction at layout mean inflow
for alpha_f in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
    ct = np.clip(rh_trip_t + alpha_f * fixed_correction_unseen, 0, None)
    configs.append((f'fixedLL_u{alpha_f:.2f}', ct))

# Bin at layout mean inflow
for alpha_b in [1.0, 1.5, 2.0]:
    ct = np.clip(rh_trip_t + alpha_b * bin_correction_unseen, 0, None)
    configs.append((f'binLayout_u{alpha_b:.2f}', ct))

# Additive flat constant to all unseen
for flat in [2.0, 4.0, 5.0, 6.0, 7.0, 8.0]:
    ct = np.clip(rh_trip_t.copy(), 0, None)
    ct[unseen_mask] = np.clip(rh_trip_t[unseen_mask] + flat, 0, None)
    configs.append((f'flat_u{flat:.1f}', ct))

# Multiplicative scaling for unseen
for scale in [1.1, 1.15, 1.2, 1.25, 1.3]:
    ct = rh_trip_t.copy()
    ct[unseen_mask] = np.clip(rh_trip_t[unseen_mask] * scale, 0, None)
    configs.append((f'mult_u{scale:.2f}', ct))

# Current best (alpha=1.5 per-row bin) for comparison
ct_base = np.clip(rh_trip_t + 1.5 * np.where(unseen_mask, test_resid, 0), 0, None)
configs.append(('asym_bin_u1.5_REF', ct_base))

ct_base10 = np.clip(rh_trip_t + 1.0 * np.where(unseen_mask, test_resid, 0), 0, None)
configs.append(('asym_bin_u1.0_REF', ct_base10))

print(f"\n{'Config':30s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
print("-"*65)
for name, ct in configs:
    print(f"  {name:30s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Which approach has unseen mean closest to expected (28-30)?")
print("="*70)
print(f"\nTarget unseen mean from cross-layout log-linear: ~28.23")
print(f"Target range: 26-30 (uncertainty band)")
print()

for name, ct in configs:
    u_mean = ct[unseen_mask].mean()
    if 24 <= u_mean <= 32:
        in_range = "***" if 26 <= u_mean <= 30 else "  ok"
        print(f"  {in_range}  {name:30s}  unseen={u_mean:.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Save promising candidates")
print("="*70)

to_save = [
    ('fixedLL_u075', next(ct for n,ct in configs if n=='fixedLL_u0.75')),
    ('fixedLL_u100', next(ct for n,ct in configs if n=='fixedLL_u1.00')),
    ('fixedLL_u125', next(ct for n,ct in configs if n=='fixedLL_u1.25')),
    ('binLayout_u100', next(ct for n,ct in configs if n=='binLayout_u1.00')),
    ('binLayout_u150', next(ct for n,ct in configs if n=='binLayout_u1.50')),
    ('flat_u6', next(ct for n,ct in configs if n=='flat_u6.0')),
    ('mult_u115', next(ct for n,ct in configs if n=='mult_u1.15')),
    ('mult_u120', next(ct for n,ct in configs if n=='mult_u1.20')),
]

print(f"\n{'Filename':65s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for label, ct in to_save:
    fname = f"FINAL_NEW_{label}_OOF{trip_mae:.5f}.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  SAVED  {fname[:63]:63s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 6: Sensitivity analysis — seen impact of unseen correction")
print("="*70)
# What if our calibration accidentally inflates unseen MAE instead of reducing it?
# Expected unseen y_true distribution: mean=28.23 (log-linear), but r=0.17 so uncertainty is high
# If actual unseen mean = 25 (lower than expected), corrections hurt
# If actual unseen mean = 32 (higher than expected), corrections help
#
# Break-even analysis: what unseen mean makes each submission equivalent to current best?
# LB ≈ 0.6*seen_MAE + 0.4*unseen_MAE (approximate decomposition)
#
# For the current triple_base: LB ≈ some value
# With correction: if unseen_MAE decreases by X, LB decreases by 0.4*X

# Simulated seen and unseen MAE at different hypothetical y_true means for unseen
print(f"\n{'Scenario':40s}  {'seen':>8}  {'unseen':>8}  {'LB_approx':>10}")
for hyp_u_mean in [20, 22, 24, 25, 26, 27, 28, 29, 30, 32, 35]:
    # Assume unseen y_true is normally distributed around hyp_u_mean with same std as training
    # Use test predictions at different correction levels
    ct_base = rh_trip_t
    ct_u15  = np.clip(rh_trip_t + 1.5 * np.where(unseen_mask, test_resid, 0), 0, None)
    ct_flat6 = rh_trip_t.copy()
    ct_flat6[unseen_mask] = np.clip(rh_trip_t[unseen_mask] + 6.0, 0, None)

    # Simulated MAE: |pred_unseen - hyp_u_mean| (simplified — treats all unseen as hyp_u_mean)
    # This is a rough approximation
    base_mae_u = abs(ct_base[unseen_mask].mean() - hyp_u_mean)
    u15_mae_u  = abs(ct_u15[unseen_mask].mean() - hyp_u_mean)
    flat6_mae_u = abs(ct_flat6[unseen_mask].mean() - hyp_u_mean)

    # This is too simplified — skip to actual LB estimation approach
    pass

print("(Mean-based LB approximation is too simplified for directional analysis)")
print("Key insight: r=0.17 means ~28% of unseen variance explained by inflow")
print("Expected unseen y_true mean: 28.23 (log-linear) vs 23.6 (current pred)")
print(f"\nConservative estimate: unseen mean y_true >= 26 (below: already well-calibrated)")
print(f"Aggressive estimate:  unseen mean y_true >= 30 (supports alpha=1.5-2.0)")

print("\nDone.")
