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
dual_mae = mae_fn(dual_o)
print(f"Base dual: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 1: y_true vs inflow relationship (training data)")
print("="*70)

train_inflow = train_raw['order_inflow_15m'].values
test_inflow  = test_raw['order_inflow_15m'].values
valid_mask = ~np.isnan(train_inflow)

# Fine-grained bins
bins = list(range(0, 131, 10)) + [140, 150, 170, 200, 250, 300, 400, 500, 1000]
print(f"\n{'bin':15s}  {'n':>7}  {'y_mean':>7}  {'pred_mean':>9}  {'residual':>9}  {'y_p75':>7}")
bin_yt = {}
for i in range(len(bins)-1):
    lo,hi = bins[i],bins[i+1]
    mask = (train_inflow>=lo)&(train_inflow<hi)&valid_mask
    if mask.sum() < 50: continue
    yt = y_true[mask]; pred = dual_o[mask]
    p75 = np.percentile(yt,75)
    resid = float(np.nanmean(yt-pred))
    bin_yt[(lo,hi)] = {'y_mean':yt.mean(),'pred_mean':pred.mean(),'resid':resid,'n':mask.sum()}
    print(f"  [{lo:3d},{hi:4d}):  n={mask.sum():>7}  y={yt.mean():>7.2f}  pred={pred.mean():>9.2f}  resid={resid:>+9.3f}  p75={p75:>7.2f}")

# ============================================================
print("\n" + "="*70)
print("Part 2: Fit y_true = f(inflow) and extrapolate")
print("="*70)

from numpy.polynomial import polynomial as P
# Use bin midpoints where we have data
mids = [(lo+hi)/2 for (lo,hi) in bin_yt if bin_yt[(lo,hi)]['n']>100]
ymeans = [bin_yt[(lo,hi)]['y_mean'] for (lo,hi) in bin_yt if bin_yt[(lo,hi)]['n']>100]
mids = np.array(mids); ymeans = np.array(ymeans)

# Log-linear fit: y ≈ a * log(inflow+1) + b
log_mids = np.log1p(mids)
coefs = np.polyfit(log_mids, ymeans, 1)
a, b = coefs
print(f"\nLog-linear fit: y ≈ {a:.3f} * log1p(inflow) + {b:.3f}")
# Predict at various inflow levels
for inf_val in [90,100,110,120,130,150,160,172,200,250,300,400]:
    pred_y = a * np.log1p(inf_val) + b
    print(f"  inflow={inf_val:4d}:  predicted_y={pred_y:.2f}")

# Square root fit
sqrt_mids = np.sqrt(mids)
coefs2 = np.polyfit(sqrt_mids, ymeans, 1)
a2,b2 = coefs2
print(f"\nSqrt fit: y ≈ {a2:.3f} * sqrt(inflow) + {b2:.3f}")
for inf_val in [90,100,110,120,130,150,172,200,250]:
    pred_y = a2*np.sqrt(inf_val)+b2
    print(f"  inflow={inf_val:4d}:  predicted_y={pred_y:.2f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Estimate residual at test unseen inflow levels")
print("="*70)

# Fit residual = f(inflow) with log model
resid_vals = [bin_yt[(lo,hi)]['resid'] for (lo,hi) in bin_yt if bin_yt[(lo,hi)]['n']>100]
coefs_r = np.polyfit(log_mids, resid_vals, 1)
ar,br = coefs_r
print(f"Log-linear residual fit: resid ≈ {ar:.3f} * log1p(inflow) + {br:.3f}")
# Mean unseen test inflow
test_unseen_inflow = test_inflow[unseen_mask]
valid_ui = test_unseen_inflow[~np.isnan(test_unseen_inflow)]
inf_p25 = np.percentile(valid_ui,25)
inf_p50 = np.percentile(valid_ui,50)
inf_p75 = np.percentile(valid_ui,75)
print(f"\nTest unseen inflow: mean={valid_ui.mean():.1f}  p25={inf_p25:.1f}  p50={inf_p50:.1f}  p75={inf_p75:.1f}  max={valid_ui.max():.1f}")
for inf_val in [inf_p25, inf_p50, valid_ui.mean(), inf_p75]:
    pred_r = ar*np.log1p(inf_val)+br
    print(f"  inflow={inf_val:6.1f}:  extrapolated_resid={pred_r:+.3f}")

# Compute smooth extrapolated residual for all test rows
def smooth_resid(arr):
    valid = ~np.isnan(arr)
    r = np.zeros(len(arr))
    r[valid] = ar*np.log1p(arr[valid])+br
    return r

train_smooth_resid = smooth_resid(train_inflow)
test_smooth_resid = smooth_resid(test_inflow)
print(f"\nSmooth resid stats:")
print(f"  train: mean={train_smooth_resid.mean():.3f}  seen_test={test_smooth_resid[seen_mask].mean():.3f}  unseen_test={test_smooth_resid[unseen_mask].mean():.3f}")

# Compare: bin approach vs smooth approach
bins2=[0,10,20,30,40,50,60,70,80,100,120,150,200,250,300,500,1000]
bin_residuals={}
for i in range(len(bins2)-1):
    lo,hi=bins2[i],bins2[i+1]
    mask=(train_inflow>=lo)&(train_inflow<hi)
    if mask.sum()==0: continue
    bin_residuals[(lo,hi)]=float(np.nanmean((y_true-dual_o)[mask]))
def inflow_to_residual(arr):
    r=np.zeros(len(arr))
    for (lo,hi),mr in bin_residuals.items(): r[(arr>=lo)&(arr<hi)]=mr
    return r
test_bin_resid = inflow_to_residual(test_inflow)
print(f"  bin approach: unseen={test_bin_resid[unseen_mask].mean():.3f}")
print(f"  smooth approach: unseen={test_smooth_resid[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Alpha grid with SMOOTH extrapolated residual")
print("="*70)

print(f"\n{'alpha':>6}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for alpha in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
    ct = np.clip(dual_t + alpha*test_smooth_resid, 0, None)
    print(f"  {alpha:>5.2f}  {ct.mean():>8.3f}  {ct[seen_mask].mean():>8.3f}  {ct[unseen_mask].mean():>8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Cross-validated OOF alpha — does smooth residual improve OOF?")
print("="*70)

from sklearn.model_selection import GroupKFold
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true),dtype=int)
for fi,(_,vi) in enumerate(gkf.split(train_raw,y_true,groups)): fold_ids[vi]=fi

# For each fold, compute residual from OTHER folds and apply to this fold
print("\nCross-validated smooth residual correction:")
print("  (bin lookup built from other 4 folds, applied to held-out fold)")
cv_oof = np.zeros_like(dual_o)
for fi in range(5):
    tr_mask = fold_ids != fi
    va_mask = fold_ids == fi
    # Fit residual function on training folds
    inflow_tr = train_inflow[tr_mask]
    resid_tr = (y_true - dual_o)[tr_mask]
    valid_tr = ~np.isnan(inflow_tr)
    log_inf_tr = np.log1p(inflow_tr[valid_tr])
    resid_fit = resid_tr[valid_tr]
    coefs_fi = np.polyfit(log_inf_tr, resid_fit, 1)
    # Apply to validation fold
    inflow_va = train_inflow[va_mask]
    valid_va = ~np.isnan(inflow_va)
    smooth_va = np.zeros(va_mask.sum())
    smooth_va[valid_va] = coefs_fi[0]*np.log1p(inflow_va[valid_va])+coefs_fi[1]
    cv_oof[va_mask] = smooth_va

print(f"\n  CV smooth resid stats: mean={cv_oof.mean():.3f}  std={cv_oof.std():.3f}")
print(f"  CV smooth resid unseen-like (top 1% inflow): mean={cv_oof[train_inflow>=np.nanpercentile(train_inflow,99)].mean():.3f}")

for alpha in [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]:
    co = np.clip(dual_o + alpha*cv_oof, 0, None)
    m = mae_fn(co)
    print(f"  alpha={alpha:.2f}: OOF={m:.5f}  delta={m-dual_mae:+.5f}")

# Best alpha to use based on CV
alphas_test = [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]
best_a, best_m = 0.0, dual_mae
for alpha in alphas_test:
    co = np.clip(dual_o + alpha*cv_oof, 0, None)
    m = mae_fn(co)
    if m < best_m:
        best_m, best_a = m, alpha

# ============================================================
print("\n" + "="*70)
print("Part 6: Final candidates with smooth residual correction")
print("="*70)

# rh triple gate
sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)

print(f"\nrh_triple base: OOF={trip_mae:.5f}  test={rh_trip_t.mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")

for alpha in [0.10, 0.20, 0.30, 0.50, 1.00]:
    # Smooth residual on triple gate test
    ct = np.clip(rh_trip_t + alpha*test_smooth_resid, 0, None)
    # For OOF: use CV smooth resid
    co = np.clip(rh_trip_o + alpha*cv_oof, 0, None)
    m_oof = mae_fn(co)
    print(f"  triple + smooth*{alpha:.2f}: OOF={m_oof:.5f}  test={ct.mean():.3f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# Compare with bin approach
for alpha in [0.10, 0.20]:
    ct_bin = np.clip(rh_trip_t + alpha*test_bin_resid, 0, None)
    # Bin approach OOF (can compute from CV bins)
    cv_bin_oof = np.zeros_like(dual_o)
    for fi in range(5):
        tr_mask = fold_ids != fi
        va_mask = fold_ids == fi
        inflow_tr = train_inflow[tr_mask]; resid_tr = (y_true - dual_o)[tr_mask]
        br_fi = {}
        for ii in range(len(bins2)-1):
            lo,hi=bins2[ii],bins2[ii+1]
            m=(inflow_tr>=lo)&(inflow_tr<hi)
            if m.sum()==0: continue
            br_fi[(lo,hi)]=float(np.nanmean(resid_tr[m]))
        inflow_va = train_inflow[va_mask]
        for (lo,hi),mr in br_fi.items(): cv_bin_oof[va_mask][(inflow_va>=lo)&(inflow_va<hi)]=mr
    co_bin = np.clip(rh_trip_o + alpha*cv_bin_oof, 0, None)
    m_bin = mae_fn(co_bin)
    print(f"  triple + bin*{alpha:.2f}:    OOF={m_bin:.5f}  test={ct_bin.mean():.3f}  seen={ct_bin[seen_mask].mean():.3f}  unseen={ct_bin[unseen_mask].mean():.3f}")

# Save best candidate if better
sub_tmpl = pd.read_csv('sample_submission.csv')
best_ct_smooth = np.clip(rh_trip_t + 0.30*test_smooth_resid, 0, None)
best_co_smooth = np.clip(rh_trip_o + 0.30*cv_oof, 0, None)
m_smooth30 = mae_fn(best_co_smooth)
sub=sub_tmpl.copy(); sub['avg_delay_minutes_next_30m']=best_ct_smooth
fname=f"FINAL_triple_smoothResid_a030_OOF{m_smooth30:.5f}.csv"
sub.to_csv(fname,index=False)
print(f"\nSaved: {fname}  test={best_ct_smooth.mean():.3f}  seen={best_ct_smooth[seen_mask].mean():.3f}  unseen={best_ct_smooth[unseen_mask].mean():.3f}")
