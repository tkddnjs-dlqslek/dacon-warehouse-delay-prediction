import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from itertools import product

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
dual_mae=mae_fn(dual_o)

# rh triple
sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)
print(f"Base: dual OOF={dual_mae:.5f}  triple OOF={trip_mae:.5f}")
print(f"triple test: {rh_trip_t.mean():.3f}  seen={rh_trip_t[seen_mask].mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")

# Build bin residuals
inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values; test_inflow=test_raw[inflow_col].values
residual=y_true-dual_o
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

# ============================================================
print("\n" + "="*70)
print("Part 1: Asymmetric calibration (alpha_seen vs alpha_unseen)")
print("="*70)
print(f"\nBase triple test: seen={rh_trip_t[seen_mask].mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")
print(f"test_resid mean: seen={test_resid[seen_mask].mean():.3f}  unseen={test_resid[unseen_mask].mean():.3f}")
print()
print(f"{'a_seen':>6}  {'a_unseen':>8}  {'test':>8}  {'seen':>8}  {'unseen':>8}")

asym_cands = []
for a_s, a_u in [(0, 0.5), (0, 1.0), (0, 1.5), (0, 2.0),
                  (0.1, 0.5), (0.1, 1.0), (0.1, 1.5),
                  (0.2, 0.5), (0.2, 1.0), (0.2, 2.0),
                  (0.0, 0.0), (0.1, 0.1), (0.3, 0.3), (1.0, 1.0)]:
    alpha_arr = np.where(unseen_mask, a_u, a_s)
    ct = np.clip(rh_trip_t + alpha_arr*test_resid, 0, None)
    print(f"  {a_s:>5.2f}  {a_u:>8.2f}  {ct.mean():>8.3f}  {ct[seen_mask].mean():>8.3f}  {ct[unseen_mask].mean():>8.3f}")
    asym_cands.append((f"asym_s{a_s:.2f}_u{a_u:.2f}", ct))

# ============================================================
print("\n" + "="*70)
print("Part 2: Consistency check — seen test layout inflow distribution")
print("="*70)

test_seen_inflow = test_inflow[seen_mask]
valid_si = test_seen_inflow[~np.isnan(test_seen_inflow)]
print(f"Seen test inflow:   mean={valid_si.mean():.1f}  p50={np.percentile(valid_si,50):.1f}  p75={np.percentile(valid_si,75):.1f}  max={valid_si.max():.1f}")
test_unseen_inflow = test_inflow[unseen_mask]
valid_ui = test_unseen_inflow[~np.isnan(test_unseen_inflow)]
print(f"Unseen test inflow: mean={valid_ui.mean():.1f}  p50={np.percentile(valid_ui,50):.1f}  p75={np.percentile(valid_ui,75):.1f}  max={valid_ui.max():.1f}")
print(f"\nSeen test inflow bins:")
for lo,hi in zip(bins[:-1],bins[1:]):
    n_s = ((test_seen_inflow>=lo)&(test_seen_inflow<hi)).sum()
    n_u = ((test_unseen_inflow>=lo)&(test_unseen_inflow<hi)).sum()
    if n_s+n_u==0: continue
    r = bin_residuals.get((lo,hi), 0)
    print(f"  [{lo:3d},{hi:4d}): seen_n={n_s:5d}  unseen_n={n_u:5d}  bin_resid={r:+.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Seen-only correction effect on OOF (validate direction)")
print("="*70)

# For OOF, all rows are "seen". Apply the bin correction and check OOF.
# This tells us: is the correction helpful for seen rows?
train_resid_calib = inflow_to_resid(train_inflow)
for alpha in [0.0, 0.05, 0.10, 0.20, 0.50, 1.0]:
    co = np.clip(dual_o + alpha*train_resid_calib, 0, None)
    m = mae_fn(co)
    print(f"  alpha={alpha:.2f}: OOF={m:.5f}  delta={m-dual_mae:+.5f}")

print()
print("NOTE: If OOF improves with alpha>0, then bin calibration DOES help seen rows")
print("       If OOF worsens, calibration is counterproductive for seen rows")

# ============================================================
print("\n" + "="*70)
print("Part 4: Save top asymmetric calibration candidates")
print("="*70)

sub_tmpl = pd.read_csv('sample_submission.csv')
to_save = [
    ('asym_s0.0_u0.5',  asym_cands[0][1]),
    ('asym_s0.0_u1.0',  asym_cands[1][1]),
    ('asym_s0.0_u1.5',  asym_cands[2][1]),
    ('asym_s0.1_u1.0',  asym_cands[5][1]),
    ('asym_s0.1_u1.5',  asym_cands[6][1]),
    ('asym_s0.2_u1.0',  asym_cands[8][1]),
    ('asym_s0.2_u2.0',  asym_cands[9][1]),
]
print(f"{'Name':40s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for name, ct in to_save:
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    fname = f"FINAL_triple_{name}_OOF{trip_mae:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"  {fname[:52]:52s}  {trip_mae:.5f}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")
