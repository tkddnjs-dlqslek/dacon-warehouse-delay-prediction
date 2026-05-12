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
dual_mae=mae_fn(dual_o)

sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)
print(f"Base: dual OOF={dual_mae:.5f}  triple OOF={trip_mae:.5f}")

# Build bin residuals (from dual_o, not rh_trip_o)
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

print(f"test_resid: seen={test_resid[seen_mask].mean():.3f}  unseen={test_resid[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 1: Layout-level training inflow statistics")
print("="*70)

# Compute per-layout training inflow means
layout_train_inflow = train_raw.groupby('layout_id')[inflow_col].mean()
print(f"Layout train inflow mean: {layout_train_inflow.mean():.1f}  std={layout_train_inflow.std():.1f}")

# Per-test-row: how much does test inflow exceed layout training mean?
test_layout_ids = test_raw['layout_id'].values
test_layout_train_mean = np.array([
    layout_train_inflow.get(lid, np.nan) for lid in test_layout_ids
])
# For unseen layouts, there's no training inflow reference — use NaN
test_inflow_elevation = test_inflow - test_layout_train_mean

print(f"\nSeen test rows (n={seen_mask.sum()}):")
print(f"  inflow mean={test_inflow[seen_mask].mean():.1f}  elevation mean={test_inflow_elevation[seen_mask].mean():.1f}")
print(f"  elevation>0: {(test_inflow_elevation[seen_mask]>0).sum()} rows")
print(f"  elevation>10: {(test_inflow_elevation[seen_mask]>10).sum()} rows")
print(f"  elevation>20: {(test_inflow_elevation[seen_mask]>20).sum()} rows")
print(f"  elevation>30: {(test_inflow_elevation[seen_mask]>30).sum()} rows")
print(f"  elevation>50: {(test_inflow_elevation[seen_mask]>50).sum()} rows")

print(f"\nUnseen test rows (n={unseen_mask.sum()}):")
print(f"  inflow mean={test_inflow[unseen_mask].mean():.1f}")
print(f"  elevation is NaN for unseen (no training reference)")

# ============================================================
print("\n" + "="*70)
print("Part 2: Layout-level test vs train inflow gap (seen only)")
print("="*70)

# Per-layout seen test inflow
test_seen = test_raw[seen_mask].copy()
test_seen['inflow'] = test_inflow[seen_mask]
layout_test_inflow = test_seen.groupby('layout_id')['inflow'].mean()

print(f"\n{'Layout':12s}  {'Train':8s}  {'Test':8s}  {'Elevation':>9s}  {'n_rows':>6s}")
print("-"*55)
elevated_layouts = []
for lid in layout_test_inflow.index:
    train_m = layout_train_inflow.get(lid, np.nan)
    test_m = layout_test_inflow[lid]
    elev = test_m - train_m
    n = (test_seen['layout_id']==lid).sum()
    if abs(elev) > 5:
        print(f"  {lid:12s}  {train_m:8.1f}  {test_m:8.1f}  {elev:+9.1f}  {n:>6d}")
    if elev > 30:
        elevated_layouts.append(lid)

print(f"\nLayouts with elevation > 30: {len(elevated_layouts)}")
print(f"  {elevated_layouts}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Layout-aware alpha assignments")
print("="*70)

# Strategy variants:
# A: unseen=1.5, seen_elevated(>20)=0.5, seen_normal=0.0
# B: unseen=1.5, seen_elevated(>20)=1.0, seen_normal=0.0
# C: unseen=1.5, seen_elevated(>30)=0.5, seen_normal=0.0
# D: unseen=1.5, seen_elevated(>30)=1.0, seen_normal=0.0
# E: unseen=1.0, seen_elevated(>20)=0.5, seen_normal=0.0

sub_tmpl = pd.read_csv('sample_submission.csv')

configs = [
    # (name, alpha_unseen, threshold_elevation, alpha_elevated, alpha_normal)
    ('layAware_u15_t20_e05', 1.5, 20, 0.5, 0.0),
    ('layAware_u15_t20_e10', 1.5, 20, 1.0, 0.0),
    ('layAware_u15_t30_e05', 1.5, 30, 0.5, 0.0),
    ('layAware_u15_t30_e10', 1.5, 30, 1.0, 0.0),
    ('layAware_u10_t20_e05', 1.0, 20, 0.5, 0.0),
    ('layAware_u10_t30_e05', 1.0, 30, 0.5, 0.0),
    ('layAware_u15_t10_e05', 1.5, 10, 0.5, 0.0),
    ('layAware_u15_t10_e10', 1.5, 10, 1.0, 0.0),
    # Comparison baselines
    ('asym_u15_pure',        1.5,  0, 0.0, 0.0),  # no seen correction
    ('asym_u15_sym_pure',    1.5,  0, 1.5, 0.0),  # symmetric for all
]

print(f"\n{'Config':32s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}  {'n_elev':>6}")
print("-"*90)

results = []
for name, a_u, thresh, a_e, a_n in configs:
    alpha_arr = np.zeros(len(test_raw))
    # Unseen rows
    alpha_arr[unseen_mask] = a_u
    # Seen rows: check elevation
    if thresh > 0 or a_e > 0:
        for i in np.where(seen_mask)[0]:
            lid = test_raw.loc[i, 'layout_id']
            elev = test_inflow[i] - layout_train_inflow.get(lid, np.nan)
            if not np.isnan(elev) and elev > thresh:
                alpha_arr[i] = a_e
            else:
                alpha_arr[i] = a_n

    ct = np.clip(rh_trip_t + alpha_arr*test_resid, 0, None)
    n_elevated = (alpha_arr[seen_mask] == a_e).sum() if a_e > 0 else 0
    print(f"  {name:32s}  {trip_mae:9.5f}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}  {n_elevated:>6d}")
    results.append((name, alpha_arr, ct))

# ============================================================
print("\n" + "="*70)
print("Part 4: Fine-grained grid for best layout-aware config")
print("="*70)

print(f"\n{'Config':35s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
print("-"*75)

grid_results = []
for a_u in [1.0, 1.5, 2.0]:
    for thresh in [10, 20, 30, 50]:
        for a_e in [0.3, 0.5, 0.7, 1.0]:
            alpha_arr = np.zeros(len(test_raw))
            alpha_arr[unseen_mask] = a_u
            for i in np.where(seen_mask)[0]:
                lid = test_raw.loc[i, 'layout_id']
                elev = test_inflow[i] - layout_train_inflow.get(lid, np.nan)
                if not np.isnan(elev) and elev > thresh:
                    alpha_arr[i] = a_e
            ct = np.clip(rh_trip_t + alpha_arr*test_resid, 0, None)
            n_elev = (alpha_arr[seen_mask] == a_e).sum()
            name = f"u{a_u:.1f}_t{thresh}_e{a_e:.1f}"
            grid_results.append((name, ct, a_u, thresh, a_e, n_elev))
            print(f"  {name:35s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Save top layout-aware candidates")
print("="*70)

to_save = [
    ('layAware_u15_t20_e05', results[0][2]),
    ('layAware_u15_t20_e10', results[1][2]),
    ('layAware_u15_t30_e05', results[2][2]),
    ('layAware_u15_t30_e10', results[3][2]),
    ('layAware_u10_t20_e05', results[4][2]),
    ('layAware_u10_t30_e05', results[5][2]),
]

print(f"\n{'Filename':65s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for label, ct in to_save:
    fname = f"FINAL_NEW_{label}_OOF{trip_mae:.5f}.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    status = "SAVED"
    print(f"  [{status}] {fname[:63]:63s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

print("\nDone.")
