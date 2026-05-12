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

inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values; test_inflow=test_raw[inflow_col].values
residual=y_true-dual_o

# Build bin residuals
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
train_resid=inflow_to_resid(train_inflow)

# ============================================================
print("\n" + "="*70)
print("Part 1: Per-layout analysis — training residuals for elevated-test layouts")
print("="*70)

# Elevated layouts (test_elevation > 30)
elevated_lids = ['WH_056', 'WH_091', 'WH_104', 'WH_147', 'WH_150', 'WH_243', 'WH_250', 'WH_299']
layout_train_inflow = train_raw.groupby('layout_id')[inflow_col].mean()

print(f"\n{'Layout':10s}  {'TrainInflow':>11s}  {'TrainResid':>10s}  {'BinResidAtTest':>14s}  {'TestInflow':>10s}  {'Elev':>6s}")
print("-"*75)

test_seen = test_raw[seen_mask].copy()
test_seen['inflow'] = test_inflow[seen_mask]
layout_test_inflow = test_seen.groupby('layout_id')['inflow'].mean()

for lid in elevated_lids:
    tr_mask = train_raw['layout_id'] == lid
    tr_inflow = layout_train_inflow[lid]
    tr_resid = residual[tr_mask.values].mean()
    test_if = layout_test_inflow.get(lid, np.nan)
    elev = test_if - tr_inflow
    # what bin residual applies at test inflow?
    bin_r = 0.0
    for (lo,hi),mr in bin_residuals.items():
        if test_if >= lo and test_if < hi:
            bin_r = mr; break
    print(f"  {lid:10s}  {tr_inflow:>11.1f}  {tr_resid:>+10.3f}  {bin_r:>+14.3f}  {test_if:>10.1f}  {elev:>+6.1f}")

# Also show well-calibrated high-inflow layouts for comparison
print(f"\n{'Layout':10s}  {'TrainInflow':>11s}  {'TrainResid':>10s}  {'Note':20s}")
print("-"*55)
# Top 10 layouts by training inflow
top_inflow_lids = layout_train_inflow.nlargest(10).index
for lid in top_inflow_lids:
    tr_mask = train_raw['layout_id'] == lid
    tr_inflow = layout_train_inflow[lid]
    tr_resid = residual[tr_mask.values].mean()
    in_test = lid in test_raw['layout_id'].values
    note = "in-test" if in_test else "no-test"
    print(f"  {lid:10s}  {tr_inflow:>11.1f}  {tr_resid:>+10.3f}  {note:20s}")

# ============================================================
print("\n" + "="*70)
print("Part 2: Cross-layout inflow effect — within-training analysis")
print("="*70)
# For each training layout, compute mean residual vs mean inflow
layout_stats = {}
for lid in train_raw['layout_id'].unique():
    m = train_raw['layout_id'] == lid
    layout_stats[lid] = {
        'inflow': float(train_inflow[m.values].mean()),
        'residual': float(residual[m.values].mean()),
        'y_true': float(y_true[m.values].mean()),
        'pred': float(dual_o[m.values].mean()),
        'n': int(m.sum()),
    }
ls_df = pd.DataFrame(layout_stats).T.reset_index().rename(columns={'index':'layout_id'})
ls_df = ls_df.sort_values('inflow', ascending=False)

print(f"\nAll training layouts sorted by inflow:")
print(f"{'Layout':10s}  {'Inflow':>8s}  {'y_true':>8s}  {'pred':>8s}  {'Resid':>8s}")
print("-"*55)
for _, row in ls_df.iterrows():
    print(f"  {row['layout_id']:10s}  {row['inflow']:>8.1f}  {row['y_true']:>8.2f}  {row['pred']:>8.2f}  {row['residual']:>+8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Does layout-level residual correlate with current inflow?")
print("="*70)

from scipy import stats as sp
inflows = ls_df['inflow'].values
resids = ls_df['residual'].values
ytrue_m = ls_df['y_true'].values
r, p = sp.pearsonr(inflows, resids)
r2, p2 = sp.pearsonr(np.log1p(inflows), resids)
print(f"\nLayout-level Pearson(inflow, residual) = {r:.4f} (p={p:.4f})")
print(f"Layout-level Pearson(log1p(inflow), residual) = {r2:.4f} (p={p2:.4f})")
print(f"N layouts = {len(inflows)}")

# Fit log-linear
from numpy.polynomial import polynomial as P
log_inf = np.log1p(inflows)
coef = np.polyfit(log_inf, resids, 1)
print(f"\nLog-linear fit: resid = {coef[0]:.4f} * log1p(inflow) + {coef[1]:.4f}")
for target_inflow in [50, 100, 130, 160, 172, 200, 250, 300]:
    pred_r = coef[0]*np.log1p(target_inflow) + coef[1]
    print(f"  inflow={target_inflow}: predicted resid = {pred_r:+.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Are elevated-test layouts actually low-residual in training?")
print("="*70)
print("(Key question: are they well-calibrated at train inflow, then need correction at test inflow?)")
print()
print(f"{'Layout':10s}  {'TrainResid':>10s}  {'BinResidAtTrain':>15s}  {'RelativeResid':>13s}  {'Category':15s}")
print("-"*75)
for lid in elevated_lids:
    tr_mask = train_raw['layout_id'] == lid
    tr_inflow = layout_train_inflow[lid]
    tr_resid = residual[tr_mask.values].mean()
    # bin residual at train inflow level
    br_train = 0.0
    for (lo,hi),mr in bin_residuals.items():
        if tr_inflow >= lo and tr_inflow < hi:
            br_train = mr; break
    # relative = how much is layout-specific vs bin-level
    relative = tr_resid - br_train
    category = "HIGH" if relative > 3 else ("LOW" if relative < -3 else "normal")
    print(f"  {lid:10s}  {tr_resid:>+10.3f}  {br_train:>+15.3f}  {relative:>+13.3f}  {category:15s}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Estimate per-layout expected test correction")
print("="*70)
# For each elevated layout:
# - We know layout-specific residual (above/below average)
# - We know test inflow
# - Expected test residual = bin_resid(test_inflow) + layout_specific_offset
# - So correction = (bin_resid(test_inflow) + layout_offset) * alpha
# If layout_specific is HIGH, it means model is ALREADY underpredicting for that layout
# even at training inflow — adding correction would be too conservative

print()
print(f"{'Layout':10s}  {'EstTestResid':>12s}  {'BinAtTest':>9s}  {'LayOffset':>9s}  {'Action':15s}")
print("-"*65)
for lid in elevated_lids:
    tr_mask = train_raw['layout_id'] == lid
    tr_inflow = layout_train_inflow[lid]
    tr_resid = residual[tr_mask.values].mean()
    br_train = 0.0
    for (lo,hi),mr in bin_residuals.items():
        if tr_inflow >= lo and tr_inflow < hi:
            br_train = mr; break
    layout_offset = tr_resid - br_train

    test_if = layout_test_inflow.get(lid, np.nan)
    br_test = 0.0
    for (lo,hi),mr in bin_residuals.items():
        if test_if >= lo and test_if < hi:
            br_test = mr; break
    est_test_resid = br_test + layout_offset
    action = "CORRECT" if est_test_resid > 1 else ("OVERCORRECT?" if est_test_resid < -1 else "neutral")
    print(f"  {lid:10s}  {est_test_resid:>+12.3f}  {br_test:>+9.3f}  {layout_offset:>+9.3f}  {action:15s}")

# Also check for non-elevated but high-residual seen layouts
print(f"\nTop 10 seen layouts by training residual (non-elevated):")
ls_df_seen = ls_df[ls_df['layout_id'].isin(test_raw['layout_id'].values)].copy()
ls_df_seen_sorted = ls_df_seen.sort_values('residual', ascending=False)
for _, row in ls_df_seen_sorted.head(10).iterrows():
    lid = row['layout_id']
    te_if = layout_test_inflow.get(lid, np.nan)
    tr_if = row['inflow']
    elev = te_if - tr_if if not np.isnan(te_if) else float('nan')
    print(f"  {lid:10s}  TrainResid={row['residual']:+.3f}  TrainInflow={tr_if:.1f}  TestInflow={te_if:.1f}  Elev={elev:+.1f}")

print("\nDone.")
