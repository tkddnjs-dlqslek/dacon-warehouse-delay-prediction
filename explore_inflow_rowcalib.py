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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask     = ~unseen_mask

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega33_test=d33['meta_avg_test'][te_id2]
mega34_oof=d34['meta_avg_oof'][id2]; mega34_test=d34['meta_avg_test'][te_id2]
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
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

def make_pred(w34=0.25, dr2=-0.04, dr3=-0.02, wf=0.72, w_cb=0.12):
    mega=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
    wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx=wm*mega+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
    fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
    w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo=np.clip(wf*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
    ot=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t,0,None)
    if w_cb>0:
        oo=np.clip((1-w_cb)*oo+w_cb*cb_oof,0,None)
        ot=np.clip((1-w_cb)*ot+w_cb*cb_test,0,None)
    return oo,ot

bb_o,bb_t=make_pred()
fw4_o=np.clip(0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
fw4_t=np.clip(0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t,0,None)

n1,w1,n2,w2=2000,0.15,5500,0.08
sfw=np.sort(fw4_o); sft=np.sort(fw4_t)
m1_o=fw4_o>=sfw[-n1]; m2_o=fw4_o>=sfw[-n2]
dual_o=fw4_o.copy()
dual_o[m1_o]=(1-w1)*fw4_o[m1_o]+w1*rh_o[m1_o]
dual_o[m2_o]=(1-w2)*dual_o[m2_o]+w2*slhm_o[m2_o]
dual_o=np.clip(dual_o,0,None)
m1_t=fw4_t>=sft[-n1]; m2_t=fw4_t>=sft[-n2]
dual_t=fw4_t.copy()
dual_t[m1_t]=(1-w1)*fw4_t[m1_t]+w1*rh_t[m1_t]
dual_t[m2_t]=(1-w2)*dual_t[m2_t]+w2*slhm_t[m2_t]
dual_t=np.clip(dual_t,0,None)
dual_mae=mae_fn(dual_o)
print(f"dual: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

groups=train_raw['layout_id'].values
gkf=GroupKFold(n_splits=5)
fold_ids=np.zeros(len(y_true),dtype=int)
for fi,(_,vi) in enumerate(gkf.split(train_raw,y_true,groups)): fold_ids[vi]=fi
fold_mae_fn=lambda p:[float(np.mean(np.abs(p[fold_ids==fi]-y_true[fold_ids==fi]))) for fi in range(5)]

print("\n" + "="*70)
print("Part 1: Build inflow-residual lookup table")
print("="*70)

inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values
test_inflow=test_raw[inflow_col].values
residual=y_true-dual_o

bins=[0,10,20,30,40,50,60,70,80,100,120,150,200,250,300,500,1000]
bin_residuals={}
for i in range(len(bins)-1):
    lo,hi=bins[i],bins[i+1]
    mask=(train_inflow>=lo)&(train_inflow<hi)
    if mask.sum()==0: continue
    mr=float(np.nanmean(residual[mask]))
    bin_residuals[(lo,hi)]=mr
    print(f"  [{lo:4d},{hi:4d}): n={mask.sum():7d}  mean_resid={mr:+.3f}")

def inflow_to_residual(inflow_arr):
    """Look up expected residual for each row's inflow."""
    result = np.zeros(len(inflow_arr))
    for (lo,hi),mr in bin_residuals.items():
        mask=(inflow_arr>=lo)&(inflow_arr<hi)
        result[mask]=mr
    return result

train_expected_resid=inflow_to_residual(train_inflow)
test_expected_resid=inflow_to_residual(test_inflow)

print(f"\nTrain expected_resid stats: mean={train_expected_resid.mean():.3f}  std={train_expected_resid.std():.3f}")
print(f"Test expected_resid stats: mean={test_expected_resid.mean():.3f}")
print(f"  seen: {test_expected_resid[seen_mask].mean():.3f}")
print(f"  unseen: {test_expected_resid[unseen_mask].mean():.3f}")

print("\n" + "="*70)
print("Part 2: Apply inflow calibration to test (alpha grid search)")
print("  - Apply ONLY to test, OOF unchanged (no double-use)")
print("="*70)

print(f"\n{'alpha':>6}  {'test':>8}  {'seen':>8}  {'unseen':>8}  OOF={dual_mae:.5f}")
saved_rows=[]
for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]:
    t=np.clip(dual_t + alpha*test_expected_resid, 0, None)
    print(f"  {alpha:>5.2f}  {t.mean():>8.3f}  {t[seen_mask].mean():>8.3f}  {t[unseen_mask].mean():>8.3f}")
    saved_rows.append((alpha, t))

print("\n" + "="*70)
print("Part 3: Combined dual gate + inflow calib + unseen boost")
print("="*70)

# Moderate inflow calib (alpha=0.20) + conservative unseen boost
m1_s_t=(fw4_t>=sft[-n1])&seen_mask; m2_s_t=(fw4_t>=sft[-n2])&seen_mask

def apply_unseen_boost_on(base_t, n1u, w1u, n2u, w2u):
    m1_u=(fw4_t>=sft[-n1u])&unseen_mask
    m2_u=(fw4_t>=sft[-n2u])&unseen_mask
    m1f=m1_s_t|m1_u; m2f=m2_s_t|m2_u
    w1a=np.where(unseen_mask,w1u,w1); w2a=np.where(unseen_mask,w2u,w2)
    t=base_t.copy()
    t[m1f]=(1-w1a[m1f])*fw4_t[m1f]+w1a[m1f]*rh_t[m1f]
    t[m2f]=(1-w2a[m2f])*t[m2f]+w2a[m2f]*slhm_t[m2f]
    return np.clip(t,0,None)

print(f"\n{'Config':50s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for alpha in [0.10, 0.20, 0.30]:
    base_t = np.clip(dual_t + alpha*test_expected_resid, 0, None)
    for n1u,w1u,n2u,w2u in [(2000,0.25,5500,0.12),(5000,0.20,11000,0.10)]:
        t = apply_unseen_boost_on(base_t, n1u, w1u, n2u, w2u)
        name=f"inflow_a{alpha:.2f}+unseen_n{n1u}_w{w1u}"
        print(f"  {name:48s}  {t.mean():>8.3f}  {t[seen_mask].mean():>8.3f}  {t[unseen_mask].mean():>8.3f}")

print("\n" + "="*70)
print("Part 4: Save top inflow calibration candidates")
print("="*70)

sub_template=pd.read_csv('sample_submission.csv')
for alpha, t in [(0.10, saved_rows[1][1]), (0.20, saved_rows[3][1]), (0.30, saved_rows[5][1])]:
    sub=sub_template.copy(); sub['avg_delay_minutes_next_30m']=t
    fname=f"submission_inflowRowCalib_a{alpha:.2f}_OOF{dual_mae:.5f}.csv"
    sub.to_csv(fname,index=False)
    print(f"Saved: {fname}  test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}")

# Best combination
best_combo = np.clip(dual_t + 0.20*test_expected_resid, 0, None)
best_combo = apply_unseen_boost_on(best_combo, 5000, 0.20, 11000, 0.10)
sub=sub_template.copy(); sub['avg_delay_minutes_next_30m']=best_combo
fname=f"submission_inflowRowCalib_a020_unsBst_n5k_w20_OOF{dual_mae:.5f}.csv"
sub.to_csv(fname,index=False)
print(f"Saved: {fname}  test={best_combo.mean():.3f}  seen={best_combo[seen_mask].mean():.3f}  unseen={best_combo[unseen_mask].mean():.3f}")
