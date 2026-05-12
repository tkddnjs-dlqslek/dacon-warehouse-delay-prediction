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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Load seqC_huber10
h10_o = np.clip(np.load('results/oracle_seq/oof_seqC_huber10.npy'), 0, None)
h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)
print(f"seqC_huber10: OOF={mae_fn(h10_o):.5f}  seen={h10_t[seen_mask].mean():.3f}  unseen={h10_t[unseen_mask].mean():.3f}")
print(f"  r(oracle_NEW) = {pearsonr(h10_t, oracle_new_t)[0]:.4f}")

# Load also seqC_pressure (slightly higher seen) and seqC_lgb_stack (seen=17.045 similar to oracle_NEW)
pres_t = np.clip(np.load('results/oracle_seq/test_C_pressure.npy'), 0, None)
lgbs_t = np.clip(np.load('results/oracle_seq/test_C_lgb_stack.npy'), 0, None)
rank_lag_t = np.clip(np.load('results/oracle_seq/test_C_ranklag.npy'), 0, None)

print(f"seqC_pressure: seen={pres_t[seen_mask].mean():.3f}  unseen={pres_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(pres_t, oracle_new_t)[0]:.4f}")
print(f"seqC_lgb_stack: seen={lgbs_t[seen_mask].mean():.3f}  unseen={lgbs_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(lgbs_t, oracle_new_t)[0]:.4f}")
print(f"seqC_ranklag: seen={rank_lag_t[seen_mask].mean():.3f}  unseen={rank_lag_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(rank_lag_t, oracle_new_t)[0]:.4f}")

sub_tmpl = pd.read_csv('sample_submission.csv')

print("\n" + "="*70)
print("seqC_huber10 UNSEEN-ONLY blend with oracle_NEW")
print("seqC_huber10 has higher unseen (23.804) and lower seen (18.193)")
print("Using UNSEEN-ONLY blend to avoid increasing seen predictions")
print("="*70)

good_candidates = {}

for w_u in [0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w_u)*oracle_new_t[unseen_mask] + w_u*h10_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    r, _ = pearsonr(ct, oracle_new_t)
    print(f"  w_u={w_u:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δunseen={du:+.3f}  r(oN)={r:.4f}")
    if w_u in [0.20, 0.50, 1.0]:
        good_candidates[f'oN_h10u{int(w_u*100)}'] = ct

# seqC_huber10 per-bucket analysis
print("\n" + "="*70)
print("seqC_huber10 per-bucket analysis")
print("="*70)
p_h10 = h10_o
y = y_true
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
print(f"\n  {'bucket':12s} {'n':>7} {'y_mean':>8} {'p_mean':>8} {'resid':>8} {'h10_vs_oN':>10}")
# Also load oracle_NEW OOF
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fwd = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o_raw=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fwd['mega33']-dr2-dr3; w2_=fwd['iter_r2']+dr2; w3_=fwd['iter_r3']+dr3
fx_o=wm*mega_oof+fwd['rank_adj']*rank_oof+fwd['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o_raw+0.10*xgbc_o+0.08*mono_o,0,None)

for lo, hi in zip(bins[:-1], bins[1:]):
    mask_h10 = (p_h10 >= lo) & (p_h10 < hi)
    mask_oN  = (fw4_oo >= lo) & (fw4_oo < hi)  # compare same bucket based on oracle_NEW pred
    if mask_oN.sum() > 0:
        ym = y[mask_oN].mean()
        pm_oN = fw4_oo[mask_oN].mean()
        resid_oN = pm_oN - ym
        pm_h10 = p_h10[mask_oN].mean()  # h10's prediction for same oracle_NEW bucket
        resid_h10 = pm_h10 - ym
        print(f"  [{lo:3d},{hi:3d}): n={mask_oN.sum():7d}  y={ym:8.3f}  oN={pm_oN:8.3f}(r={resid_oN:+.3f})  h10={pm_h10:8.3f}(r={resid_h10:+.3f})")

# Compare h10 vs oracle_NEW on unseen test distribution
print("\n" + "="*70)
print("seqC_huber10 vs oracle_NEW for unseen test buckets")
print("="*70)
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_test_mega=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_t_raw=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
mega_t=(1-w34)*mega33_test+w34*mega34_test
fxt=wm*mega_t+fwd['rank_adj']*rank_test+fwd['iter_r1']*r1_test+w2_*r2_test+w3_*r3_test
bb_tt=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem2*rem_t,0,None)
bb_tt=np.clip((1-w_cb)*bb_tt+w_cb*cb_test_mega,0,None)
fw4_tt=np.clip(0.74*bb_tt+0.08*slh_t_raw+0.10*xgbc_t+0.08*mono_t,0,None)

p_oN_unseen = oracle_new_t[unseen_mask]
p_h10_unseen = h10_t[unseen_mask]
p_fw4_unseen = fw4_tt[unseen_mask]
diff = p_h10_unseen - p_oN_unseen
print(f"\nh10 - oracle_NEW for unseen test:")
print(f"  mean={diff.mean():.3f}  std={diff.std():.3f}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN_unseen >= lo) & (p_oN_unseen < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  oN={p_oN_unseen[mask].mean():.3f}  h10={p_h10_unseen[mask].mean():.3f}  diff={diff[mask].mean():+.3f}")

# Save good candidates
print("\n" + "="*70)
print("Save seqC_huber10 blend candidates")
print("="*70)

for label, ct in good_candidates.items():
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# Also try: for high-prediction unseen rows (>25), use h10; for low-prediction rows, keep oracle_NEW
print("\n" + "="*70)
print("Selective h10: apply only for high-prediction unseen rows")
print("="*70)
for threshold in [20, 25, 30]:
    for w_u in [0.30, 0.50]:
        ct = oracle_new_t.copy()
        high_mask = (oracle_new_t > threshold) & unseen_mask
        ct[high_mask] = (1-w_u)*oracle_new_t[high_mask] + w_u*h10_t[high_mask]
        ct = np.clip(ct, 0, None)
        du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  h10_high>{threshold}_w{int(w_u*100)}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}  n_affected={high_mask.sum()}")

print("\nDone.")
