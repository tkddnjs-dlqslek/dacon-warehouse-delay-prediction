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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

# Reconstruct oracle_NEW OOF (fw4_oo from final_candidate_summary.py)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
cb_test_mega=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o_raw=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o_raw+0.10*xgbc_o+0.08*mono_o,0,None)

print(f"Reconstructed oracle_NEW OOF (fw4_oo): MAE={mae_fn(fw4_oo):.5f}")
print(f"  Expected: 8.37624")
print(f"  Mean prediction: {fw4_oo.mean():.3f}")

y = y_true
p = fw4_oo

print()
print("="*70)
print("oracle_NEW (fw4_oo) calibration by prediction bucket")
print("="*70)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
print(f"\n  {'pred_bucket':20s} {'n':>7} {'y_mean':>8} {'p_mean':>8} {'resid':>8} {'MAE':>8} {'needed_corr':>12}")
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (p >= lo) & (p < hi)
    if mask.sum() > 0:
        ym = y[mask].mean()
        pm = p[mask].mean()
        mae = np.mean(np.abs(p[mask] - y[mask]))
        resid = pm - ym
        needed_corr = -resid  # how much to add to fix bias
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y={ym:8.3f}  p={pm:8.3f}  resid={resid:+8.3f}  MAE={mae:8.3f}  need_add={needed_corr:+12.3f}")

# Overall bias
print(f"\n  Overall: mean_pred={p.mean():.3f}  mean_y={y.mean():.3f}  bias={p.mean()-y.mean():+.3f}")

print()
print("="*70)
print("Compare oracle5 vs oracle_NEW residuals by bucket")
print("="*70)
oracle5_o = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)
p5 = oracle5_o; pN = fw4_oo

print(f"\n  {'bucket':12s} {'n5':>7} {'resid_5way':>12} {'n_N':>7} {'resid_oN':>10} {'improvement':>12}")
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    m5 = (p5 >= lo) & (p5 < hi)
    mN = (pN >= lo) & (pN < hi)
    if m5.sum() > 0 and mN.sum() > 0:
        r5 = p5[m5].mean() - y[m5].mean()
        rN = pN[mN].mean() - y[mN].mean()
        improvement = r5 - rN  # positive = oracle_NEW better calibrated in this bucket
        print(f"  [{lo:3d},{hi:3d}): n5={m5.sum():7d} r5={r5:+10.3f}  nN={mN.sum():7d} rN={rN:+10.3f}  imp={improvement:+12.3f}")

print()
print("="*70)
print("Unseen test oracle_NEW predictions: bucket-based bias correction estimate")
print("="*70)

# Load oracle_NEW test
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

p_test_unseen = oracle_new_t[unseen_mask]

# For each bucket, compute expected bias correction based on train calibration
print(f"\nTest unseen predictions in each bucket with expected bias correction:")
print(f"  {'bucket':12s} {'n':>7} {'pred_mean':>10} {'train_resid':>12} {'corr_pred_mean':>15}")
expected_correction = np.zeros(len(p_test_unseen))
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask_train = (pN >= lo) & (pN < hi)
    mask_test  = (p_test_unseen >= lo) & (p_test_unseen < hi)
    if mask_train.sum() > 0 and mask_test.sum() > 0:
        train_resid = pN[mask_train].mean() - y[mask_train].mean()
        correction = -train_resid
        expected_correction[mask_test] = correction
        corr_mean = p_test_unseen[mask_test].mean() + correction
        print(f"  [{lo:3d},{hi:3d}): n={mask_test.sum():7d}  p_mean={p_test_unseen[mask_test].mean():10.3f}  "
              f"train_resid={train_resid:+12.3f}  corr_mean={corr_mean:15.3f}")

# Apply bucket-based correction to test
ct_corrected = oracle_new_t.copy()
ct_corrected[unseen_mask] = p_test_unseen + expected_correction
ct_corrected = np.clip(ct_corrected, 0, None)
print(f"\nBucket-corrected unseen test:")
print(f"  oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"  bucket_corr: seen={ct_corrected[seen_mask].mean():.3f}  unseen={ct_corrected[unseen_mask].mean():.3f}")
print(f"  correction applied: mean={expected_correction.mean():.3f}  std={expected_correction.std():.3f}")

# Also apply only partial correction (50%, 25%)
sub_tmpl = pd.read_csv('sample_submission.csv')
for frac, label in [(1.0, 'full'), (0.5, 'half'), (0.25, 'qtr'), (0.1, 'tenth')]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = p_test_unseen + frac * expected_correction
    ct = np.clip(ct, 0, None)
    print(f"  bucket_corr_{label} (frac={frac}): seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
    if frac in [0.25, 0.5]:
        fname = f"FINAL_NEW_oN_bucketCorr_{label}_OOF8.3825.csv"
        sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
        sub.to_csv(fname, index=False)
        print(f"    Saved: {fname}")

# Full correction
ct = ct_corrected
fname = f"FINAL_NEW_oN_bucketCorr_full_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Summary: training bias correction profile for oracle_NEW")
print("="*70)
# For the [30-40) bucket which contains 41% of unseen test:
mask_30_40 = (pN >= 30) & (pN < 40)
if mask_30_40.sum() > 0:
    ym = y[mask_30_40].mean()
    pm = pN[mask_30_40].mean()
    resid = pm - ym
    print(f"  [30-40) train: pred={pm:.3f}, true={ym:.3f}, resid={resid:+.3f}")
    print(f"  If test [30-40) has same bias: true test rows ≈ pred + {-resid:.3f} = pred + {-resid:.1f}")
    print(f"  oracle_NEW unseen [30-40) mean pred = {p_test_unseen[(p_test_unseen>=30)&(p_test_unseen<40)].mean():.3f}")
    print(f"  Implied true for these test rows ≈ {p_test_unseen[(p_test_unseen>=30)&(p_test_unseen<40)].mean() + (-resid):.3f}")

print("\nDone.")
