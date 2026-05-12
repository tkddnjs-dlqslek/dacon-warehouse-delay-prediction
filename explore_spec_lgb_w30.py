import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Load models
w30mae_o = np.clip(np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2], 0, None)
w30mae_t = np.clip(np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2], 0, None)
w30hub_o = np.clip(np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2], 0, None)
w30hub_t = np.clip(np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2], 0, None)
spec_avg_o = np.clip(np.load('results/cascade/spec_avg_oof.npy')[id2], 0, None)
spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)

# Rebuild oracle_NEW OOF
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)

sub_tmpl = pd.read_csv('sample_submission.csv')

print(f"w30mae OOF={mae_fn(w30mae_o):.5f}  seen={w30mae_t[seen_mask].mean():.3f}  unseen={w30mae_t[unseen_mask].mean():.3f}")
print(f"w30hub OOF={mae_fn(w30hub_o):.5f}  seen={w30hub_t[seen_mask].mean():.3f}  unseen={w30hub_t[unseen_mask].mean():.3f}")
print(f"spec_avg OOF={mae_fn(spec_avg_o):.5f}  unseen={spec_avg_t[unseen_mask].mean():.3f}")
print(f"oracle_NEW OOF={mae_fn(fw4_oo):.5f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

# ============================================================
# Per-bucket calibration analysis on TRAINING
# ============================================================
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
y = y_true
p_oN = fw4_oo
p_w30 = w30mae_o
p_hub = w30hub_o
p_avg = spec_avg_o

print("\n" + "="*70)
print("Per-bucket OOF residuals: oracle_NEW vs w30mae vs w30hub vs spec_avg")
print("(bucketed by oracle_NEW prediction)")
print("="*70)
print(f"\n  {'bucket':12s} {'n':>7} {'y_mean':>8} {'oN_r':>8} {'w30mae_r':>10} {'w30hub_r':>10} {'avg_r':>8}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN >= lo) & (p_oN < hi)
    if mask.sum() > 0:
        ym = y[mask].mean()
        r_oN = p_oN[mask].mean() - ym
        r_w30 = p_w30[mask].mean() - ym
        r_hub = p_hub[mask].mean() - ym
        r_avg = p_avg[mask].mean() - ym
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y={ym:8.3f}  oN={r_oN:+8.3f}  w30mae={r_w30:+10.3f}  w30hub={r_hub:+10.3f}  avg={r_avg:+8.3f}")

# ============================================================
# Key: per-bucket MAE comparison
# ============================================================
print("\n" + "="*70)
print("Per-bucket MAE: oracle_NEW vs w30mae vs w30hub vs spec_avg")
print("="*70)
print(f"\n  {'bucket':12s} {'n':>7} {'oN_mae':>8} {'w30mae_mae':>11} {'w30hub_mae':>11} {'avg_mae':>9}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN >= lo) & (p_oN < hi)
    if mask.sum() > 0:
        mae_oN = np.mean(np.abs(p_oN[mask]-y[mask]))
        mae_w30 = np.mean(np.abs(p_w30[mask]-y[mask]))
        mae_hub = np.mean(np.abs(p_hub[mask]-y[mask]))
        mae_avg = np.mean(np.abs(p_avg[mask]-y[mask]))
        w30_marker = " ← BETTER" if mae_w30 < mae_oN else ""
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  oN={mae_oN:8.3f}  w30mae={mae_w30:11.3f}  w30hub={mae_hub:11.3f}  avg={mae_avg:9.3f}{w30_marker}")

# ============================================================
# OOF blend analysis: oracle_NEW + w30mae
# ============================================================
print("\n" + "="*70)
print("OOF blend: oracle_NEW + spec_lgb_w30_mae")
print("="*70)
for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
    blend_oof = np.clip((1-w)*fw4_oo + w*w30mae_o, 0, None)
    blend_test = np.clip((1-w)*oracle_new_t + w*w30mae_t, 0, None)  # full blend
    blend_u = oracle_new_t.copy()
    blend_u[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
    oof_mae = mae_fn(blend_oof)
    print(f"  w={w:.2f}: OOF={oof_mae:.5f}(Δ={oof_mae-8.37624:+.5f})  full_unseen={blend_test[unseen_mask].mean():.3f}  u_only_unseen={blend_u[unseen_mask].mean():.3f}")

# ============================================================
# Cross-model ensemble: w30mae + spec_avg as joint unseen signal
# ============================================================
print("\n" + "="*70)
print("3-way unseen blend: oracle_NEW + spec_avg + w30mae")
print("="*70)
for w1, w2 in [(0.10,0.10), (0.10,0.05), (0.05,0.10), (0.20,0.05), (0.05,0.20)]:
    blend_u = oracle_new_t.copy()
    blend_u[unseen_mask] = (1-w1-w2)*oracle_new_t[unseen_mask] + w1*spec_avg_t[unseen_mask] + w2*w30mae_t[unseen_mask]
    blend_u = np.clip(blend_u, 0, None)
    du = blend_u[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  spAvg={w1} w30mae={w2}: seen={blend_u[seen_mask].mean():.3f}  unseen={blend_u[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# Optimal magnitude: find w* that minimizes OOF for blend
# ============================================================
print("\n" + "="*70)
print("Fine-grained w sweep: oracle_NEW + w30mae unseen-only")
print("(checking OOF impact of unseen-only blend doesn't change OOF)")
print("="*70)
# Note: unseen-only blend doesn't change TRAINING OOF (training has no unseen rows!)
# So OOF is unchanged for any unseen-only blend
# The "OOF" check above was for full blend (both seen and unseen)
print("Note: unseen-only blend does NOT change training OOF (0 unseen rows in train)")
print("The only metric to check is test distribution quality")
for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w={w:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# Save key w30mae candidates
# ============================================================
print("\n" + "="*70)
print("Saving w30mae unseen-only blend candidates")
print("="*70)
for w in [0.02, 0.05, 0.08]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_w30mae_u{int(w*100):02d}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  {fname}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# Also: 3-way blend at conservative weights
ct = oracle_new_t.copy()
ct[unseen_mask] = 0.80*oracle_new_t[unseen_mask] + 0.10*spec_avg_t[unseen_mask] + 0.10*w30mae_t[unseen_mask]
ct = np.clip(ct, 0, None)
du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname = "FINAL_NEW_oN_3way10_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"  {fname}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

print("\nDone.")
