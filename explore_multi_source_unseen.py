import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os
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

# OOF data
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)
h10_o = np.clip(np.load('results/oracle_seq/oof_seqC_huber10.npy'), 0, None)
spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
spec_avg_o = np.clip(np.load('results/cascade/spec_avg_oof.npy')[id2], 0, None)
lgbs_t = np.clip(np.load('results/oracle_seq/test_C_lgb_stack.npy'), 0, None)
lgbs_o = np.clip(np.load('results/oracle_seq/oof_seqC_lgb_stack.npy'), 0, None)

sub_tmpl = pd.read_csv('sample_submission.csv')

print("Model reference stats:")
print(f"  oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"  h10:        seen={h10_t[seen_mask].mean():.3f}  unseen={h10_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(h10_t,oracle_new_t)[0]:.4f}")
print(f"  spec_avg:   seen={spec_avg_t[seen_mask].mean():.3f}  unseen={spec_avg_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(spec_avg_t,oracle_new_t)[0]:.4f}")
print(f"  lgb_stack:  seen={lgbs_t[seen_mask].mean():.3f}  unseen={lgbs_t[unseen_mask].mean():.3f}  r(oN)={pearsonr(lgbs_t,oracle_new_t)[0]:.4f}")
print(f"  r(h10,spec_avg)={pearsonr(h10_t,spec_avg_t)[0]:.4f}  r(h10,lgbs)={pearsonr(h10_t,lgbs_t)[0]:.4f}  r(spec,lgbs)={pearsonr(spec_avg_t,lgbs_t)[0]:.4f}")

# For OOF: check cross-model correlations
print(f"\n  OOF: h10={mae_fn(h10_o):.5f}  spec_avg={mae_fn(spec_avg_o):.5f}  lgbs={mae_fn(lgbs_o):.5f}")
print(f"  r(h10_oof,spec_oof)={pearsonr(h10_o,spec_avg_o)[0]:.4f}  r(h10_oof,lgbs_oof)={pearsonr(h10_o,lgbs_o)[0]:.4f}")

# ============================================================
# Unseen-only blends with multiple sources
# ============================================================
print("\n" + "="*70)
print("Unseen-only blend: oracle_NEW + h10 + spec_avg (3-way)")
print("="*70)
# avg of h10 + spec_avg as the calibration target
h10_specAvg_avg = (h10_t + spec_avg_t) / 2
print(f"h10+spec_avg avg: seen={h10_specAvg_avg[seen_mask].mean():.3f}  unseen={h10_specAvg_avg[unseen_mask].mean():.3f}  r(oN)={pearsonr(h10_specAvg_avg,oracle_new_t)[0]:.4f}")

for w_u in [0.10, 0.15, 0.20, 0.30, 0.50]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w_u)*oracle_new_t[unseen_mask] + w_u*h10_specAvg_avg[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w_u={w_u:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# Hybrid h10 (below40) + spec_avg for above40
# ============================================================
print("\n" + "="*70)
print("Hybrid: h10 for unseen<40, spec_avg for unseen>=40")
print("spec_avg is better than h10 for [40+) — checking...")
print("="*70)
below40_mask = unseen_mask & (oracle_new_t < 40)
above40_mask = unseen_mask & (oracle_new_t >= 40)

# spec_avg vs h10 for above40 unseen rows
diff_spec_40 = spec_avg_t[above40_mask] - oracle_new_t[above40_mask]
diff_h10_40  = h10_t[above40_mask] - oracle_new_t[above40_mask]
print(f"  above40 unseen (n={above40_mask.sum()}):")
print(f"    h10-oN   mean={diff_h10_40.mean():.3f}  (h10 vs oracle_NEW for [40+))")
print(f"    spec-oN  mean={diff_spec_40.mean():.3f}  (spec_avg vs oracle_NEW for [40+))")

# Full hybrid: h10 for <40, spec_avg for >=40
for w_lo, w_hi in [(0.5,0.1), (0.5,0.2), (1.0,0.1), (1.0,0.2), (1.0,0.5)]:
    ct = oracle_new_t.copy()
    ct[below40_mask] = (1-w_lo)*oracle_new_t[below40_mask] + w_lo*h10_t[below40_mask]
    ct[above40_mask] = (1-w_hi)*oracle_new_t[above40_mask] + w_hi*spec_avg_t[above40_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    ds = ct[seen_mask].mean() - oracle_new_t[seen_mask].mean()
    print(f"  h10<40 w={w_lo} + spec>=40 w={w_hi}: seen={ct[seen_mask].mean():.3f}({ds:+.3f})  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# Best combo: oracle_NEW OOF + multi-source blend
# ============================================================
print("\n" + "="*70)
print("OOF check: oracle_NEW + h10 + spec_avg blends")
print("(using oracle_NEW OOF proxy for blends)")
print("="*70)

import pickle
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

# Check OOF impact of different blends
for label, src_o in [('h10', h10_o), ('spec_avg', spec_avg_o)]:
    for w in [0.05, 0.10, 0.20]:
        blend_oof = np.clip((1-w)*fw4_oo + w*src_o, 0, None)
        print(f"  oracle_NEW+{label} w={w}: OOF={mae_fn(blend_oof):.5f}  ΔOOF={mae_fn(blend_oof)-8.37624:+.5f}")

# ============================================================
# Per-bucket spec_avg vs oracle_NEW comparison
# ============================================================
print("\n" + "="*70)
print("spec_avg vs oracle_NEW per-bucket (unseen test)")
print("="*70)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
p_oN_u = oracle_new_t[unseen_mask]
p_spec_u = spec_avg_t[unseen_mask]
p_h10_u = h10_t[unseen_mask]
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN_u >= lo) & (p_oN_u < hi)
    if mask.sum() > 0:
        diff_spec = (p_spec_u[mask]-p_oN_u[mask]).mean()
        diff_h10  = (p_h10_u[mask]-p_oN_u[mask]).mean()
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  oN={p_oN_u[mask].mean():.3f}  spec_diff={diff_spec:+.3f}  h10_diff={diff_h10:+.3f}")

# ============================================================
# Save best multi-source candidates
# ============================================================
print("\n" + "="*70)
print("Save multi-source candidates")
print("="*70)

# spec_avg unseen only at w=0.15 and w=0.25
for w in [0.15, 0.25]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*spec_avg_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_specAvg_u{int(w*100)}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# h10+spec_avg avg, unseen only
for w in [0.15, 0.25]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*h10_specAvg_avg[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_h10spec_u{int(w*100)}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# hybrid h10(below40) + spec(above40), w_lo=0.5, w_hi=0.2
ct = oracle_new_t.copy()
ct[below40_mask] = 0.5*oracle_new_t[below40_mask] + 0.5*h10_t[below40_mask]
ct[above40_mask] = 0.8*oracle_new_t[above40_mask] + 0.2*spec_avg_t[above40_mask]
ct = np.clip(ct, 0, None)
du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname = "FINAL_NEW_oN_hybh10spec50_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

print("\nDone.")
