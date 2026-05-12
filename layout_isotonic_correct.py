import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression

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
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

# ============================================================
# Layout-level: oof_mean → bias (layout isotonic, CORRECT usage)
# For each unseen test layout:
# 1. Compute oracle_NEW test mean for that layout
# 2. Use layout isotonic (oof_mean → y_mean) to predict y_mean for that layout
# 3. Correction = isotonic_pred(test_mean) - test_mean
# This is correct because test_mean ≈ oof_mean (both are oracle_NEW predictions)
# ============================================================
print("="*70)
print("LAYOUT ISOTONIC: per-layout correction via test_mean → y_pred")
print("Training: oof_mean → y_mean, applied at TEST LAYOUT MEAN level")
print("="*70)

layout_oof_means = []
layout_y_means   = []
layout_ids_tr    = []
for lid in train_raw['layout_id'].unique():
    m = train_raw['layout_id'] == lid
    layout_oof_means.append(fw4_oo[m.values].mean())
    layout_y_means.append(y_true[m.values].mean())
    layout_ids_tr.append(lid)
layout_oof_means = np.array(layout_oof_means)
layout_y_means   = np.array(layout_y_means)

# Isotonic regression: oof_mean → y_mean
sort_idx = np.argsort(layout_oof_means)
iso_lv = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso_lv.fit(layout_oof_means[sort_idx], layout_y_means[sort_idx])

# For each unseen test layout: compute test_mean, predict y_mean, compute correction
unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()
seen_test_layouts   = test_raw[seen_mask]['layout_id'].unique()

# Per-layout test mean
layout_test_means_u = {}
for lid in unseen_test_layouts:
    te_m = test_raw['layout_id'] == lid
    layout_test_means_u[lid] = oracle_new_t[te_m.values].mean()

layout_test_means_s = {}
for lid in seen_test_layouts:
    te_m = test_raw['layout_id'] == lid
    layout_test_means_s[lid] = oracle_new_t[te_m.values].mean()

u_test_means = np.array([layout_test_means_u[lid] for lid in unseen_test_layouts])
s_test_means = np.array([layout_test_means_s[lid] for lid in seen_test_layouts])

print(f"\n  Unseen layout oracle_NEW test means:")
print(f"    min={u_test_means.min():.3f}  max={u_test_means.max():.3f}  mean={u_test_means.mean():.3f}")
print(f"    (training layout oof range: {layout_oof_means.min():.3f} to {layout_oof_means.max():.3f})")
print(f"  Unseen layouts OUTSIDE training oof range: {(u_test_means > layout_oof_means.max()).sum()}")

# Isotonic prediction
iso_pred_u = iso_lv.predict(u_test_means)
iso_corr_u = iso_pred_u - u_test_means
iso_pred_s = iso_lv.predict(s_test_means)
iso_corr_s = iso_pred_s - s_test_means

print(f"\n  Per-layout isotonic correction for UNSEEN test layouts:")
print(f"    mean correction={iso_corr_u.mean():+.4f}  std={iso_corr_u.std():.4f}")
print(f"    test_mean range: [{u_test_means.min():.3f}, {u_test_means.max():.3f}]")
print(f"    iso_pred range: [{iso_pred_u.min():.3f}, {iso_pred_u.max():.3f}]")

print(f"\n  Per-layout isotonic correction for SEEN test layouts (sanity check):")
print(f"    mean correction={iso_corr_s.mean():+.4f}")
print(f"    (should be ~+3.5, the training bias for seen layouts)")

# Show per-layout corrections for unseen
print(f"\n  Unseen layout-level corrections (top 10 by test_mean):")
lid_sort = np.argsort(u_test_means)[::-1]
for i in lid_sort[:10]:
    lid = unseen_test_layouts[i]
    print(f"    layout={lid}: test_mean={u_test_means[i]:.3f}  iso_pred={iso_pred_u[i]:.3f}  corr={iso_corr_u[i]:+.3f}")

# Apply per-layout correction
ct_iso_lv = oracle_new_t.copy()
for i, lid in enumerate(unseen_test_layouts):
    te_m = (test_raw['layout_id'] == lid).values
    ct_iso_lv[te_m] = oracle_new_t[te_m] + iso_corr_u[i]
ct_iso_lv = np.clip(ct_iso_lv, 0, None)
du_iso_lv = ct_iso_lv[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Full layout-isotonic correction: Δ={du_iso_lv:+.4f}  seen={ct_iso_lv[seen_mask].mean():.3f}  unseen={ct_iso_lv[unseen_mask].mean():.3f}")

# Save
for scale, label in [(0.50,'iso_lvL50'), (0.75,'iso_lvL75'), (1.00,'iso_lvL100')]:
    ct_s = oracle_new_t.copy()
    for i, lid in enumerate(unseen_test_layouts):
        te_m = (test_raw['layout_id'] == lid).values
        ct_s[te_m] = oracle_new_t[te_m] + scale * iso_corr_u[i]
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_{label}_OOF8.3825.csv"
    sub_s = sub_tmpl.copy(); sub_s['avg_delay_minutes_next_30m'] = ct_s
    sub_s.to_csv(fname, index=False)
    print(f"  {label}*{scale:.2f}: Δ={du_s:+.4f}  unseen={ct_s[unseen_mask].mean():.3f}  → {fname}")

# ============================================================
# Compare: correct at LAYOUT level vs ROW level
# Row-level: isotonic(individual_row_pred) → overcorrects high-pred rows
# Layout-level: isotonic(layout_mean_pred) → uniform correction per layout
# ============================================================
print("\n" + "="*70)
print("COMPARISON: Row-level vs Layout-level isotonic correction")
print("="*70)
# Row-level (from isotonic_calibration.py)
corr_iso_rowlevel = iso_lv.predict(oracle_new_t[unseen_mask]) - oracle_new_t[unseen_mask]
ct_iso_row = oracle_new_t.copy()
ct_iso_row[unseen_mask] = oracle_new_t[unseen_mask] + corr_iso_rowlevel
ct_iso_row = np.clip(ct_iso_row, 0, None)
du_row = ct_iso_row[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Row-level isotonic correction: Δ={du_row:+.4f}  unseen={ct_iso_row[unseen_mask].mean():.3f}")
print(f"    (row: each row corrected based on its own prediction)")
print(f"  Layout-level isotonic correction: Δ={du_iso_lv:+.4f}  unseen={ct_iso_lv[unseen_mask].mean():.3f}")
print(f"    (layout: each layout corrected based on its mean prediction)")
print(f"\n  Difference: row-level applies LARGER corrections to high-pred rows,")
print(f"  which inflates variance. Layout-level applies UNIFORM correction per layout.")

# ============================================================
# FINAL TABLE: All principled candidates for submission
# ============================================================
print("\n" + "="*70)
print("SUBMISSION PRIORITY — PRINCIPLED ESTIMATES ONLY")
print("="*70)
files_to_show = [
    ('submission_oracle_NEW_OOF8.3825.csv', 'BASELINE LB=9.7527'),
    ('FINAL_NEW_oN_iso_lv25_OOF8.3825.csv', 'Iso layout 25%'),
    ('FINAL_NEW_oN_iso_lvL50_OOF8.3825.csv', 'Iso layout-LVL 50%'),
    ('FINAL_NEW_oN_residRidge_half_OOF8.3825.csv', 'Residual Ridge 50%'),
    ('FINAL_NEW_oN_iso_lv50_OOF8.3825.csv', 'Iso layout 50%'),
    ('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv', 'Iso layout-LVL 75%'),
    ('FINAL_NEW_oN_quadBias_half_OOF8.3825.csv', 'Quad bias 50%'),
    ('FINAL_NEW_oN_linearBias_3qtr_OOF8.3825.csv', 'Linear bias 75%'),
    ('FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv', 'Ridge layout'),
    ('FINAL_NEW_oN_iso_lvL100_OOF8.3825.csv', 'Iso layout-LVL 100%'),
    ('FINAL_NEW_oN_linearBias_OOF8.3825.csv', 'Linear bias 100%'),
    ('FINAL_NEW_oN_iso_lv75_OOF8.3825.csv', 'Iso layout 75%'),
    ('FINAL_NEW_oN_iso_lv100_OOF8.3825.csv', 'Iso layout 100%'),
    ('FINAL_NEW_oN_udelta7_OOF8.3825.csv', 'Flat +7'),
    ('FINAL_NEW_oN_udelta8_OOF8.3825.csv', 'Flat +8'),
]
print(f"  {'file':52s}  {'seen':>8}  {'unseen':>8}  {'Δ':>9}")
for fname, label in files_to_show:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname[:50]:50s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}  {label}")
    except Exception as e:
        print(f"  {fname[:50]:50s}  MISSING")

print("\nDone.")
