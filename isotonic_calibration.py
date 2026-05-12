import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression

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
residuals_train = y_true - fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Isotonic Regression Calibration of oracle_NEW OOF → y_true")
print("="*70)

# ============================================================
# Method 1: Layout-level isotonic calibration
# Fit isotonic: layout_oof_mean → layout_y_mean
# ============================================================
print("\n--- Method 1: Layout-level isotonic calibration ---")
layout_oof_means = []
layout_y_means   = []
for lid in train_raw['layout_id'].unique():
    m = train_raw['layout_id'] == lid
    layout_oof_means.append(fw4_oo[m.values].mean())
    layout_y_means.append(y_true[m.values].mean())

layout_oof_means = np.array(layout_oof_means)
layout_y_means   = np.array(layout_y_means)

# Sort by oof for isotonic
sort_idx = np.argsort(layout_oof_means)
iso_x = layout_oof_means[sort_idx]
iso_y = layout_y_means[sort_idx]

iso = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso.fit(iso_x, iso_y)

# Show calibration table
print(f"\n  Layout oof → isotonic y calibration:")
for v in [5, 10, 12, 15, 17, 20, 22.72, 25, 30, 35, 40]:
    pred = iso.predict([v])[0]
    print(f"    oof={v:.2f} → y_hat={pred:.3f}  (correction={pred-v:+.3f})")

# Apply to unseen test predictions (row level)
corr_iso_layout = iso.predict(oracle_new_t[unseen_mask]) - oracle_new_t[unseen_mask]
ct_iso = oracle_new_t.copy()
ct_iso[unseen_mask] = oracle_new_t[unseen_mask] + corr_iso_layout
ct_iso = np.clip(ct_iso, 0, None)
du_iso = ct_iso[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Full isotonic correction: Δ={du_iso:+.4f}  seen={ct_iso[seen_mask].mean():.3f}  unseen={ct_iso[unseen_mask].mean():.3f}")

# Scaled versions
for scale, label in [(0.25,'iso_lv25'), (0.50,'iso_lv50'), (0.75,'iso_lv75'), (1.00,'iso_lv100')]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * corr_iso_layout
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  iso_layout*{scale:.2f}: Δ={du_s:+.4f}  seen={ct_s[seen_mask].mean():.3f}  unseen={ct_s[unseen_mask].mean():.3f}")
    fname = f"FINAL_NEW_oN_{label}_OOF8.3825.csv"
    sub_s = sub_tmpl.copy(); sub_s['avg_delay_minutes_next_30m'] = ct_s
    sub_s.to_csv(fname, index=False)
    print(f"    Saved: {fname}")

# ============================================================
# Method 2: Row-level isotonic calibration
# Direct: fw4_oo → y_true for all 250K training rows
# ============================================================
print("\n--- Method 2: Row-level isotonic calibration ---")
# Use a 1% sample for speed (full isotonic on 250K rows)
np.random.seed(42)
samp = np.random.choice(len(y_true), 50000, replace=False)
iso_row = IsotonicRegression(increasing=True, out_of_bounds='clip')
iso_row.fit(fw4_oo[samp], y_true[samp])

# Validate OOF MAE
pred_iso_oof = np.clip(iso_row.predict(fw4_oo), 0, None)
mae_iso = np.mean(np.abs(pred_iso_oof - y_true))
mae_orig = np.mean(np.abs(fw4_oo - y_true))
print(f"  Original OOF MAE: {mae_orig:.4f}")
print(f"  Isotonic OOF MAE: {mae_iso:.4f}  (BIASED — isotonic is trained on OOF itself)")

# Check isotonic mapping
print(f"\n  Row-level isotonic: oof → y calibration:")
for v in [5, 10, 15, 17, 20, 22.72, 25, 30]:
    pred = iso_row.predict([v])[0]
    print(f"    oof={v:.2f} → y_hat={pred:.3f}  (correction={pred-v:+.3f})")

# Apply to unseen test
corr_iso_row = iso_row.predict(oracle_new_t[unseen_mask]) - oracle_new_t[unseen_mask]
ct_iso_row = oracle_new_t.copy()
ct_iso_row[unseen_mask] = oracle_new_t[unseen_mask] + corr_iso_row
ct_iso_row = np.clip(ct_iso_row, 0, None)
du_iso_row = ct_iso_row[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Row-level isotonic correction: Δ={du_iso_row:+.4f}  seen={ct_iso_row[seen_mask].mean():.3f}  unseen={ct_iso_row[unseen_mask].mean():.3f}")

# ============================================================
# Method 3: Saved linear bias correction (from quintile_bias_correction.py)
# ============================================================
print("\n--- Method 3: Linear bias correction (bias ~ oof_mean, row-level) ---")
lr1 = LinearRegression().fit(layout_oof_means.reshape(-1,1), layout_y_means - layout_oof_means)
corr_linear = lr1.predict(oracle_new_t[unseen_mask].reshape(-1,1))
ct_lin = oracle_new_t.copy()
ct_lin[unseen_mask] = oracle_new_t[unseen_mask] + corr_linear
ct_lin = np.clip(ct_lin, 0, None)
du_lin = ct_lin[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Linear bias correction: Δ={du_lin:+.4f}  seen={ct_lin[seen_mask].mean():.3f}  unseen={ct_lin[unseen_mask].mean():.3f}")
fname_lin = "FINAL_NEW_oN_linearBias_OOF8.3825.csv"
sub_lin = sub_tmpl.copy(); sub_lin['avg_delay_minutes_next_30m'] = ct_lin
sub_lin.to_csv(fname_lin, index=False)
print(f"  Saved: {fname_lin}")

# Half
for scale, label in [(0.50,'linearBias_half'), (0.75,'linearBias_3qtr')]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * corr_linear
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_{label}_OOF8.3825.csv"
    sub_s = sub_tmpl.copy(); sub_s['avg_delay_minutes_next_30m'] = ct_s
    sub_s.to_csv(fname, index=False)
    print(f"  {label}: Δ={du_s:+.4f}  unseen={ct_s[unseen_mask].mean():.3f}  Saved: {fname}")

# ============================================================
# Summary table of ALL new correction candidates
# ============================================================
print("\n" + "="*70)
print("FINAL CANDIDATE SUMMARY (New from this session)")
print("="*70)
new_candidates = [
    ('FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv', 'Ridge layout features'),
    ('FINAL_NEW_oN_2featCorr_OOF8.3825.csv', '2-feat Ridge (pack+truck)'),
    ('FINAL_NEW_oN_5featCapped_OOF8.3825.csv', '5-feat Ridge capped'),
    ('FINAL_NEW_oN_linearBias_half_OOF8.3825.csv', 'Linear bias 50%'),
    ('FINAL_NEW_oN_quadBias_half_OOF8.3825.csv', 'Quad bias 50%'),
    ('FINAL_NEW_oN_iso_lv25_OOF8.3825.csv', 'Isotonic 25%'),
    ('FINAL_NEW_oN_iso_lv50_OOF8.3825.csv', 'Isotonic 50%'),
    ('FINAL_NEW_oN_linearBias_OOF8.3825.csv', 'Linear bias 100%'),
    ('FINAL_NEW_oN_linearBias_3qtr_OOF8.3825.csv', 'Linear bias 75%'),
    ('FINAL_NEW_oN_iso_lv75_OOF8.3825.csv', 'Isotonic 75%'),
    ('FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv', 'Ridge correction'),
    ('FINAL_NEW_oN_quadBias_3qtr_OOF8.3825.csv', 'Quad bias 75%'),
    ('FINAL_NEW_oN_iso_lv100_OOF8.3825.csv', 'Isotonic 100%'),
    ('FINAL_NEW_oN_quadBias_full_OOF8.3825.csv', 'Quad bias 100%'),
]
print(f"  {'file':50s}  {'seen':>8}  {'unseen':>8}  {'Δ':>9}")
for fname, label in new_candidates:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname[:50]:50s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}  [{label}]")
    except Exception as e:
        print(f"  {fname[:50]:50s}  MISSING")

print("\nDone.")
