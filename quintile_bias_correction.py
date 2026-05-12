import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

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

# ============================================================
# Layout-level bias as function of OOF mean prediction
# ============================================================
print("="*70)
print("Layout-level bias (y_true - oof) as function of oof prediction")
print("KEY: oracle_NEW unseen test mean = 22.716, ≈ top quintile oof mean (22.76)")
print("="*70)

layout_biases = {}
for lid in train_raw['layout_id'].unique():
    tr_mask = train_raw['layout_id'] == lid
    oof_mean_lid = fw4_oo[tr_mask.values].mean()
    y_mean_lid = y_true[tr_mask.values].mean()
    layout_biases[lid] = {
        'oof_mean': oof_mean_lid,
        'y_mean': y_mean_lid,
        'bias': y_mean_lid - oof_mean_lid,  # positive = underprediction
    }

all_oof = np.array([v['oof_mean'] for v in layout_biases.values()])
all_bias = np.array([v['bias'] for v in layout_biases.values()])
all_y = np.array([v['y_mean'] for v in layout_biases.values()])

# Quintile of oof_mean
bins = np.percentile(all_oof, [0, 20, 40, 60, 80, 100])
print(f"\n  {'oof_mean_range':22s}  {'n':>4}  {'bias':>10}  {'y_true':>8}  {'oof_mean':>8}")
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (all_oof >= lo) & (all_oof <= hi)
    if m.sum() > 0:
        print(f"  [{lo:6.2f},{hi:6.2f}): n={m.sum():4d}  bias={all_bias[m].mean():+10.4f}  "
              f"y={all_y[m].mean():8.3f}  oof={all_oof[m].mean():8.3f}")

# Specific value: layouts with oof_mean near 22.716
target_oof = 22.716
for delta in [2, 5, 10]:
    m = np.abs(all_oof - target_oof) < delta
    if m.sum() > 0:
        print(f"\n  Layouts with |oof_mean - {target_oof}| < {delta}: n={m.sum()}")
        print(f"    bias (y - oof): {all_bias[m].mean():+.4f}")
        print(f"    y_true mean:    {all_y[m].mean():.4f}")
        print(f"    oof mean:       {all_oof[m].mean():.4f}")
        print(f"    Implied test unseen y_true ≈ {target_oof + all_bias[m].mean():.4f}")

# ============================================================
# Regression: bias ~ oof_mean (layout level)
# ============================================================
print("\n" + "="*70)
print("Regression: bias ~ oof_mean (linear + quadratic at layout level)")
print("="*70)

lr1 = LinearRegression().fit(all_oof.reshape(-1,1), all_bias)
print(f"  Linear: bias = {lr1.coef_[0]:.4f}*oof + {lr1.intercept_:.4f}  R²={lr1.score(all_oof.reshape(-1,1),all_bias):.4f}")
print(f"  Predicted bias at oof=22.716: {lr1.predict([[target_oof]])[0]:+.4f}")

poly = PolynomialFeatures(2)
X2 = poly.fit_transform(all_oof.reshape(-1,1))
lr2 = LinearRegression().fit(X2, all_bias)
print(f"  Quadratic R²={lr2.score(X2, all_bias):.4f}")
print(f"  Predicted bias at oof=22.716: {lr2.predict(poly.transform([[target_oof]]))[0]:+.4f}")

# For key OOF values
print(f"\n  Bias predictions at key oof values:")
for oof_val in [10, 15, 17, 20, 22.72, 25, 30]:
    b1 = lr1.predict([[oof_val]])[0]
    b2 = lr2.predict(poly.transform([[oof_val]]))[0]
    print(f"    oof={oof_val:6.2f}: linear={b1:+.3f}  quadratic={b2:+.3f}")

# ============================================================
# Apply regression-based correction to test unseen rows
# ============================================================
print("\n" + "="*70)
print("Row-level regression bias correction applied to unseen test")
print("bias_hat = f(oracle_NEW_pred) from layout-level regression")
print("This is justified: oracle_NEW test ≈ oracle_NEW oof for same pred range")
print("="*70)
# The correction magnitude for each unseen test row is:
# correction = bias_hat(oracle_NEW_pred_i) = linear or quadratic function

# Linear correction
corr_linear = lr1.predict(oracle_new_t[unseen_mask].reshape(-1,1))
ct_lin = oracle_new_t.copy()
ct_lin[unseen_mask] = oracle_new_t[unseen_mask] + corr_linear
ct_lin = np.clip(ct_lin, 0, None)
du_lin = ct_lin[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Linear bias correction: Δ={du_lin:+.4f}  seen={ct_lin[seen_mask].mean():.3f}  unseen={ct_lin[unseen_mask].mean():.3f}")

# Quadratic correction
corr_quad = lr2.predict(poly.transform(oracle_new_t[unseen_mask].reshape(-1,1)))
ct_quad = oracle_new_t.copy()
ct_quad[unseen_mask] = oracle_new_t[unseen_mask] + corr_quad
ct_quad = np.clip(ct_quad, 0, None)
du_quad = ct_quad[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Quadratic bias correction: Δ={du_quad:+.4f}  seen={ct_quad[seen_mask].mean():.3f}  unseen={ct_quad[unseen_mask].mean():.3f}")

# Scaled versions
for scale in [0.25, 0.50, 0.75, 1.00]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * corr_quad
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  Quad*{scale:.2f}: Δ={du_s:+.4f}  seen={ct_s[seen_mask].mean():.3f}  unseen={ct_s[unseen_mask].mean():.3f}")

# Save key versions
for scale, label in [(0.50, 'quadBias_half'), (0.75, 'quadBias_3qtr'), (1.00, 'quadBias_full')]:
    ct_s = oracle_new_t.copy()
    ct_s[unseen_mask] = oracle_new_t[unseen_mask] + scale * corr_quad
    ct_s = np.clip(ct_s, 0, None)
    du_s = ct_s[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_{label}_OOF8.3825.csv"
    sub_s = sub_tmpl.copy(); sub_s['avg_delay_minutes_next_30m'] = ct_s
    sub_s.to_csv(fname, index=False)
    print(f"  Saved: {fname}  Δ={du_s:+.4f}")

# ============================================================
# Summary: convergence of estimates
# ============================================================
print("\n" + "="*70)
print("SUMMARY: Convergence of correction estimates")
print("="*70)
print(f"""
  Method                          Δ estimate
  ─────────────────────────────────────────
  Training bias (all layouts)     +3.2
  Seen layout training bias       +3.5
  Scenario shift (seen test)      +0.9
  Combined (bias + shift)         +4.4
  Linear bias ~ oof regression    {du_lin:+.1f}
  Quadratic bias ~ oof regression {du_quad:+.1f}
  Inflow regression extrapolation +7.9
  y_mean ~ inflow extrapolation   +8.6
  High-inflow proxy (2 layouts)   +8.4
  Quintile match (oof≈22.76)      +8.6
  ─────────────────────────────────────────
  Cluster 1 (conservative):       ~+3.2 to +5.0
  Cluster 2 (aggressive):         ~+7.9 to +8.6

  CRITICAL QUESTION: Is oracle_NEW unseen test in the SAME regime
  as the top training quintile (oof=22.76, bias=-8.59)?

  Evidence FOR: oracle_NEW unseen pred = 22.716 ≈ 22.76 (near-exact match)
  Evidence AGAINST: top quintile layouts are the MOST EXTREME training
    layouts; unseen test might be easier than the hardest training layouts
    even with higher inflow (different layout structure).
""")

print("\nDone.")
