import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
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

print("="*70)
print("Scenario Difficulty Premium Analysis")
print("For seen layouts: oracle_NEW test - oof = scenario_difficulty_shift")
print("Hypothesis: unseen layouts with higher inflow have larger shifts")
print("="*70)

# Per seen-test layout: test_mean - oof_mean
seen_test_layouts = test_raw[seen_mask]['layout_id'].unique()
unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()

layout_shift = {}
layout_inflow_train = {}
layout_inflow_test  = {}

inflow_col = 'order_inflow_15m'
train_median_inflow = train_raw[inflow_col].median()
test_median_inflow = test_raw[inflow_col].median()

for lid in seen_test_layouts:
    tr_m = train_raw['layout_id'] == lid
    te_m = test_raw['layout_id'] == lid
    oof_m = fw4_oo[tr_m.values].mean()
    test_m = oracle_new_t[te_m.values].mean()
    y_m = y_true[tr_m.values].mean()
    # Inflow for this layout (train and test scenarios)
    inf_tr = train_raw.loc[tr_m, inflow_col].fillna(train_median_inflow).mean()
    inf_te = test_raw.loc[te_m, inflow_col].fillna(test_median_inflow).mean()
    layout_shift[lid] = {
        'oof_mean': oof_m,
        'test_mean': test_m,
        'y_mean_train': y_m,
        'shift': test_m - oof_m,  # test difficulty premium
        'inflow_train': inf_tr,
        'inflow_test': inf_te,
        'inflow_ratio': inf_te / (inf_tr + 1e-9),
    }

shifts = np.array([v['shift'] for v in layout_shift.values()])
inflow_train_l = np.array([v['inflow_train'] for v in layout_shift.values()])
inflow_test_l  = np.array([v['inflow_test'] for v in layout_shift.values()])
inflow_ratio_l = np.array([v['inflow_ratio'] for v in layout_shift.values()])
oof_means_l    = np.array([v['oof_mean'] for v in layout_shift.values()])

print(f"\n  Seen test layout scenario difficulty shift (test_mean - oof_mean):")
print(f"    mean={shifts.mean():+.4f}  std={shifts.std():.4f}  min={shifts.min():+.4f}  max={shifts.max():+.4f}")
print(f"\n  Inflow ratio (test / train) for seen layouts:")
print(f"    mean={inflow_ratio_l.mean():.4f}  min={inflow_ratio_l.min():.4f}  max={inflow_ratio_l.max():.4f}")

# Correlation: shift ~ inflow_ratio, inflow_test, inflow_train
for col_name, col_vals in [('inflow_test', inflow_test_l), ('inflow_train', inflow_train_l),
                            ('inflow_ratio', inflow_ratio_l), ('oof_mean', oof_means_l)]:
    r, p = pearsonr(col_vals, shifts)
    print(f"  r(shift, {col_name}) = {r:+.4f}  (p={p:.4f})")

# ============================================================
# Regression: shift ~ inflow features (for seen layouts)
# Then extrapolate to unseen test layouts
# ============================================================
print("\n" + "="*70)
print("Regression: shift ~ inflow features → extrapolate to unseen")
print("="*70)

# Use inflow_test and inflow_train as predictors
lr_shift = LinearRegression().fit(
    np.column_stack([inflow_test_l, inflow_train_l, inflow_ratio_l]),
    shifts
)
print(f"  Linear model: R²={lr_shift.score(np.column_stack([inflow_test_l, inflow_train_l, inflow_ratio_l]), shifts):.4f}")

# Get inflow for unseen test layouts
unseen_inflow_test  = {}
for lid in unseen_test_layouts:
    te_m = test_raw['layout_id'] == lid
    inf_te = test_raw.loc[te_m, inflow_col].fillna(test_median_inflow).mean()
    unseen_inflow_test[lid] = inf_te

# Since unseen layouts have NO training data, use the overall training mean
# as a proxy for what "similar" training scenarios would look like
# We'll estimate inflow_train from the overall training distribution
# and use the actual unseen test inflow
u_inf_test = np.array([v for v in unseen_inflow_test.values()])
u_inf_train_proxy = np.full_like(u_inf_test, train_raw[inflow_col].fillna(train_median_inflow).mean())
u_inf_ratio = u_inf_test / (u_inf_train_proxy + 1e-9)

print(f"\n  Unseen test inflow: mean={u_inf_test.mean():.2f}  min={u_inf_test.min():.2f}  max={u_inf_test.max():.2f}")
print(f"  Seen test inflow: mean={inflow_test_l.mean():.2f}")
print(f"  Train overall inflow mean: {u_inf_train_proxy.mean():.2f}")

pred_shift_unseen = lr_shift.predict(
    np.column_stack([u_inf_test, u_inf_train_proxy, u_inf_ratio])
)
print(f"\n  Predicted shift for unseen layouts: mean={pred_shift_unseen.mean():+.4f}")

# Simple single-feature regression
for col_name, col_vals, u_col in [
    ('inflow_test', inflow_test_l, u_inf_test),
    ('inflow_ratio', inflow_ratio_l, u_inf_ratio),
]:
    lr1 = LinearRegression().fit(col_vals.reshape(-1,1), shifts)
    r2 = lr1.score(col_vals.reshape(-1,1), shifts)
    pred_u = lr1.predict(u_col.reshape(-1,1))
    print(f"\n  shift ~ {col_name}: R²={r2:.4f}  coef={lr1.coef_[0]:.5f}  intercept={lr1.intercept_:.4f}")
    print(f"    Predicted shift for unseen: mean={pred_u.mean():+.4f}  std={pred_u.std():.4f}")

print("\n" + "="*70)
print("Scenario-adjusted correction: combine training bias + scenario shift")
print("="*70)
# training bias: y_true_train - oof_mean for each layout
training_bias = {}
for lid in seen_test_layouts:
    tr_m = train_raw['layout_id'] == lid
    training_bias[lid] = y_true[tr_m.values].mean() - fw4_oo[tr_m.values].mean()

biases = np.array([training_bias[lid] for lid in seen_test_layouts])
print(f"\n  Training bias for seen test layouts: mean={biases.mean():+.4f}  std={biases.std():.4f}")

# Regression: bias ~ inflow_train for seen layouts
lr_bias = LinearRegression().fit(inflow_train_l.reshape(-1,1), biases)
print(f"  bias ~ inflow_train: R²={lr_bias.score(inflow_train_l.reshape(-1,1), biases):.4f}")
print(f"    coef={lr_bias.coef_[0]:.4f}  intercept={lr_bias.intercept_:.4f}")

# Extrapolate for unseen layouts (using training mean inflow as proxy)
# Assume unseen layout's "training inflow" = similar to what we'd see if they were in training
# We have no direct info, so use test inflow as proxy for expected training inflow
pred_bias_unseen = lr_bias.predict(u_inf_test.reshape(-1,1))
print(f"\n  Predicted training bias for unseen (using test inflow as proxy):")
print(f"    mean={pred_bias_unseen.mean():+.4f}  std={pred_bias_unseen.std():.4f}")

# Combined correction = bias + shift
combined_corr = pred_bias_unseen + pred_shift_unseen
print(f"\n  Combined correction (bias + scenario shift): mean={combined_corr.mean():+.4f}")

# Apply layout-level corrections
ct_comb = oracle_new_t.copy()
for i, lid in enumerate(unseen_test_layouts):
    te_m = (test_raw['layout_id'] == lid).values
    ct_comb[te_m] = oracle_new_t[te_m] + combined_corr[i]
ct_comb = np.clip(ct_comb, 0, None)
du_comb = ct_comb[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Correction applied: seen={ct_comb[seen_mask].mean():.3f}  unseen={ct_comb[unseen_mask].mean():.3f}  Δ={du_comb:+.4f}")
fname_c = "FINAL_NEW_oN_scenarioCombined_OOF8.3825.csv"
sub_c = sub_tmpl.copy(); sub_c['avg_delay_minutes_next_30m'] = ct_comb
sub_c.to_csv(fname_c, index=False)
print(f"  Saved: {fname_c}")

# ============================================================
# FINAL: comprehensive comparison of all estimates
# ============================================================
print("\n" + "="*70)
print("CONSOLIDATED CORRECTION ESTIMATE")
print("="*70)

all_estimates = [
    ('Training bias only (seen layouts)', biases.mean()),
    ('Training bias only (all layouts)', -3.2066),
    ('Scenario shift (seen test)', shifts.mean()),
    ('Combined bias + shift (seen)', biases.mean() + shifts.mean()),
    ('Residual Ridge (row-level)', 4.1062),
    ('5-feat Ridge capped', 4.4126),
    ('Linear bias regression', 5.8447),
    ('Ridge layout features', 5.4218),
    ('Isotonic layout calib', 6.4810),
    ('Inflow regression Δ', 7.8515),
    ('y_mean ~ inflow extrap (172)', -2.4794 + 0.06*172),  # lr coef*172 + intercept
    ('High-inflow proxy (n=1000)', 8.4),
    ('Quintile match (nearby oof)', 5.32),
    ('Layout quintile top (oof=23.74)', 6.21),  # 29.95 - 23.74
]
print(f"\n  {'Method':50s}  {'Δ estimate':>12}")
for name, val in sorted(all_estimates, key=lambda x: x[1]):
    print(f"  {name:50s}  {val:+12.4f}")

print(f"""
  ─────────────────────────────────────────────────────────────────

  LOWER BOUND: training bias only = +3.2 to +3.5
  UPPER BOUND: inflow extrapolation = +8.4 to +8.6
  MEDIAN:      ~+5 to +6 (most principled methods)

  Recommended submission order (5 submissions remaining today?):
  1. oN_linearBias (Δ=+5.84) — data-driven middle estimate
  2. oN_udelta5 (Δ=+5.00) — round number in the cluster
  3. oN_iso_lv75 (Δ=+4.86) — conservative isotonic
  4. oN_quadBias_3qtr (Δ=+7.34) — test aggressive hypothesis
  5. oN_udelta8 (Δ=+8.00) — test very aggressive hypothesis
""")

print("Done.")
