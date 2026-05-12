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
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo + test
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

# Load test predictions
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

# ============================================================
# KEY ANALYSIS: OOF vs Test calibration gap for SEEN layouts
# ============================================================
print("="*70)
print("Calibration gap: oracle_NEW OOF vs Test for SEEN layouts")
print("="*70)

seen_test_layouts = test_raw[seen_mask]['layout_id'].unique()

# For each seen test layout, compute:
# (a) oracle_NEW OOF mean = prediction on training rows for that layout
# (b) oracle_NEW test mean = prediction on test rows for that layout
# (c) y_true mean = actual training outcome for that layout
# If (a) ≈ (c) → oracle_NEW is well calibrated for seen layouts
# If (b) < (c) → oracle_NEW underpredicts in test even for seen layouts

layout_oof_stats = {}
for lid in seen_test_layouts:
    tr_mask = train_raw['layout_id'] == lid
    te_mask = test_raw['layout_id'] == lid
    if tr_mask.sum() > 0 and te_mask.sum() > 0:
        layout_oof_stats[lid] = {
            'y_mean': y_true[tr_mask.values].mean(),
            'oof_mean': fw4_oo[tr_mask.values].mean(),
            'test_mean': oracle_new_t[te_mask.values].mean(),
            'n_train': tr_mask.sum(),
            'n_test': te_mask.sum(),
        }

y_means = np.array([v['y_mean'] for v in layout_oof_stats.values()])
oof_means = np.array([v['oof_mean'] for v in layout_oof_stats.values()])
test_means = np.array([v['test_mean'] for v in layout_oof_stats.values()])

print(f"\n  Seen test layouts: {len(layout_oof_stats)}")
print(f"\n  y_true mean (train): {y_means.mean():.4f}")
print(f"  OOF mean (train):    {oof_means.mean():.4f}  bias={oof_means.mean()-y_means.mean():+.4f}")
print(f"  Test mean (test):    {test_means.mean():.4f}")
print(f"  Test vs y_true gap:  {test_means.mean()-y_means.mean():+.4f}")
print(f"  OOF-to-Test shift:   {test_means.mean()-oof_means.mean():+.4f}")

# For unseen test layouts — compare test mean vs implied y from layout regression
print("\n" + "="*70)
print("Training: y_true vs OOF bias by layout (ALL 250 train layouts)")
print("="*70)
layout_biases = {}
for lid in train_raw['layout_id'].unique():
    tr_mask = train_raw['layout_id'] == lid
    layout_biases[lid] = {
        'y_mean': y_true[tr_mask.values].mean(),
        'oof_mean': fw4_oo[tr_mask.values].mean(),
        'bias': fw4_oo[tr_mask.values].mean() - y_true[tr_mask.values].mean(),
        'n': tr_mask.sum()
    }

all_biases = np.array([v['bias'] for v in layout_biases.values()])
all_y = np.array([v['y_mean'] for v in layout_biases.values()])
all_oof = np.array([v['oof_mean'] for v in layout_biases.values()])
print(f"  Mean layout-level OOF bias: {all_biases.mean():+.4f}  (pred - y_true)")
print(f"  = mean underprediction: {-all_biases.mean():+.4f}")
print(f"  Std: {all_biases.std():.4f}")
print(f"  %layouts with negative bias (underprediction): {100*(all_biases < 0).mean():.1f}%")

# Distribution of biases
print(f"\n  Layout bias distribution:")
for pct in [5, 25, 50, 75, 95]:
    print(f"    p{pct}: {np.percentile(all_biases, pct):+.4f}")

# ============================================================
# Key insight: what is the OOF bias pattern for layouts
# sorted by y_true mean?
# ============================================================
print("\n" + "="*70)
print("Layout OOF bias vs y_true mean (quintiles)")
print("="*70)
bins = np.percentile(all_y, [0, 20, 40, 60, 80, 100])
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (all_y >= lo) & (all_y < hi)
    if m.sum() > 0:
        print(f"  y=[{lo:.1f},{hi:.1f}): n={m.sum()}  mean_bias={all_biases[m].mean():+.4f}  "
              f"(y_mean={all_y[m].mean():.2f}  oof_mean={all_oof[m].mean():.2f})")

# ============================================================
# Scenario-level analysis: do unseen test scenarios look like
# high-y_true training scenarios?
# ============================================================
print("\n" + "="*70)
print("Scenario-level: oracle_NEW test prediction distribution")
print("(seen vs unseen test)")
print("="*70)
bins_pred = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
print(f"  {'bucket':12s}  {'n_seen':>8}  {'%_seen':>8}  {'n_unseen':>10}  {'%_unseen':>10}")
for lo, hi in zip(bins_pred[:-1], bins_pred[1:]):
    ns = ((oracle_new_t[seen_mask] >= lo) & (oracle_new_t[seen_mask] < hi)).sum()
    nu = ((oracle_new_t[unseen_mask] >= lo) & (oracle_new_t[unseen_mask] < hi)).sum()
    ps = 100*ns/seen_mask.sum()
    pu = 100*nu/unseen_mask.sum()
    print(f"  [{lo:3d},{hi:3d}): {ns:8d}  {ps:7.1f}%  {nu:10d}  {pu:9.1f}%")

# ============================================================
# The OOF-to-Test shift for seen layouts
# If test scenarios are harder than training scenarios for same layout,
# then oracle_NEW will also underpredict test seen rows
# ============================================================
print("\n" + "="*70)
print("OOF vs Test comparison for seen layouts: scenario-level")
print("Since ALL test scenarios are NEW (0% overlap), test y_true for seen")
print("layouts might be higher than OOF y_true for same layouts")
print("="*70)
# Cannot compute directly (no test y_true), but can check:
# (1) Distribution of oracle_NEW for seen train vs seen test
# (2) If test predictions are higher → harder scenarios → y_true will be higher too
seen_train_preds = []
seen_test_preds  = []
for lid in seen_test_layouts[:20]:  # first 20
    tr_mask = train_raw['layout_id'] == lid
    te_mask = test_raw['layout_id'] == lid
    seen_train_preds.append(fw4_oo[tr_mask.values].mean())
    seen_test_preds.append(oracle_new_t[te_mask.values].mean())

seen_train_preds = np.array(seen_train_preds)
seen_test_preds  = np.array(seen_test_preds)
print(f"\n  For 20 seen layouts:")
print(f"    OOF mean (train scenarios): {seen_train_preds.mean():.4f}")
print(f"    Test mean (test scenarios): {seen_test_preds.mean():.4f}")
print(f"    Test - OOF shift: {seen_test_preds.mean()-seen_train_preds.mean():+.4f}")

# Full seen test layouts
all_seen_train = []
all_seen_test  = []
for lid in seen_test_layouts:
    tr_mask = train_raw['layout_id'] == lid
    te_mask = test_raw['layout_id'] == lid
    all_seen_train.append(fw4_oo[tr_mask.values].mean())
    all_seen_test.append(oracle_new_t[te_mask.values].mean())
all_seen_train = np.array(all_seen_train)
all_seen_test  = np.array(all_seen_test)
print(f"\n  All 50 seen layouts:")
print(f"    OOF mean (train scenarios): {all_seen_train.mean():.4f}")
print(f"    Test mean (test scenarios): {all_seen_test.mean():.4f}")
print(f"    Test - OOF shift: {all_seen_test.mean()-all_seen_train.mean():+.4f}")
print(f"    r(oof, test) across layouts: {np.corrcoef(all_seen_train, all_seen_test)[0,1]:.4f}")

# Same for unseen
all_unseen_test = []
unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()
for lid in unseen_test_layouts:
    te_mask = test_raw['layout_id'] == lid
    all_unseen_test.append(oracle_new_t[te_mask.values].mean())
all_unseen_test = np.array(all_unseen_test)
print(f"\n  All 50 unseen layouts test mean: {all_unseen_test.mean():.4f}")

# ============================================================
# Key: if test scenarios for seen layouts give higher predictions
# than training scenarios for same layouts, it means oracle_NEW
# is pushed higher by more demanding test scenarios.
# The test-train shift for seen (+X) may indicate unseen test y_true
# is also X higher than what oracle_NEW predicts.
# ============================================================
seen_train_y = []
for lid in seen_test_layouts:
    tr_mask = train_raw['layout_id'] == lid
    seen_train_y.append(y_true[tr_mask.values].mean())
seen_train_y = np.array(seen_train_y)
print(f"\n  y_true mean for seen test layouts (training data): {seen_train_y.mean():.4f}")
print(f"  oracle_NEW OOF mean for seen test layouts: {all_seen_train.mean():.4f}")
print(f"  oracle_NEW TEST mean for seen test layouts: {all_seen_test.mean():.4f}")
print(f"\n  Training underprediction (y - oof): {seen_train_y.mean() - all_seen_train.mean():+.4f}")
print(f"  Test scenarios harder than training: {all_seen_test.mean() - all_seen_train.mean():+.4f}")
print(f"  Implied test underprediction = training_bias + scenario_shift:")
print(f"    = {seen_train_y.mean() - all_seen_train.mean():+.4f} + {all_seen_test.mean() - all_seen_train.mean():+.4f}")
print(f"    = {seen_train_y.mean() - all_seen_train.mean() + all_seen_test.mean() - all_seen_train.mean():+.4f}")
print(f"    (seen test y_true ≈ {seen_train_y.mean() - all_seen_train.mean() + all_seen_test.mean():.4f}?)")

print("\nDone.")
