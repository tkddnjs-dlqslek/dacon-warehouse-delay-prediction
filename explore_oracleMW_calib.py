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

# Load oracle_NEW reference
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values
print(f"oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

sub_tmpl = pd.read_csv('sample_submission.csv')

print("\n" + "="*70)
print("Flat upward calibration of oracle_NEW (unseen only)")
print("="*70)

for delta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = ct[unseen_mask] + delta
    ct = np.clip(ct, 0, None)
    print(f"  oracle_NEW + unseen_delta={delta:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Flat upward calibration of oracle_NEW (seen only)")
print("="*70)

for delta in [0.5, 1.0, 1.5, 2.0]:
    ct = oracle_new_t.copy()
    ct[seen_mask] = ct[seen_mask] + delta
    ct = np.clip(ct, 0, None)
    print(f"  oracle_NEW + seen_delta={delta:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Multiplicative calibration of oracle_NEW (unseen only)")
print("="*70)

for mult in [1.02, 1.05, 1.08, 1.10, 1.15, 1.20]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = ct[unseen_mask] * mult
    ct = np.clip(ct, 0, None)
    print(f"  oracle_NEW * unseen_mult={mult:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Symmetric calibration of oracle_NEW (both seen + unseen)")
print("="*70)

for delta in [0.5, 1.0, 1.5, 2.0]:
    ct = np.clip(oracle_new_t + delta, 0, None)
    print(f"  oracle_NEW + all_delta={delta:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Prediction-level-based calibration (boost more for high predictions)")
print("="*70)

# Rationale: training layout residuals show high-delay layouts are underpredicted MORE
# Apply correction proportional to prediction level for unseen
for threshold, alpha in [(20, 0.05), (20, 0.10), (25, 0.10), (25, 0.15), (15, 0.05)]:
    ct = oracle_new_t.copy()
    mask_high = (ct > threshold) & unseen_mask
    ct[mask_high] = ct[mask_high] * (1 + alpha)
    ct = np.clip(ct, 0, None)
    print(f"  pred>{threshold} * (1+{alpha}): seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  n_rows_boosted={mask_high.sum()}")

print()
print("="*70)
print("Key question: what are the training residuals by prediction level?")
print("="*70)

# Load oracle5_oof to check training residuals by level
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]

xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
oracle5_o = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)

# For each prediction bucket, what is the mean residual?
p = oracle5_o  # prediction
y = y_true
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
print(f"\n  {'pred_bucket':20s} {'n':>7} {'y_mean':>8} {'p_mean':>8} {'resid':>8} {'MAE':>8}")
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (p >= lo) & (p < hi)
    if mask.sum() > 0:
        ym = y[mask].mean()
        pm = p[mask].mean()
        mae = np.mean(np.abs(p[mask] - y[mask]))
        resid = pm - ym  # positive = overpredicting
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y_mean={ym:8.3f}  p_mean={pm:8.3f}  resid={resid:+8.3f}  MAE={mae:8.4f}")

# Now check oracle5_t prediction distribution for test
xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
oracle5_t = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)

print(f"\noracle5_t distribution for test:")
print(f"  All:    mean={oracle5_t.mean():.3f}  std={oracle5_t.std():.3f}")
print(f"  Seen:   mean={oracle5_t[seen_mask].mean():.3f}  std={oracle5_t[seen_mask].std():.3f}")
print(f"  Unseen: mean={oracle5_t[unseen_mask].mean():.3f}  std={oracle5_t[unseen_mask].std():.3f}")

print(f"\noracle_NEW distribution for test:")
print(f"  All:    mean={oracle_new_t.mean():.3f}  std={oracle_new_t.std():.3f}")
print(f"  Seen:   mean={oracle_new_t[seen_mask].mean():.3f}  std={oracle_new_t[seen_mask].std():.3f}")
print(f"  Unseen: mean={oracle_new_t[unseen_mask].mean():.3f}  std={oracle_new_t[unseen_mask].std():.3f}")

# Upward correction per prediction bucket (for oracle_NEW test, unseen)
print(f"\nPrediction bucket distribution for oracle_NEW unseen test rows:")
p_unseen = oracle_new_t[unseen_mask]
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (p_unseen >= lo) & (p_unseen < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  mean={p_unseen[mask].mean():.3f}")

print()
print("="*70)
print("Save promising oracle_NEW calibration candidates")
print("="*70)

candidates_to_save = {}

# delta +1.0 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] += 1.0; ct = np.clip(ct, 0, None)
candidates_to_save['oN_udelta1'] = ct

# delta +2.0 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] += 2.0; ct = np.clip(ct, 0, None)
candidates_to_save['oN_udelta2'] = ct

# delta +3.0 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] += 3.0; ct = np.clip(ct, 0, None)
candidates_to_save['oN_udelta3'] = ct

# delta +5.0 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] += 5.0; ct = np.clip(ct, 0, None)
candidates_to_save['oN_udelta5'] = ct

# mult 1.05 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] *= 1.05; ct = np.clip(ct, 0, None)
candidates_to_save['oN_umult105'] = ct

# mult 1.10 unseen
ct = oracle_new_t.copy(); ct[unseen_mask] *= 1.10; ct = np.clip(ct, 0, None)
candidates_to_save['oN_umult110'] = ct

# delta +1.0 all
ct = np.clip(oracle_new_t + 1.0, 0, None)
candidates_to_save['oN_alldelta1'] = ct

for label, ct in candidates_to_save.items():
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print("\nDone.")
