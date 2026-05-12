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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Multiplicative correction for unseen rows")
print("oracle_NEW_unseen * factor")
print("Training bias: -3.175 → factor ≈ 1 + 3.175/22.716 ≈ 1.140")
print("High-inflow proxy: -8.425 → factor ≈ 1 + 8.425/22.716 ≈ 1.371")
print("="*70)

# Optimal factor if mean(y_true_unseen) / mean(oracle_NEW_unseen) applies
# Training overall bias: pred=19.541, y=22.716+3.175=25.891 → factor=1.327? No, let me think again.
# oracle_NEW pred for unseen = 22.716
# If true mean for unseen ≈ 22.716 + 7.66 = 30.376 (from inflow analysis)
# factor = 30.376 / 22.716 = 1.337

candidates = []

print(f"\n  {'label':30s}  {'seen':>8}  {'unseen':>8}  {'Δ':>8}")
for factor in [1.10, 1.15, 1.20, 1.25, 1.30, 1.35, 1.40, 1.50, 1.60, 1.80, 2.00]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = ct[unseen_mask] * factor
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_umult{int(factor*100)}"
    print(f"  {label:30s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")
    candidates.append((label, ct, du))

print("\n" + "="*70)
print("Per-bucket comparison: multiplicative vs additive for unseen test")
print("(optimal factor per-bucket based on training bias)")
print("="*70)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
# Per-bucket optimal factor from training:
# [30-40): bucket_mean = 34.441 test, training resid=-4.706 → implied factor = (34.441+4.706)/34.441 = 1.137
# [40-50): 43.773 + 15.859 → factor = 1.362
# [50-200): 68.093 + 31.039 → factor = 1.456

print(f"\n  {'bucket':12s} {'n':>7} {'oN_pred':>9} {'train_resid':>12} {'implied_factor':>15} {'mult+opt_pred':>14}")
train_residuals = {
    (0,5): -0.652, (5,10): -1.027, (10,15): -2.927, (15,20): -5.333,
    (20,25): -7.842, (25,30): -9.003, (30,40): -4.706, (40,50): -15.859, (50,200): -31.039
}
p_oN_u = oracle_new_t[unseen_mask]
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN_u >= lo) & (p_oN_u < hi)
    if mask.sum() > 0:
        pred_mean = p_oN_u[mask].mean()
        resid = train_residuals.get((lo, hi), 0)
        implied_y = pred_mean - resid  # if same bias in test
        implied_factor = implied_y / pred_mean if pred_mean > 0 else 1.0
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  oN={pred_mean:9.3f}  resid={resid:12.3f}  factor={implied_factor:15.4f}  opt_pred={implied_y:14.3f}")

print("\n" + "="*70)
print("Bucket-wise multiplicative correction")
print("Use per-bucket optimal factor based on training bias")
print("="*70)
ct_opt_mult = oracle_new_t.copy()
for lo, hi in zip(bins[:-1], bins[1:]):
    # Use 25% of training bias as correction (conservative)
    resid = train_residuals.get((lo, hi), 0)
    mask_u = unseen_mask & (oracle_new_t >= lo) & (oracle_new_t < hi)
    if mask_u.sum() > 0 and abs(resid) > 0:
        bucket_mean = oracle_new_t[mask_u].mean()
        factor_qtr = 1.0 - 0.25 * resid / bucket_mean if bucket_mean > 0 else 1.0
        ct_opt_mult[mask_u] = ct_opt_mult[mask_u] * factor_qtr

ct_opt_mult = np.clip(ct_opt_mult, 0, None)
du = ct_opt_mult[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Bucket-wise 25% mult correction: seen={ct_opt_mult[seen_mask].mean():.3f}  unseen={ct_opt_mult[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# 50% version
ct_opt_mult50 = oracle_new_t.copy()
for lo, hi in zip(bins[:-1], bins[1:]):
    resid = train_residuals.get((lo, hi), 0)
    mask_u = unseen_mask & (oracle_new_t >= lo) & (oracle_new_t < hi)
    if mask_u.sum() > 0 and abs(resid) > 0:
        bucket_mean = oracle_new_t[mask_u].mean()
        factor_half = 1.0 - 0.50 * resid / bucket_mean if bucket_mean > 0 else 1.0
        ct_opt_mult50[mask_u] = ct_opt_mult50[mask_u] * factor_half

ct_opt_mult50 = np.clip(ct_opt_mult50, 0, None)
du50 = ct_opt_mult50[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Bucket-wise 50% mult correction: seen={ct_opt_mult50[seen_mask].mean():.3f}  unseen={ct_opt_mult50[unseen_mask].mean():.3f}  Δ={du50:+.3f}")

# Save key multiplicative candidates
print("\n" + "="*70)
print("Saving multiplicative candidates")
print("="*70)
for factor in [1.20, 1.30, 1.40]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = ct[unseen_mask] * factor
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_umult{int(factor*100)}"
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  {fname}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# Bucket-wise mult
for label, ct_bwm, du_bwm in [('oN_bucketMult25', ct_opt_mult, du), ('oN_bucketMult50', ct_opt_mult50, du50)]:
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_bwm
    sub.to_csv(fname, index=False)
    print(f"  {fname}: seen={ct_bwm[seen_mask].mean():.3f}  unseen={ct_bwm[unseen_mask].mean():.3f}  Δ={du_bwm:+.3f}")

print("\nDone.")
