import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, glob
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

print(f"oracle_NEW reference: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"Test composition: seen={seen_mask.sum()} (60%)  unseen={unseen_mask.sum()} (40%)")

print("\n" + "="*80)
print("COMPLETE SUBMISSION CANDIDATE LIST (New candidates from today's analysis)")
print("="*80)

# Load all new FINAL_NEW candidates from today
new_files = sorted(glob.glob('FINAL_NEW_*.csv'))
all_cands = {}
for fname in new_files:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        all_cands[fname] = p
    except: pass

# Add oracle_NEW
all_cands['[oracle_NEW] LB=9.7527'] = oracle_new_t

# Sort by unseen mean
items = sorted(all_cands.items(), key=lambda x: x[1][unseen_mask].mean())
print(f"\n{'Filename':65s}  {'seen':>8}  {'unseen':>8}")
print("-"*85)
for name, p in items:
    marker = " ◄ BEST" if '[oracle_NEW]' in name else ""
    print(f"  {name[:63]:63s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}{marker}")

print("\n" + "="*80)
print("PRIORITY SUBMISSION LIST — Final Recommendation")
print("="*80)
print("""
KEY FINDINGS FROM TODAY'S ANALYSIS:
  1. oracle_NEW OOF calibration by bucket: mean bias = -3.175 min (underpredicting!)
     - [30-40) bucket: residual = -4.706 (41% of unseen test rows)
     - [25-30) bucket: residual = -9.003
  2. Layout-level correction: r(p_mean, resid) = -0.505, R² = 0.2555
     → Moderate evidence for upward calibration
  3. ALL unexplored model files explored: iter_pseudo round4 (OOF=8.568, useless),
     lag_target (OOF=9.23, too poor), everything else OOF=18+ (garbage)
  4. Training data bias strongly suggests oracle_NEW underpredicts unseen by 3-6 min
  5. NOTE: Previous failed calibrations were on triple_base (seen=17.640),
     NOT on oracle_NEW (seen=17.046). This is an important untested direction.

PRIORITY ORDER (higher priority = submit first):
  TIER 1 — Data-driven conservative calibration:
  1. oN_udelta1 (seen=17.046, unseen=23.716): Flat +1 min to unseen. Simplest, safest.
  2. oN_bucketCorr_qtr (seen=17.046, unseen=23.875): 25% of per-bucket training bias correction.
  3. oN_linCorr2 (seen=17.046, unseen=23.881): 20% of per-layout linear correction.
     [All three effectively test same hypothesis: +1.2 min unseen correction]

  TIER 2 — Medium correction:
  4. oN_udelta2 (seen=17.046, unseen=24.716): Flat +2 min correction.
  5. oN_bucketCorr_half (seen=17.046, unseen=25.033): 50% bucket correction.

  TIER 3 — Going lower (to test if oracle_NEW already overcalibrated):
  6. oracle5way_pure (seen=16.789, unseen=22.358): Pure oracle blend, OOF=8.409.
  7. seg_oNseen_o5unseen (seen=17.046, unseen=22.358): Lower unseen with oracle_NEW seen.

  TIER 4 — Larger corrections (speculative based on full training bias):
  8. oN_udelta3 (seen=17.046, unseen=25.716): +3 min
  9. oN_linCorrFull (seen=17.046, unseen=28.542): Full layout-level linear correction.

  DO NOT SUBMIT: All triple_base variants (seen=17.640), all calibration variants
  with seen>17.1, all oracle variants BELOW oracle_NEW level.
""")

# Compute pairwise correlations for tier 1 candidates
print("="*80)
print("Correlation analysis: Tier 1 candidates vs oracle_NEW")
print("="*80)
tier1_names = [
    'FINAL_NEW_oN_udelta1_OOF8.3825.csv',
    'FINAL_NEW_oN_bucketCorr_qtr_OOF8.3825.csv',
    'FINAL_NEW_oN_linCorr2_OOF8.3825.csv',
    'FINAL_NEW_oN_udelta2_OOF8.3825.csv',
    'FINAL_NEW_oN_bucketCorr_half_OOF8.3825.csv',
    'FINAL_NEW_oracle5way_pure_OOF_oracle.csv',
    'FINAL_NEW_seg_oNseen_o5unseen_OOF_oracle.csv',
]

for fname in tier1_names:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        r, _ = pearsonr(p, oracle_new_t)
        delta_seen = p[seen_mask].mean() - oracle_new_t[seen_mask].mean()
        delta_unseen = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname.replace('FINAL_NEW_','').replace('_OOF8.3825','').replace('_OOF_oracle',''):45s}"
              f"  Δseen={delta_seen:+.3f}  Δunseen={delta_unseen:+.3f}  r(oN)={r:.4f}")
    except Exception as e:
        print(f"  {fname[:45]}: ERROR {e}")

print()
print("="*80)
print("CONCLUSION: Best unexplored direction")
print("="*80)
print("""
The most promising untested direction is oracle_NEW + small upward unseen calibration:
  - Training analysis: oracle_NEW underpredicts by -3.2 to -4.7 min for high-delay scenarios
  - Previous calibration failures were on triple_base (seen=17.640), NOT oracle_NEW
  - Oracle_NEW's lower seen (17.046) may allow unseen correction to improve LB
  - Recommended first submission: oN_udelta1 (unseen=23.716, Δ=+1.0 from oracle_NEW)

Expected outcome based on evidence:
  - If true unseen mean ≈ 22.7 (oracle_NEW is optimal): +delta variants hurt LB
  - If true unseen mean ≈ 24-26 (moderate underprediction): +1 to +2 variants help
  - If true unseen mean ≈ 27-30 (significant underprediction): +3 to +7 variants help

The most informative first submission: oN_udelta1
  - If LB improves: try oN_udelta2
  - If LB stays same or worsens: try oracle_5way (going lower)
""")

print("\nDone.")
