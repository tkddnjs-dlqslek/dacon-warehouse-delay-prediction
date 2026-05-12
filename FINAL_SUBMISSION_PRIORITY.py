"""
FINAL SUBMISSION PRIORITY GUIDE
================================
Analysis date: 2026-05-02
Best LB: 9.7527 (oracle_NEW, submission_oracle_NEW_OOF8.3825.csv)

KEY FINDINGS:
1. oracle_NEW is optimal for per-row MAE even when weighted by unseen test distribution
2. All calibrations increase variance > decrease bias on training data
3. BUT: test unseen is OOD (inflow +58.8% higher), bias might dominate in test
4. Training underprediction evidence:
   - Overall: -3.175 (all layouts)
   - High-inflow proxy (only 2 layouts > inflow 120): -8.425
   - Inflow regression: Δ≈+7.66 for unseen test (inflow=161)
5. Conservative calibrations cost <0.05 LB if wrong, but could gain >0.5 if right

SUBMISSION STRATEGY:
- All FINAL_NEW_* candidates maintain seen=17.046 unchanged (unseen-only correction)
- OOF unchanged at 8.37624 for all unseen-only candidates

TIER 1 — CONSERVATIVE (most likely to help, least risk)
  - oracle_NEW (DONE, LB=9.7527)
  - oN_specAvg_u05: Δ=+0.162, spec_avg blend (r=0.9249)
  - oN_w30mae_u05: Δ=+0.444, w30mae near-zero [0-15) residual
  - oN_udelta1: Δ=+1.000, flat +1 for all unseen

TIER 2 — MODERATE (training bias match territory)
  - oN_w30mae_u10: Δ=+0.888, near-zero [0-25) training residual
  - oN_hybh10sp_52: Δ=+0.927, h10<40 + spec≥40 hybrid
  - oN_w30mae_u36: Δ=+3.198, matches overall training bias (-3.175)
  - oN_bucketCorr_qtr: Δ=+1.158, data-driven per-bucket 25%

TIER 3 — AGGRESSIVE (matching inflow analysis estimates)
  - oN_udelta5: Δ=+5.000, moderate inflow correction
  - oN_w30mae_u50: Δ=+4.442, 50/50 blend, near-zero [20-40) training residual
  - oN_umult130: Δ=+6.815, multiplicative × 1.30

TIER 4 — VERY AGGRESSIVE (high-inflow proxy estimate)
  - oN_udelta7: Δ=+7.000, proxy analysis estimate
  - oN_w30mae_u80: Δ=+7.108, matching proxy
  - oN_udelta8: Δ=+8.000, near high-inflow proxy residual

THEORETICAL REASONING:
  - oracle_NEW underpredicts unseen layouts because they are OOD (higher inflow)
  - Training correction magnitude: -3.175 → +3.175 needed
  - Inflow proxy shows -8.4 for high-inflow seen layouts → +8 for unseen
  - Per-row MAE analysis shows calibration HURTS on training data (variance effect)
  - LB outcome depends on which effect dominates: variance (→ no calibration)
    or bias (→ calibration helps, higher tier wins)
"""

# Quick summary print
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")
import glob, pandas as pd, numpy as np

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
train_layouts = set(pd.read_csv('train.csv')['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
id_order = test_raw['ID'].values

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

priority_files = [
    # Tier 0
    ('submission_oracle_NEW_OOF8.3825.csv', 'BASELINE [LB=9.7527]'),
    # Tier 1
    ('FINAL_NEW_oN_specAvg_u05_OOF8.3825.csv', 'T1: spec_avg 5%'),
    ('FINAL_NEW_oN_w30mae_u05_OOF8.3825.csv', 'T1: w30mae 5%'),
    ('FINAL_NEW_oN_udelta1_OOF8.3825.csv', 'T1: flat +1'),
    # Tier 2
    ('FINAL_NEW_oN_w30mae_u10_OOF8.3825.csv', 'T2: w30mae 10%'),
    ('FINAL_NEW_oN_hybh10sp_52_OOF8.3825.csv', 'T2: hyb h10<40+spec≥40'),
    ('FINAL_NEW_oN_w30mae_u36_OOF8.3825.csv', 'T2: w30mae 36% (bias match)'),
    ('FINAL_NEW_oN_bucketCorr_qtr_OOF8.3825.csv', 'T2: bucket 25%'),
    # Tier 3
    ('FINAL_NEW_oN_w30mae_u50_OOF8.3825.csv', 'T3: w30mae 50%'),
    ('FINAL_NEW_oN_udelta5_OOF8.3825.csv', 'T3: flat +5'),
    ('FINAL_NEW_oN_umult130_OOF8.3825.csv', 'T3: mult ×1.30'),
    # Tier 4
    ('FINAL_NEW_oN_udelta7_OOF8.3825.csv', 'T4: flat +7 (proxy)'),
    ('FINAL_NEW_oN_w30mae_u80_OOF8.3825.csv', 'T4: w30mae 80%'),
]

print("FINAL SUBMISSION PRIORITY")
print("=" * 75)
print(f"{'Filename':50s}  {'seen':>8}  {'unseen':>8}  {'Δunseen':>9}  {'tier'}")
print("-" * 75)
for fname, label in priority_files:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname[:48]:48s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}  [{label}]")
    except Exception as e:
        print(f"  {fname[:48]:48s}  MISSING: {e}")

print()
print(f"Total FINAL_NEW_* candidates: {len(glob.glob('FINAL_NEW_*.csv'))}")
