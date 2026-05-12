import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
train_layouts = set(pd.read_csv('train.csv')['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
id_order = test_raw['ID'].values

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

def load(fname):
    df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
    return df['avg_delay_minutes_next_30m'].values

def stats(arr, label):
    du = arr[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {label:45s}: D={du:+.4f}  seen={arr[seen_mask].mean():.3f}  unseen={arr[unseen_mask].mean():.3f}")
    return du

def save_blend(pred, label, fname):
    du = pred[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std = pred[unseen_mask].std()
    print(f"  {label:45s}: D={du:+.4f}  seen={pred[seen_mask].mean():.3f}  unseen={pred[unseen_mask].mean():.3f}  std(unseen)={std:.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = pred
    sub.to_csv(fname, index=False)
    return pred

print("="*70)
print("Ensemble: iso_lvL75 × phys5feat × physOffset × flat")
print("="*70)

# Load all key files
iso_lvL75  = load('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv')
phys5_a100 = load('FINAL_NEW_oN_phys5feat5p5_a100_OOF8.3825.csv')
physOff    = load('FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv')
phys3_off  = load('FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv')  # same as physOff (3-feat)
phys5bl3   = load('FINAL_NEW_oN_phys5blend3_OOF8.3825.csv')
phys5cap2  = load('FINAL_NEW_oN_phys5capped2p5_OOF8.3825.csv')

# Flat +5.5 reference
flat_5p5 = oracle_new_t.copy()
flat_5p5[unseen_mask] += 5.5

print(f"\n  Individual deltas (for reference):")
stats(iso_lvL75,  'iso_lvL75')
stats(phys5_a100, 'phys5feat_a100')
stats(physOff,    'physOffset5p5')
stats(phys5bl3,   'phys5blend3(30%phys+70%flat)')
stats(phys5cap2,  'phys5capped2p5')
stats(flat_5p5,   'flat+5.5')

print(f"\n  --- Iso × Phys3 ensembles ---")
for w_iso in [0.3, 0.4, 0.5, 0.6, 0.7]:
    w_phys = 1 - w_iso
    ens = w_iso * iso_lvL75 + w_phys * physOff
    save_blend(ens, f'iso_lvL75({w_iso:.1f}) + physOff5p5({w_phys:.1f})',
               f'FINAL_NEW_oN_isoPhys3_w{int(w_iso*10)}_OOF8.3825.csv')

print(f"\n  --- Iso × Phys5 ensembles ---")
for w_iso in [0.3, 0.5, 0.7]:
    w_phys = 1 - w_iso
    ens = w_iso * iso_lvL75 + w_phys * phys5_a100
    save_blend(ens, f'iso_lvL75({w_iso:.1f}) + phys5feat({w_phys:.1f})',
               f'FINAL_NEW_oN_isoPhys5_w{int(w_iso*10)}_OOF8.3825.csv')

print(f"\n  --- 3-way: iso + phys + flat ---")
for (wi, wp, wf) in [(0.33,0.33,0.34), (0.40,0.40,0.20), (0.50,0.30,0.20),
                      (0.30,0.50,0.20), (0.25,0.25,0.50)]:
    ens = wi * iso_lvL75 + wp * physOff + wf * flat_5p5
    save_blend(ens, f'iso({wi:.2f})+phys3({wp:.2f})+flat({wf:.2f})',
               f'FINAL_NEW_oN_isoPhys3Flat_{int(wi*100):02d}_{int(wp*100):02d}_OOF8.3825.csv')

print(f"\n  --- Correlation between iso and phys corrections (unseen only) ---")
iso_corr = iso_lvL75[unseen_mask] - oracle_new_t[unseen_mask]
phys3_corr = physOff[unseen_mask] - oracle_new_t[unseen_mask]
phys5_corr = phys5_a100[unseen_mask] - oracle_new_t[unseen_mask]
from scipy.stats import pearsonr
r_ip3, _ = pearsonr(iso_corr, phys3_corr)
r_ip5, _ = pearsonr(iso_corr, phys5_corr)
r_p3p5, _ = pearsonr(phys3_corr, phys5_corr)
print(f"  r(iso_corr, phys3_corr) = {r_ip3:.4f}")
print(f"  r(iso_corr, phys5_corr) = {r_ip5:.4f}")
print(f"  r(phys3_corr, phys5_corr) = {r_p3p5:.4f}")
print(f"  iso_corr: mean={iso_corr.mean():.3f}  std={iso_corr.std():.3f}  "
      f"min={iso_corr.min():.3f}  max={iso_corr.max():.3f}")
print(f"  phys3_corr: mean={phys3_corr.mean():.3f}  std={phys3_corr.std():.3f}  "
      f"min={phys3_corr.min():.3f}  max={phys3_corr.max():.3f}")
print(f"  phys5_corr: mean={phys5_corr.mean():.3f}  std={phys5_corr.std():.3f}  "
      f"min={phys5_corr.min():.3f}  max={phys5_corr.max():.3f}")

print(f"\n  --- LOOO-style layout holdout: which correction is most stable? ---")
# Group corrections by layout, check layout-level std
test_raw_copy = test_raw[unseen_mask].copy()
test_raw_copy['iso_corr'] = iso_corr
test_raw_copy['phys3_corr'] = phys3_corr
test_raw_copy['phys5_corr'] = phys5_corr

# Average corrections per layout
lv_stats = test_raw_copy.groupby('layout_id').agg(
    iso_mean=('iso_corr','mean'), phys3_mean=('phys3_corr','mean'),
    phys5_mean=('phys5_corr','mean'), n=('iso_corr','count')
).reset_index()

# These are all scaled to 5.5 mean, so compare layout-level variance
print(f"  Layout-level correction std (across 50 unseen layouts):")
print(f"  iso:   std={lv_stats['iso_mean'].std():.4f}  min={lv_stats['iso_mean'].min():.3f}  max={lv_stats['iso_mean'].max():.3f}")
print(f"  phys3: std={lv_stats['phys3_mean'].std():.4f}  min={lv_stats['phys3_mean'].min():.3f}  max={lv_stats['phys3_mean'].max():.3f}")
print(f"  phys5: std={lv_stats['phys5_mean'].std():.4f}  min={lv_stats['phys5_mean'].min():.3f}  max={lv_stats['phys5_mean'].max():.3f}")

# Cross-correction correlation at layout level
r_lv_ip3, _ = pearsonr(lv_stats['iso_mean'], lv_stats['phys3_mean'])
r_lv_ip5, _ = pearsonr(lv_stats['iso_mean'], lv_stats['phys5_mean'])
r_lv_p3p5, _ = pearsonr(lv_stats['phys3_mean'], lv_stats['phys5_mean'])
print(f"  Layout-level r(iso, phys3) = {r_lv_ip3:.4f}")
print(f"  Layout-level r(iso, phys5) = {r_lv_ip5:.4f}")
print(f"  Layout-level r(phys3, phys5) = {r_lv_p3p5:.4f}")

# Top disagreement layouts (iso vs phys differ most)
lv_stats['diff_ip3'] = abs(lv_stats['iso_mean'] - lv_stats['phys3_mean'])
top_disagree = lv_stats.nlargest(10, 'diff_ip3')[['layout_id','iso_mean','phys3_mean','phys5_mean','n']]
print(f"\n  Top 10 disagreement layouts (|iso - phys3|):")
print(top_disagree.to_string(index=False))

print("\nDone.")
