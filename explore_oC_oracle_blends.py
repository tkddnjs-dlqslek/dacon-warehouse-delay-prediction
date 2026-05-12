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

c_base_df = pd.read_csv('FINAL_NEW_oracle_C_as_base_OOF_oracle.csv')
c_base_df = c_base_df.set_index('ID').reindex(id_order).reset_index()
c_base_t  = c_base_df['avg_delay_minutes_next_30m'].values

oracle5way_df = pd.read_csv('FINAL_NEW_oracle5way_pure_OOF_oracle.csv')
oracle5way_df = oracle5way_df.set_index('ID').reindex(id_order).reset_index()
oracle5way_t  = oracle5way_df['avg_delay_minutes_next_30m'].values

print(f"oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"oracle_C_as_base: seen={c_base_t[seen_mask].mean():.3f}  unseen={c_base_t[unseen_mask].mean():.3f}")
print(f"oracle_5way: seen={oracle5way_t[seen_mask].mean():.3f}  unseen={oracle5way_t[unseen_mask].mean():.3f}")

print(f"\nr(oracle_NEW, oracle_C) = {pearsonr(oracle_new_t, c_base_t)[0]:.4f}")
print(f"r(oracle_NEW, oracle_5way) = {pearsonr(oracle_new_t, oracle5way_t)[0]:.4f}")
print(f"r(oracle_C, oracle_5way) = {pearsonr(c_base_t, oracle5way_t)[0]:.4f}")

# oracle_C_as_base has: higher seen (17.209) + lower unseen (22.439)
# This is an unusual combination. What if we blend oracle_NEW + oracle_C?
# oracle_C * w + oracle_NEW * (1-w):
#   seen: 17.046 + w*(17.209-17.046) = 17.046 + 0.163w
#   unseen: 22.716 + w*(22.439-22.716) = 22.716 - 0.277w

print("\n" + "="*70)
print("oracle_NEW × (1-w) + oracle_C × w blend")
print("="*70)
print(f"{'w':>5}  {'seen':>8}  {'unseen':>8}  {'delta_u':>8}")
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    blend = np.clip((1-w)*oracle_new_t + w*c_base_t, 0, None)
    du = blend[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w={w:.1f}: seen={blend[seen_mask].mean():.3f}  unseen={blend[unseen_mask].mean():.3f}  Δunseen={du:+.3f}")

# This blend actually LOWERS unseen! Different from oracle_NEW + oracle_5way which also lowers unseen
# oracle_C has lower unseen (22.439) but higher seen (17.209)
# So blending oracle_NEW with oracle_C at w=0.2: seen=17.079, unseen=22.660
# That's almost the same as oracle_NEW... too correlated to help

print("\n" + "="*70)
print("Three-way blend: oracle_NEW + oracle_C + oracle_5way")
print("For unseen: want HIGHER (so exclude oracle_C and oracle_5way for unseen)")
print("Ideal: oracle_NEW seen + oracle_NEW unseen + correction")
print("="*70)

# Actually oracle_C_as_base (seen=17.209, unseen=22.439) has LOWER unseen than oracle_NEW
# This is not what we want for unseen correction
# oracle_C seems to be a model that shifts SEEN higher and UNSEEN lower

# What about: use oracle_C for seen, oracle_NEW for unseen?
seg_C_seen_N_unseen = oracle_new_t.copy()
seg_C_seen_N_unseen[seen_mask] = c_base_t[seen_mask]
seg_C_seen_N_unseen = np.clip(seg_C_seen_N_unseen, 0, None)
print(f"\nseg_C_seen_N_unseen: seen={seg_C_seen_N_unseen[seen_mask].mean():.3f}  unseen={seg_C_seen_N_unseen[unseen_mask].mean():.3f}")

# What about: use oracle_NEW for seen, oracle_C for unseen?
seg_N_seen_C_unseen = oracle_new_t.copy()
seg_N_seen_C_unseen[unseen_mask] = c_base_t[unseen_mask]
seg_N_seen_C_unseen = np.clip(seg_N_seen_C_unseen, 0, None)
print(f"seg_N_seen_C_unseen: seen={seg_N_seen_C_unseen[seen_mask].mean():.3f}  unseen={seg_N_seen_C_unseen[unseen_mask].mean():.3f}")

# These go in wrong direction (lower unseen) — skip them

print("\n" + "="*70)
print("oracle_NEW + various delta variants — final comparison table")
print("="*70)
import glob
new_cands = {}
for fname in sorted(glob.glob('FINAL_NEW_oN_*.csv')):
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        new_cands[fname] = p
    except: pass

new_cands['[oracle_NEW]'] = oracle_new_t
items = sorted(new_cands.items(), key=lambda x: x[1][unseen_mask].mean())
print(f"\n{'Filename':55s}  {'seen':>8}  {'unseen':>8}  {'Δunseen':>9}")
for name, p in items:
    du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {name.replace('FINAL_NEW_',''):53s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}")

print("\n" + "="*70)
print("FINAL SUBMISSION PRIORITY (oracle_NEW calibration path only)")
print("="*70)
priority = [
    ('FINAL_NEW_oN_udelta1_OOF8.3825.csv', 'Tier1: +1 flat — most conservative'),
    ('FINAL_NEW_oN_bucketCorr_qtr_OOF8.3825.csv', 'Tier1: +1.16 bucket 25% — data-driven'),
    ('FINAL_NEW_oracle5way_pure_OOF_oracle.csv', 'Tier3: oracle_5way — go lower (OOF=8.409)'),
    ('FINAL_NEW_oN_udelta2_OOF8.3825.csv', 'Tier2: +2 flat'),
    ('FINAL_NEW_oN_bucketCorr_half_OOF8.3825.csv', 'Tier2: +2.32 bucket 50%'),
    ('FINAL_NEW_oN_layIso20_OOF8.3825.csv', 'Tier2: +1.49 layout-isotonic 20%'),
    ('FINAL_NEW_oN_udelta3_OOF8.3825.csv', 'Tier4: +3 flat'),
    ('FINAL_NEW_oN_linCorrFull_OOF8.3825.csv', 'Tier4: +5.83 full linear correction'),
]
for fname, desc in priority:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname.replace('FINAL_NEW_',''):45s}  seen={p[seen_mask].mean():.3f}  unseen={p[unseen_mask].mean():.3f}  Δ={du:+.3f}  [{desc}]")
    except Exception as e:
        print(f"  {fname}: ERROR {e}")

print("\nDone.")
