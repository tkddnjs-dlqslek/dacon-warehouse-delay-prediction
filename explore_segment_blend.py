import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os

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

print(f"seen rows: {seen_mask.sum()}  unseen rows: {unseen_mask.sum()}  total: {len(test_raw)}")
print(f"seen fraction: {seen_mask.mean():.3f}  unseen fraction: {unseen_mask.mean():.3f}")

# Load oracle_NEW and oracle_5way
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

oracle5way_df = pd.read_csv('FINAL_NEW_oracle5way_pure_OOF_oracle.csv')
oracle5way_df = oracle5way_df.set_index('ID').reindex(id_order).reset_index()
oracle5way_t  = oracle5way_df['avg_delay_minutes_next_30m'].values

# rem_only
rem_only_df = pd.read_csv('FINAL_NEW_rem_only_OOF_oracle.csv')
rem_only_df  = rem_only_df.set_index('ID').reindex(id_order).reset_index()
rem_only_t   = rem_only_df['avg_delay_minutes_next_30m'].values

print(f"\noracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"oracle_5way: seen={oracle5way_t[seen_mask].mean():.3f}  unseen={oracle5way_t[unseen_mask].mean():.3f}")
print(f"rem_only: seen={rem_only_t[seen_mask].mean():.3f}  unseen={rem_only_t[unseen_mask].mean():.3f}")

sub_tmpl = pd.read_csv('sample_submission.csv')

print("\n" + "="*70)
print("Segment-aware blends (oracle_NEW seen + oracle_5way unseen, etc.)")
print("="*70)

candidates = {}

# 1. oracle_NEW seen + oracle_5way unseen
seg1 = oracle_new_t.copy()
seg1[unseen_mask] = oracle5way_t[unseen_mask]
candidates['seg_oNseen_o5unseen'] = seg1
print(f"seg_oNseen_o5unseen: seen={seg1[seen_mask].mean():.3f}  unseen={seg1[unseen_mask].mean():.3f}")

# 2. oracle_5way seen + oracle_NEW unseen
seg2 = oracle5way_t.copy()
seg2[unseen_mask] = oracle_new_t[unseen_mask]
candidates['seg_o5seen_oNunseen'] = seg2
print(f"seg_o5seen_oNunseen: seen={seg2[seen_mask].mean():.3f}  unseen={seg2[unseen_mask].mean():.3f}")

# 3. oracle_NEW seen + rem_only unseen
seg3 = oracle_new_t.copy()
seg3[unseen_mask] = rem_only_t[unseen_mask]
candidates['seg_oNseen_remunseen'] = seg3
print(f"seg_oNseen_remunseen: seen={seg3[seen_mask].mean():.3f}  unseen={seg3[unseen_mask].mean():.3f}")

# 4. oracle_5way seen + rem_only unseen
seg4 = oracle5way_t.copy()
seg4[unseen_mask] = rem_only_t[unseen_mask]
candidates['seg_o5seen_remunseen'] = seg4
print(f"seg_o5seen_remunseen: seen={seg4[seen_mask].mean():.3f}  unseen={seg4[unseen_mask].mean():.3f}")

print("\n" + "="*70)
print("Fine-grained grid: oracle_NEW × w + oracle_5way × (1-w)  [seen/unseen separate]")
print("="*70)

# For each blend, also compute CORRELATION with oracle_NEW (for diversity measure)
from scipy.stats import pearsonr

for w_seen in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
    for w_unseen in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        blend = np.zeros(len(test_raw))
        blend[seen_mask] = w_seen * oracle_new_t[seen_mask] + (1-w_seen) * oracle5way_t[seen_mask]
        blend[unseen_mask] = w_unseen * oracle_new_t[unseen_mask] + (1-w_unseen) * oracle5way_t[unseen_mask]
        blend = np.clip(blend, 0, None)
        r, _ = pearsonr(blend, oracle_new_t)
        print(f"  ws={w_seen:.1f} wu={w_unseen:.1f}: seen={blend[seen_mask].mean():.3f}  unseen={blend[unseen_mask].mean():.3f}  r(oracle_NEW)={r:.4f}")

print("\n" + "="*70)
print("Diagonal: w_seen=w_unseen (standard blend)")
print("="*70)
for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    blend = np.clip(w*oracle_new_t + (1-w)*oracle5way_t, 0, None)
    r, _ = pearsonr(blend, oracle_new_t)
    print(f"  w={w:.1f}: seen={blend[seen_mask].mean():.3f}  unseen={blend[unseen_mask].mean():.3f}  r={r:.4f}")

print("\n" + "="*70)
print("Save segment-aware candidates")
print("="*70)

for label, ct in candidates.items():
    fname = f"FINAL_NEW_{label}_OOF_oracle.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# Also test: what if we do a 3-segment blend by layout group?
print("\n" + "="*70)
print("Layout group analysis: n_layouts, inflow distribution")
print("="*70)
test_raw['order_inflow'] = test_raw.get('order_inflow', pd.Series(np.nan, index=test_raw.index))

# Try to get inflow
try:
    # Check if there's an inflow column
    cols = pd.read_csv('test.csv', nrows=1).columns.tolist()
    inflow_col = [c for c in cols if 'inflow' in c.lower() or 'order' in c.lower() and c != 'order_id']
    print(f"Possible inflow cols: {inflow_col[:5]}")
except: pass

unseen_layouts = test_raw[unseen_mask]['layout_id'].unique()
seen_layouts_test = test_raw[seen_mask]['layout_id'].unique()
print(f"Unseen layouts in test: {len(unseen_layouts)}")
print(f"Seen layouts in test: {len(seen_layouts_test)}")

# Per-layout oracle_NEW vs oracle_5way difference
layout_diffs = []
for lay in unseen_layouts:
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    if mask.sum() > 0:
        diff = oracle5way_t[mask].mean() - oracle_new_t[mask].mean()
        layout_diffs.append((lay, mask.sum(), oracle_new_t[mask].mean(), oracle5way_t[mask].mean(), diff))

layout_diffs.sort(key=lambda x: x[4])
print(f"\nPer unseen-layout: oracle_5way - oracle_NEW difference (sorted)")
for lay, n, mn, m5, diff in layout_diffs[:10]:
    print(f"  {lay}: n={n:4d}  oN={mn:.3f}  o5={m5:.3f}  diff={diff:+.3f}")
print("  ...")
for lay, n, mn, m5, diff in layout_diffs[-10:]:
    print(f"  {lay}: n={n:4d}  oN={mn:.3f}  o5={m5:.3f}  diff={diff:+.3f}")

print("\nDone.")
