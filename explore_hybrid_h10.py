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

h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)

sub_tmpl = pd.read_csv('sample_submission.csv')

# Unseen rows split by oracle_NEW prediction level
p_unseen = oracle_new_t[unseen_mask]
below40_mask = unseen_mask & (oracle_new_t < 40)
above40_mask = unseen_mask & (oracle_new_t >= 40)
print(f"Unseen rows: total={unseen_mask.sum()} below40={below40_mask.sum()} above40={above40_mask.sum()}")
print(f"oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"h10 below40 diff vs oN: {(h10_t[below40_mask]-oracle_new_t[below40_mask]).mean():+.3f}")
print(f"h10 above40 diff vs oN: {(h10_t[above40_mask]-oracle_new_t[above40_mask]).mean():+.3f}")

# ============================================================
# Strategy A: Hybrid — h10 for unseen<40, oracle_NEW for 40+
# ============================================================
print("\n" + "="*70)
print("Strategy A: Hybrid h10 — apply h10 ONLY for unseen rows where oN < 40")
print("="*70)
for w_u in [0.10, 0.20, 0.30, 0.50, 0.70, 1.0]:
    ct = oracle_new_t.copy()
    ct[below40_mask] = (1-w_u)*oracle_new_t[below40_mask] + w_u*h10_t[below40_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    ds = ct[seen_mask].mean() - oracle_new_t[seen_mask].mean()
    print(f"  w_u={w_u:.2f}: seen={ct[seen_mask].mean():.3f}({ds:+.3f})  unseen={ct[unseen_mask].mean():.3f}  Δunseen={du:+.3f}")

# Save hybrids
print("\nSaving hybrid h10 candidates:")
for w_u in [0.20, 0.50, 1.0]:
    ct = oracle_new_t.copy()
    ct[below40_mask] = (1-w_u)*oracle_new_t[below40_mask] + w_u*h10_t[below40_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_hybh10u{int(w_u*100)}"
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# Strategy B: Hybrid + flat delta for above40
# ============================================================
print("\n" + "="*70)
print("Strategy B: Hybrid h10 (below40) + flat delta +1 for all unseen")
print("="*70)
for w_u in [0.30, 0.50, 1.0]:
    for delta in [0.5, 1.0, 1.5]:
        ct = oracle_new_t.copy()
        ct[below40_mask] = (1-w_u)*oracle_new_t[below40_mask] + w_u*h10_t[below40_mask]
        ct[unseen_mask] = ct[unseen_mask] + delta
        ct = np.clip(ct, 0, None)
        du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  w_u={w_u:.2f} Δ={delta}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δunseen={du:+.3f}")

# ============================================================
# Strategy C: h10 threshold comparison
# ============================================================
print("\n" + "="*70)
print("Strategy C: h10 below-threshold sweep (different cut points)")
print("="*70)
for threshold in [30, 35, 40, 50]:
    below_mask = unseen_mask & (oracle_new_t < threshold)
    for w_u in [0.50, 1.0]:
        ct = oracle_new_t.copy()
        ct[below_mask] = (1-w_u)*oracle_new_t[below_mask] + w_u*h10_t[below_mask]
        ct = np.clip(ct, 0, None)
        du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        ds = ct[seen_mask].mean() - oracle_new_t[seen_mask].mean()
        print(f"  thr<{threshold} w={w_u:.1f}: n_affected={below_mask.sum():6d}  seen={ct[seen_mask].mean():.3f}({ds:+.3f})  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# ============================================================
# All candidates summary
# ============================================================
print("\n" + "="*70)
print("FULL CANDIDATE TABLE (all FINAL_NEW_*.csv sorted by unseen)")
print("="*70)
import glob
cands = {}
for fname in sorted(glob.glob('FINAL_NEW_*.csv')):
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        cands[fname] = p
    except: pass
cands['[oracle_NEW]'] = oracle_new_t
items = sorted(cands.items(), key=lambda x: x[1][unseen_mask].mean())
print(f"\n{'Filename':60s}  {'seen':>8}  {'unseen':>8}  {'Δunseen':>9}")
for name, p in items:
    du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {name.replace('FINAL_NEW_',''):58s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}")

print("\nDone.")
