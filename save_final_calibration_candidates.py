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
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

w30mae_t = np.clip(np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2], 0, None)
spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)

sub_tmpl = pd.read_csv('sample_submission.csv')

candidates = []

print("="*70)
print("w30mae unseen-only blend — full range")
print("Training bias match at w≈0.36 (Δ≈+3.175)")
print("="*70)
for w in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w={w:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")
    if w in [0.10, 0.15, 0.20, 0.30, 0.36, 0.45]:
        candidates.append((f"oN_w30mae_u{int(w*100):02d}", ct, du))

# w=0.36 specifically (training bias match)
w = 0.36
ct = oracle_new_t.copy()
ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
ct = np.clip(ct, 0, None)
du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
candidates.append((f"oN_w30mae_u36", ct, du))
print(f"  w={w:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}  ← training bias match")

print("\n" + "="*70)
print("3-way unseen blend: oracle_NEW + spec_avg + w30mae")
print("="*70)
for w1, w2 in [(0.10,0.10), (0.15,0.15), (0.10,0.20), (0.20,0.10), (0.10,0.30), (0.30,0.10)]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w1-w2)*oracle_new_t[unseen_mask] + w1*spec_avg_t[unseen_mask] + w2*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"3way_sp{int(w1*100)}_wm{int(w2*100)}"
    print(f"  sp={w1:.2f}+wm={w2:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")
    candidates.append((label, ct, du))

print("\n" + "="*70)
print("Hybrid: h10 below40 + w30mae above40 (best of each)")
print("(h10 better for <40, w30mae better for 40+ vs oracle_NEW)")
print("="*70)
below40_mask = unseen_mask & (oracle_new_t < 40)
above40_mask = unseen_mask & (oracle_new_t >= 40)
for w_lo, w_hi in [(0.5, 0.3), (0.5, 0.45), (1.0, 0.3), (1.0, 0.45)]:
    ct = oracle_new_t.copy()
    ct[below40_mask] = (1-w_lo)*oracle_new_t[below40_mask] + w_lo*h10_t[below40_mask]
    ct[above40_mask] = (1-w_hi)*oracle_new_t[above40_mask] + w_hi*w30mae_t[above40_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"hybh10wm_{int(w_lo*10)}{int(w_hi*10)}"
    print(f"  h10<40 w={w_lo:.1f} + w30mae>=40 w={w_hi:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")
    candidates.append((label, ct, du))

print("\n" + "="*70)
print("Summary: saving all candidates")
print("="*70)
saved = []
for label, ct, du in candidates:
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    saved.append((label, ct[seen_mask].mean(), ct[unseen_mask].mean(), du))

# Print sorted by unseen
saved.sort(key=lambda x: x[2])
print(f"\n{'label':40s}  {'seen':>8}  {'unseen':>8}  {'Δunseen':>9}")
for label, seen_m, unseen_m, du in saved:
    print(f"  {label:40s}  {seen_m:8.3f}  {unseen_m:8.3f}  {du:+9.3f}")

print("\nDone.")
