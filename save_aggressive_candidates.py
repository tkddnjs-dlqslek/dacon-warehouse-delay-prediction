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
te_ls_pos_df = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in te_ls_pos_df.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

w30mae_t = np.clip(np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2], 0, None)
spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)

sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Aggressive unseen correction candidates (Δ = +4 to +10)")
print("High-inflow proxy suggests optimal Δ might be +4-8")
print("="*70)
print(f"\n{'label':40s}  {'seen':>8}  {'unseen':>8}  {'Δunseen':>9}")

candidates = []

# w30mae at high weights
for w in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.90, 1.00]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_w30mae_u{int(w*100):02d}"
    print(f"  {label:40s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")
    candidates.append((label, ct, du))

# Flat deltas at larger values
for delta in [4, 5, 6, 7, 8]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = ct[unseen_mask] + delta
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_udelta{delta}"
    print(f"  {label:40s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")
    candidates.append((label, ct, du))

# spec_avg at high weights
for w in [0.40, 0.50, 0.70, 1.00]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*spec_avg_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"oN_specAvg_u{int(w*100):02d}"
    print(f"  {label:40s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")
    candidates.append((label, ct, du))

# Combined w30mae + spec_avg at high weights
for w1, w2 in [(0.20,0.30), (0.30,0.30), (0.40,0.20), (0.30,0.40)]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w1-w2)*oracle_new_t[unseen_mask] + w1*spec_avg_t[unseen_mask] + w2*w30mae_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"3way_sp{int(w1*100)}_wm{int(w2*100)}"
    print(f"  {label:40s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")
    candidates.append((label, ct, du))

# Save all
print(f"\nSaving {len(candidates)} candidates...")
for label, ct, du in candidates:
    fname = f"FINAL_NEW_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
print("Done saving.")

# Summary sorted by Δ
candidates.sort(key=lambda x: x[2])
print(f"\n{'label':40s}  {'seen':>8}  {'unseen':>8}  {'Δ':>9}")
for label, ct, du in candidates:
    print(f"  {label:40s}  {ct[seen_mask].mean():.3f}  {ct[unseen_mask].mean():.3f}  {du:+.3f}")

print("\nDone.")
