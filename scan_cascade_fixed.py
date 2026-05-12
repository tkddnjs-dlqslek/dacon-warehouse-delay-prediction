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

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

sub_tmpl = pd.read_csv('sample_submission.csv')

# All cascade model files — ALL need id2/te_id2 reindex
cascade_models = [
    'spec_cb_w30', 'spec_lgb_raw_huber', 'spec_lgb_raw_mae',
    'spec_lgb_w30_mae', 'spec_lgb_w30_huber', 'spec_avg',
    'spec_cb_raw',
]

print("="*75)
print("CASCADE models (CORRECT reindex via id2/te_id2)")
print("="*75)
print(f"\n  {'model':30s} {'OOF':>9} {'seen':>8} {'unseen':>8} {'r(oN)':>7} {'Δunseen':>8}")

good_cascade = []
for name in cascade_models:
    try:
        oof_raw = np.load(f'results/cascade/{name}_oof.npy')
        te_raw  = np.load(f'results/cascade/{name}_test.npy')
        oof = np.clip(oof_raw[id2], 0, None)
        te  = np.clip(te_raw[te_id2], 0, None)
        oof_mae = mae_fn(oof)
        r_oN, _ = pearsonr(te, oracle_new_t)
        du = te[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        marker = " ← HIGH" if du > 0 else ""
        print(f"  {name:30s}: OOF={oof_mae:.5f}  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r={r_oN:.4f}  Δ={du:+.3f}{marker}")
        good_cascade.append((name, oof, te, oof_mae, r_oN, du))
    except Exception as e:
        print(f"  {name:30s}: ERR {str(e)[:60]}")

# Sort by unseen descending
good_cascade.sort(key=lambda x: -x[2][unseen_mask].mean())

print("\n" + "="*75)
print("CASCADE models — per-bucket analysis for TOP-2 unseen models")
print("="*75)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
for name, oof, te, oof_mae, r_oN, du in good_cascade[:3]:
    print(f"\n{name}: OOF={oof_mae:.5f}  unseen={te[unseen_mask].mean():.3f}  Δ={du:+.3f}")
    p_oN_u = oracle_new_t[unseen_mask]
    p_te_u = te[unseen_mask]
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (p_oN_u >= lo) & (p_oN_u < hi)
        if mask.sum() > 0:
            diff = (p_te_u[mask]-p_oN_u[mask]).mean()
            print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  oN={p_oN_u[mask].mean():.3f}  {name}={p_te_u[mask].mean():.3f}  diff={diff:+.3f}")

# ============================================================
# Cross-correlation between cascade models
# ============================================================
print("\n" + "="*75)
print("Cross-correlations between cascade models (test, unseen rows)")
print("="*75)
names = [x[0] for x in good_cascade]
tes   = [x[2] for x in good_cascade]
print("  Unseen rows:")
print(f"  {'':25s}", end='')
for n in names[:5]:
    print(f"  {n[:12]:12s}", end='')
print()
for i, (n1, te1) in enumerate(zip(names[:5], tes[:5])):
    print(f"  {n1:25s}", end='')
    for te2 in tes[:5]:
        r, _ = pearsonr(te1[unseen_mask], te2[unseen_mask])
        print(f"  {r:12.4f}", end='')
    print()

# ============================================================
# Unseen-only blend: oracle_NEW + each cascade model
# ============================================================
print("\n" + "="*75)
print("Unseen-only blend with each cascade model")
print("="*75)
for name, oof, te, oof_mae, r_oN, du in good_cascade:
    if te[unseen_mask].mean() < 21.0:
        continue
    print(f"\n  {name}: unseen={te[unseen_mask].mean():.3f}  Δ={du:+.3f}")
    for w in [0.05, 0.10, 0.20, 0.30]:
        ct = oracle_new_t.copy()
        ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*te[unseen_mask]
        ct = np.clip(ct, 0, None)
        duu = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"    w={w}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={duu:+.3f}")

# ============================================================
# Save best new candidates from cascade
# ============================================================
print("\n" + "="*75)
print("Saving best cascade blend candidates")
print("="*75)
for name, oof, te, oof_mae, r_oN, du in good_cascade:
    if te[unseen_mask].mean() < oracle_new_t[unseen_mask].mean():
        continue
    for w in [0.10, 0.20]:
        ct = oracle_new_t.copy()
        ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*te[unseen_mask]
        ct = np.clip(ct, 0, None)
        duu = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        label_name = name.replace('_','').replace('spec','sp').replace('avg','avg')
        fname = f"FINAL_NEW_oN_{label_name}_u{int(w*100)}_OOF8.3825.csv"
        sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
        sub.to_csv(fname, index=False)
        print(f"  {fname}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={duu:+.3f}")

print("\nDone.")
