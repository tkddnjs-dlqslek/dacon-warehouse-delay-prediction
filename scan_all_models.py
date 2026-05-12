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

# ============================================================
# Cascade models — all unchecked
# ============================================================
print("="*75)
print("CASCADE models — all")
print("="*75)
cascade_models = [
    ('spec_cb_w30', 'results/cascade/spec_cb_w30_oof.npy', 'results/cascade/spec_cb_w30_test.npy', False),
    ('spec_lgb_raw_huber', 'results/cascade/spec_lgb_raw_huber_oof.npy', 'results/cascade/spec_lgb_raw_huber_test.npy', False),
    ('spec_lgb_raw_mae', 'results/cascade/spec_lgb_raw_mae_oof.npy', 'results/cascade/spec_lgb_raw_mae_test.npy', False),
    ('spec_lgb_w30_mae', 'results/cascade/spec_lgb_w30_mae_oof.npy', 'results/cascade/spec_lgb_w30_mae_test.npy', False),
    ('spec_lgb_w30_huber', 'results/cascade/spec_lgb_w30_huber_oof.npy', 'results/cascade/spec_lgb_w30_huber_test.npy', False),
    ('spec_avg', 'results/cascade/spec_avg_oof.npy', 'results/cascade/spec_avg_test.npy', False),
    ('clf', 'results/cascade/clf_oof.npy', 'results/cascade/clf_test.npy', False),
    ('clf_v2', 'results/cascade/clf_v2_oof.npy', 'results/cascade/clf_v2_test.npy', False),
    ('spec_cb_raw', 'results/cascade/spec_cb_raw_oof.npy', 'results/cascade/spec_cb_raw_test.npy', False),
]

good_cascade = []
for name, oof_path, test_path, use_reindex in cascade_models:
    try:
        oof_raw = np.load(oof_path)
        te_raw  = np.load(test_path)
        # Check if reindex needed
        if len(oof_raw) == len(y_true):
            oof = np.clip(oof_raw, 0, None)
        elif len(oof_raw) >= len(ls_pos):
            oof = np.clip(oof_raw[id2], 0, None)
        else:
            print(f"  {name:30s}: oof_shape mismatch {len(oof_raw)}")
            continue
        if len(te_raw) == len(id_order):
            te = np.clip(te_raw, 0, None)
        elif len(te_raw) >= len(te_ls_pos):
            te = np.clip(te_raw[te_id2], 0, None)
        else:
            print(f"  {name:30s}: test_shape mismatch {len(te_raw)}")
            continue
        oof_mae = mae_fn(oof)
        r_oN, _ = pearsonr(te, oracle_new_t)
        print(f"  {name:30s}: OOF={oof_mae:.5f}  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r(oN)={r_oN:.4f}")
        if oof_mae < 10.0:
            good_cascade.append((name, oof, te, oof_mae, r_oN))
    except Exception as e:
        print(f"  {name:30s}: ERR {str(e)[:60]}")

# ============================================================
# Oracle_seq models — ALL (rescan for any high-unseen models)
# ============================================================
print("\n" + "="*75)
print("ORACLE_SEQ models — full rescan looking for high-unseen models")
print("="*75)
import glob
oof_files = glob.glob('results/oracle_seq/oof_seqC_*.npy')
all_models = []
for oof_path in sorted(oof_files):
    model = os.path.basename(oof_path).replace('oof_','').replace('.npy','')  # seqC_xxx
    suffix = model.replace('seqC_','')
    test_path = f'results/oracle_seq/test_C_{suffix}.npy'
    try:
        oof_raw = np.load(oof_path)
        oof = np.clip(oof_raw, 0, None)
        if len(oof) != len(y_true): continue
        oof_mae = mae_fn(oof)
        te_raw = np.load(test_path)
        te = np.clip(te_raw, 0, None)
        if len(te) != len(id_order): continue
        r_oN, _ = pearsonr(te, oracle_new_t)
        unseen_mean = te[unseen_mask].mean()
        all_models.append((model, oof_mae, te[seen_mask].mean(), unseen_mean, r_oN))
    except: continue

# Sort by unseen (descending)
all_models.sort(key=lambda x: -x[3])
print(f"\n  {'model':35s} {'OOF':>8} {'seen':>8} {'unseen':>8} {'r(oN)':>7} {'Δunseen':>8}")
for model, oof_mae, seen_m, unseen_m, r_oN in all_models:
    du = unseen_m - oracle_new_t[unseen_mask].mean()
    marker = " ← HIGH" if unseen_m > oracle_new_t[unseen_mask].mean() else ""
    print(f"  {model:35s}: OOF={oof_mae:.5f}  seen={seen_m:.3f}  unseen={unseen_m:.3f}  r={r_oN:.4f}  Δ={du:+.3f}{marker}")

# ============================================================
# Detailed analysis of highest-unseen cascade models
# ============================================================
print("\n" + "="*75)
print("Detailed: best cascade models for unseen-only blend")
print("="*75)
for name, oof, te, oof_mae, r_oN in sorted(good_cascade, key=lambda x: -x[2][unseen_mask].mean()):
    du = te[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    if te[unseen_mask].mean() > 22.0:  # only show meaningful ones
        print(f"\n  {name}: OOF={oof_mae:.5f}  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r={r_oN:.4f}  Δ={du:+.3f}")
        for w in [0.05, 0.10, 0.20]:
            ct = oracle_new_t.copy()
            ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*te[unseen_mask]
            ct = np.clip(ct, 0, None)
            duu = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
            print(f"    unseen blend w={w}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={duu:+.3f}")

print("\nDone.")
