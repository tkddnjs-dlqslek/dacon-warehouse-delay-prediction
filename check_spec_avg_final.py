import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
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

# spec_avg
spec_avg_o = np.clip(np.load('results/cascade/spec_avg_oof.npy')[id2], 0, None)
spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
print(f"spec_avg: OOF={mae_fn(spec_avg_o):.5f}  seen={spec_avg_t[seen_mask].mean():.3f}  unseen={spec_avg_t[unseen_mask].mean():.3f}")
print(f"  r(oracle_NEW) = {pearsonr(spec_avg_t, oracle_new_t)[0]:.4f}")

# spec_v2_avg
try:
    spec_v2_o = np.clip(np.load('results/cascade/spec_v2_avg_oof.npy')[id2], 0, None)
    spec_v2_t = np.clip(np.load('results/cascade/spec_v2_avg_test.npy')[te_id2], 0, None)
    print(f"spec_v2_avg: OOF={mae_fn(spec_v2_o):.5f}  seen={spec_v2_t[seen_mask].mean():.3f}  unseen={spec_v2_t[unseen_mask].mean():.3f}")
    print(f"  r(oracle_NEW) = {pearsonr(spec_v2_t, oracle_new_t)[0]:.4f}")
except Exception as e:
    print(f"spec_v2_avg: {e}")

# Blend oracle_NEW with spec_avg at various weights
# spec_avg: seen=18.?, unseen=25.964 — higher unseen
print("\n" + "="*60)
print("oracle_NEW + spec_avg blend (spec_avg has higher unseen)")
print("="*60)

sub_tmpl = pd.read_csv('sample_submission.csv')
for w in [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]:
    blend_test = np.clip((1-w)*oracle_new_t + w*spec_avg_t, 0, None)
    blend_oof  = np.clip((1-w)*oracle_new_t[:0] + (1-w)*0 + 0, 0, None)  # placeholder
    # compute OOF blend for oracle_NEW + spec_avg
    # Need oracle_NEW OOF to blend properly
    print(f"  w={w:.2f}: seen={blend_test[seen_mask].mean():.3f}  unseen={blend_test[unseen_mask].mean():.3f}")

# Unseen-only blend with spec_avg
print("\noracle_NEW + spec_avg (unseen only)")
for w in [0.05, 0.10, 0.15, 0.20]:
    blend_test = oracle_new_t.copy()
    blend_test[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*spec_avg_t[unseen_mask]
    blend_test = np.clip(blend_test, 0, None)
    delta_u = blend_test[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  w={w:.2f}: seen={blend_test[seen_mask].mean():.3f}  unseen={blend_test[unseen_mask].mean():.3f}  Δunseen={delta_u:+.3f}")
    if w in [0.10, 0.20]:
        fname = f"FINAL_NEW_oN_specAvg_u{int(w*100)}_OOF8.3825.csv"
        sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = blend_test
        sub.to_csv(fname, index=False)
        print(f"    Saved: {fname}")

print("\nDone.")
