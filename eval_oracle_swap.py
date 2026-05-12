"""
Quick targeted test: swap oracle_xgb → oracle_xgb_combined/xgb_v31 in the proven scipy base.
These have better individual MAE (8.425/8.426 vs 8.438).
Also try oracle_cb as drop-in replacement.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from scipy.optimize import minimize

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)

# Base components (no iter, proven)
base_oof_comps = {
    'mega33':    d33['meta_avg_oof'][id2],
    'rank_adj':  np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb':       np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_log_v2':    np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_rem':   np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
    # New candidates
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':   np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'oracle_cb':        np.load('results/oracle_seq/oof_seqC_cb.npy'),
    'oracle_xgb_bestproxy': np.load('results/oracle_seq/oof_seqC_xgb_bestproxy.npy'),
}
base_test_comps = {
    'mega33':    d33['meta_avg_test'][te_id2],
    'rank_adj':  np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb':       np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_log_v2':    np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_rem':   np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':   np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'oracle_cb':        np.load('results/oracle_seq/test_C_cb.npy'),
    'oracle_xgb_bestproxy': np.load('results/oracle_seq/test_C_xgb_bestproxy.npy'),
}

def run_scipy(keys, C_oof, C_test, tag):
    def mae_fn(w):
        w = np.abs(w); w = w / w.sum()
        return np.mean(np.abs(np.clip(C_oof @ w, 0, None) - y_true))
    bounds = [(0, 1)] * len(keys)
    # Start from uniform
    w0 = np.ones(len(keys)) / len(keys)
    res = minimize(mae_fn, w0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 5000, 'ftol': 1e-11})
    w_opt = np.abs(res.x); w_opt /= w_opt.sum()
    mae_val = np.mean(np.abs(np.clip(C_oof @ w_opt, 0, None) - y_true))
    print(f"  [{tag}] OOF={mae_val:.5f}")
    for k, w in zip(keys, w_opt):
        if w > 0.005: print(f"    {k}: {w:.4f}")
    test_pred = np.clip(C_test @ w_opt, 0, None)
    return mae_val, w_opt, test_pred

# Cascade helpers
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
m11    = (clf_oof  > 0.11).astype(float); m11_te = (clf_test > 0.11).astype(float)
m25    = (clf_oof  > 0.25).astype(float); m25_te = (clf_test > 0.25).astype(float)
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

def apply_cascade(base_oof, base_test):
    o = (1-m11*0.03)*base_oof + m11*0.03*rh_oof
    o = (1-m25*0.03)*o        + m25*0.03*rm_oof
    t = (1-m11_te*0.03)*base_test + m11_te*0.03*rh_te
    t = (1-m25_te*0.03)*t         + m25_te*0.03*rm_te
    return o, t

prev_best = 8.37851
best_mae = prev_best
best_pred_test = None
best_tag = None

print("=== Oracle swap experiments ===\n")

# Test sets to try
experiments = {
    'original_5comp': ['mega33','rank_adj','oracle_xgb','oracle_log_v2','oracle_xgb_rem'],
    'swap_xgb→combined': ['mega33','rank_adj','oracle_xgb_combined','oracle_log_v2','oracle_xgb_rem'],
    'swap_xgb→v31':     ['mega33','rank_adj','oracle_xgb_v31','oracle_log_v2','oracle_xgb_rem'],
    'add_combined':      ['mega33','rank_adj','oracle_xgb','oracle_xgb_combined','oracle_log_v2','oracle_xgb_rem'],
    'add_cb':            ['mega33','rank_adj','oracle_xgb','oracle_cb','oracle_log_v2','oracle_xgb_rem'],
    'add_v31':           ['mega33','rank_adj','oracle_xgb','oracle_xgb_v31','oracle_log_v2','oracle_xgb_rem'],
    'top4_oracle':       ['mega33','rank_adj','oracle_xgb_combined','oracle_xgb_v31','oracle_log_v2','oracle_xgb_rem'],
    'all_new_oracle':    ['mega33','rank_adj','oracle_xgb_combined','oracle_xgb_v31','oracle_cb','oracle_xgb_bestproxy','oracle_log_v2','oracle_xgb_rem'],
}

for tag, keys in experiments.items():
    C_tr = np.column_stack([base_oof_comps[k]  for k in keys])
    C_te = np.column_stack([base_test_comps[k] for k in keys])
    mae_base, w_opt, test_base = run_scipy(keys, C_tr, C_te, tag)
    # Apply cascade
    oof_casc, te_casc = apply_cascade(np.clip(C_tr @ w_opt, 0, None), test_base)
    mae_casc = np.mean(np.abs(oof_casc - y_true))
    print(f"    +cascade: {mae_casc:.5f}  delta_vs_best={mae_casc-prev_best:+.5f}")
    if mae_casc < best_mae:
        best_mae = mae_casc; best_pred_test = te_casc; best_tag = tag + '+cascade'
        print(f"    ★ NEW BEST: {best_mae:.5f}")
    elif mae_base < best_mae:
        best_mae = mae_base; best_pred_test = test_base; best_tag = tag
        print(f"    ★ NEW BEST (base): {best_mae:.5f}")
    print()

print(f"\n=== SUMMARY ===")
print(f"  Prev best: {prev_best:.5f}")
print(f"  Best found: {best_mae:.5f}  ({best_tag})")

if best_mae < prev_best - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_pred_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_swap_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print("No improvement.")
print("Done.")
