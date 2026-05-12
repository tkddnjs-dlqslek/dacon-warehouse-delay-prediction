"""
Comprehensive scipy optimization: ALL available oracle + mega34/mega37 + cascade gate.
Prev best: OOF=8.37851 (scipy 8-component + dual gate).
Goal: find better blend by including untried oracle variants and newer mega models.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
from scipy.optimize import minimize, differential_evolution

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

# === Load mega pkl files ===
print("Loading mega pkl files...", flush=True)
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)

def get_mega(d, id2, te_id2):
    oof = d['meta_avg_oof'][id2]
    test = d['meta_avg_test'][te_id2]
    return oof, test

mega33_oof, mega33_test = get_mega(d33, id2, te_id2)
mega34_oof, mega34_test = get_mega(d34, id2, te_id2)
mega37_oof, mega37_test = get_mega(d37, id2, te_id2)

print(f"  mega33 OOF MAE={np.mean(np.abs(mega33_oof-y_true)):.5f}")
print(f"  mega34 OOF MAE={np.mean(np.abs(mega34_oof-y_true)):.5f}")
print(f"  mega37 OOF MAE={np.mean(np.abs(mega37_oof-y_true)):.5f}")

# === Rank adj + iter ===
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]

# === Oracle seqC variants (row_id order — no id2 indexing needed) ===
oracle_names = [
    'xgb', 'log_v2', 'xgb_remaining',        # already in best blend
    'cb', 'lgb_dual', 'lgb_latepos',
    'lgb_remaining', 'lgb_remaining_v3',
    'xgb_v31', 'xgb_v31_sc',
    'xgb_combined', 'xgb_bestproxy',
    'ranklag', 'raw',
    'v2', 'v3', 'log', 'lgb_log1',
    'xgb_lag1', 'xgb_sch_raw2', 'lgb_stack',
]
oracle_oof  = {}
oracle_test = {}
for name in oracle_names:
    try:
        oracle_oof[name]  = np.load(f'results/oracle_seq/oof_seqC_{name}.npy')
        oracle_test[name] = np.load(f'results/oracle_seq/test_C_{name}.npy')
    except FileNotFoundError:
        print(f"  [MISSING] {name}")

print(f"\nOracle variants loaded: {list(oracle_oof.keys())}")

# === Print individual MAEs ===
print("\n=== Individual component MAEs ===")
for name in oracle_oof:
    mae = np.mean(np.abs(oracle_oof[name] - y_true))
    print(f"  oracle_{name}: {mae:.5f}")

# Correlation with best current (mega33-based blend)
fw_fixed = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
               iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
iter_r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
iter_r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
iter_r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
fixed_oof = (fw_fixed['mega33']*mega33_oof + fw_fixed['rank_adj']*rank_oof
           + fw_fixed['iter_r1']*iter_r1_oof + fw_fixed['iter_r2']*iter_r2_oof
           + fw_fixed['iter_r3']*iter_r3_oof)
best_oof_base = (0.64*fixed_oof + 0.12*oracle_oof['xgb']
                + 0.16*oracle_oof['log_v2'] + 0.08*oracle_oof['xgb_remaining'])
base_mae = np.mean(np.abs(best_oof_base - y_true))
print(f"\nCurrent best base OOF: {base_mae:.5f}")
for name in oracle_oof:
    corr = np.corrcoef(oracle_oof[name], best_oof_base)[0,1]
    print(f"  oracle_{name} corr w/base: {corr:.4f}")

# === Build candidate component matrix for scipy ===
# Strategy: use scipy base (no iter) + all oracle variants + mega34/37
# Scipy-proven weights: mega33 dominant, iter≈0 → start without iter

# Core components (no iter)
core_keys = ['mega33', 'mega34', 'mega37', 'rank_adj']
core_oof  = [mega33_oof, mega34_oof, mega37_oof, rank_oof]
core_test = [mega33_test, mega34_test, mega37_test, rank_test]

# All oracle variants
all_oracle_keys = list(oracle_oof.keys())
all_oracle_oof  = [oracle_oof[k] for k in all_oracle_keys]
all_oracle_test = [oracle_test[k] for k in all_oracle_keys]

# Full component list
all_keys = core_keys + all_oracle_keys
C_oof  = np.column_stack(core_oof  + all_oracle_oof)
C_test = np.column_stack(core_test + all_oracle_test)
print(f"\nTotal components: {len(all_keys)}")
print(f"C_oof shape: {C_oof.shape}")

def mae_fn(w, C, y):
    pred = np.clip(C @ np.abs(w) / np.abs(w).sum(), 0, None)
    return np.mean(np.abs(pred - y))

def mae_constrained(w):
    w = np.abs(w); w = w / w.sum()
    pred = np.clip(C_oof @ w, 0, None)
    return np.mean(np.abs(pred - y_true))

bounds = [(0, 1)] * len(all_keys)

# === Search 1: L-BFGS-B from scipy proven start ===
print(f"\n[Search1] L-BFGS-B from proven scipy weights", flush=True)
# Start from scipy-proven: mega33≈0.4997, rank≈0.1019, oracle_xgb≈0.1338, log_v2≈0.1741, rem≈0.0991
w0 = np.zeros(len(all_keys))
w0[all_keys.index('mega33')]        = 0.4997
w0[all_keys.index('rank_adj')]      = 0.1019
w0[all_keys.index('xgb')]           = 0.1338
w0[all_keys.index('log_v2')]        = 0.1741
w0[all_keys.index('xgb_remaining')] = 0.0991
w0 = w0 / w0.sum()

constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1.0}
res1 = minimize(lambda w: mae_fn(w, C_oof, y_true), w0,
                method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-11, 'gtol': 1e-8})
w1 = np.abs(res1.x); w1 = w1 / w1.sum()
mae1 = np.mean(np.abs(np.clip(C_oof @ w1, 0, None) - y_true))
print(f"  OOF={mae1:.5f}  delta={mae1-base_mae:+.5f}")
for k, w in zip(all_keys, w1):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# === Search 2: Differential evolution (global) — full component space ===
print(f"\n[Search2] Differential evolution (full space)", flush=True)
res2 = differential_evolution(mae_constrained, bounds, seed=42,
                               maxiter=500, popsize=20, tol=1e-9,
                               workers=1, polish=True,
                               callback=lambda xk, convergence: None)
w2 = np.abs(res2.x); w2 = w2 / w2.sum()
mae2 = np.mean(np.abs(np.clip(C_oof @ w2, 0, None) - y_true))
print(f"  OOF={mae2:.5f}  delta={mae2-base_mae:+.5f}")
for k, w in zip(all_keys, w2):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# === Search 3: Restrict to top performers, DE again ===
print(f"\n[Search3] Select top individual MAE components, re-run DE", flush=True)
# Pick components with individual MAE < 10.5
comp_maes = {k: np.mean(np.abs(oracle_oof[k]-y_true)) for k in oracle_oof}
comp_maes['mega33'] = np.mean(np.abs(mega33_oof-y_true))
comp_maes['mega34'] = np.mean(np.abs(mega34_oof-y_true))
comp_maes['mega37'] = np.mean(np.abs(mega37_oof-y_true))
comp_maes['rank_adj'] = np.mean(np.abs(rank_oof-y_true))

good_keys = [k for k in all_keys if comp_maes.get(k, 999) < 10.5]
good_idx  = [all_keys.index(k) for k in good_keys]
C_good    = C_oof[:, good_idx]
C_good_te = C_test[:, good_idx]
print(f"  Good components ({len(good_keys)}): {good_keys}")

def mae_good(w):
    w = np.abs(w); w = w / w.sum()
    pred = np.clip(C_good @ w, 0, None)
    return np.mean(np.abs(pred - y_true))

bounds_good = [(0, 1)] * len(good_keys)
res3 = differential_evolution(mae_good, bounds_good, seed=7,
                               maxiter=600, popsize=25, tol=1e-10,
                               workers=1, polish=True)
w3 = np.abs(res3.x); w3 = w3 / w3.sum()
mae3 = np.mean(np.abs(np.clip(C_good @ w3, 0, None) - y_true))
print(f"  OOF={mae3:.5f}  delta={mae3-base_mae:+.5f}")
for k, w in zip(good_keys, w3):
    if w > 0.005: print(f"    {k}: {w:.4f}")

# === Pick best base ===
results = [(mae1, w1, C_oof, C_test, "lbfgsb_full"),
           (mae2, w2, C_oof, C_test, "de_full"),
           (mae3, w3, C_good, C_good_te, "de_good")]
results.sort(key=lambda x: x[0])
best_mae_base, best_w, C_tr_best, C_te_best, best_name = results[0]
print(f"\n=== Best base: {best_name}  OOF={best_mae_base:.5f} ===")

base_oof_best  = np.clip(C_tr_best @ best_w, 0, None)
base_test_best = np.clip(C_te_best @ best_w, 0, None)

# === Apply cascade dual gate ===
print(f"\n[Cascade] Apply dual gate on best base", flush=True)
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best_overall = 8.37851
best_final_mae = best_mae_base
best_final_oof  = base_oof_best
best_final_test = base_test_best
best_final_name = best_name

# Grid search dual gate
print(f"  Searching dual gate...", flush=True)
p1_vals = [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15]
p2_vals = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]
for p1 in p1_vals:
    m1    = (clf_oof  > p1).astype(float)
    m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.015, 0.065, 0.005):
        b1    = (1-m1*w1)*base_oof_best  + m1*w1*lgb_rh_oof
        b1_te = (1-m1_te*w1)*base_test_best + m1_te*w1*lgb_rh_test
        for p2 in p2_vals:
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float)
            m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.015, 0.065, 0.005):
                blend = (1-m2*w2)*b1 + m2*w2*lgb_rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_final_mae:
                    best_final_mae  = mm
                    best_final_oof  = blend
                    best_final_test = (1-m2_te*w2)*b1_te + m2_te*w2*lgb_rm_test
                    best_final_name = f"{best_name}+dual_p{p1}_w{w1:.3f}+p{p2}_w{w2:.3f}"
                    print(f"★ {best_final_name}  OOF={mm:.5f}  delta={mm-base_mae:+.5f}", flush=True)

print(f"\n=== FINAL RESULT ===")
print(f"  Base:  OOF={best_mae_base:.5f}  delta={best_mae_base-base_mae:+.5f}")
print(f"  +Gate: OOF={best_final_mae:.5f}  delta={best_final_mae-base_mae:+.5f}")
print(f"  vs prev_best={prev_best_overall:.5f}: delta={best_final_mae-prev_best_overall:+.5f}")

if best_final_mae < prev_best_overall - 0.00005:
    print(f"\n★★ NEW BEST: {best_final_mae:.5f} ({best_final_name}) ★★")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_final_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mega_scipy_full_OOF{best_final_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")

    # Also save the weights
    weight_dict = {}
    if 'good' in best_name:
        for k, w in zip(good_keys, w3):
            if w > 0.001: weight_dict[k] = float(w)
    elif 'de_full' in best_name:
        for k, w in zip(all_keys, w2):
            if w > 0.001: weight_dict[k] = float(w)
    else:
        for k, w in zip(all_keys, w1):
            if w > 0.001: weight_dict[k] = float(w)
    print(f"  Weights: {weight_dict}")
else:
    print(f"\nNo improvement over {prev_best_overall:.5f}")
    # Save base anyway if it beats prev
    if best_mae_base < prev_best_overall - 0.00005:
        print(f"  But base alone is better: saving base")
        sample_sub = pd.read_csv('sample_submission.csv')
        sub = np.maximum(0, base_test_best)
        sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
        sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
        fname = f'submission_scipy_base_only_OOF{best_mae_base:.5f}.csv'
        sub_df.to_csv(fname, index=False)
        print(f"*** SAVED: {fname} ***")

print("Done.")
