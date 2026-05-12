"""
Global static oracle blend — no per-position splitting.
Key insight: 5-way static (OOF 8.3831) -> LB 9.7558 outperforms
N-way per-pos (OOF 8.3796) -> LB 9.7569 (overfit!).
Static global weights (2-4 params) generalize better than per-pos (75 params).
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from itertools import combinations

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_ls = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[i] for i in test_raw['ID'].values]

mega_oof  = d['meta_avg_oof'][id_to_ls]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls]
i1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
i2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
i3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed = fw['mega33']*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*i1_oof+fw['iter_r2']*i2_oof+fw['iter_r3']*i3_oof
y = train_raw['avg_delay_minutes_next_30m'].values
fixed_mae = np.mean(np.abs(fixed - y))
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
i1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
i2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
i3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
fixed_test = fw['mega33']*mega_test+fw['rank_adj']*rank_test+fw['iter_r1']*i1_test+fw['iter_r2']*i2_test+fw['iter_r3']*i3_test

# Load oracle models (skip if missing)
ORACLE_OOF = {
    'xgb':    'results/oracle_seq/oof_seqC_xgb.npy',
    'lv2':    'results/oracle_seq/oof_seqC_log_v2.npy',
    'lag1':   'results/oracle_seq/oof_seqC_xgb_lag1.npy',
    'late':   'results/oracle_seq/oof_seqC_lgb_latepos.npy',
    'xgbfp':  'results/oracle_seq/oof_seqC_xgb_fixedproxy.npy',
    'lgbfp':  'results/oracle_seq/oof_seqC_lgb_fixedproxy.npy',
}
ORACLE_TEST = {
    'xgb':    'results/oracle_seq/test_C_xgb.npy',
    'lv2':    'results/oracle_seq/test_C_log_v2.npy',
    'lag1':   'results/oracle_seq/test_C_xgb_lag1.npy',
    'late':   'results/oracle_seq/test_C_lgb_latepos.npy',
    'xgbfp':  'results/oracle_seq/test_C_xgb_fixedproxy.npy',
    'lgbfp':  'results/oracle_seq/test_C_lgb_fixedproxy.npy',
}

train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()
late_mask_tr = (train_raw['row_in_sc'].values >= 17)
late_mask_te = (test_raw['row_in_sc'].values >= 17)

oracles_oof  = {}
oracles_test = {}
for name, path in ORACLE_OOF.items():
    if os.path.exists(path):
        arr = np.load(path)
        # latepos: fill early-position zeros with fixed (avoids penalizing pos 0-16)
        if name == 'late':
            arr_filled = fixed.copy()
            arr_filled[late_mask_tr] = arr[late_mask_tr]
            arr = arr_filled
        oracles_oof[name] = arr
        mae = np.mean(np.abs(arr - y))
        res = y - fixed
        corr = np.corrcoef(res, arr - fixed)[0,1]
        late_mae = np.mean(np.abs(arr[late_mask_tr] - y[late_mask_tr]))
        print(f"  {name}: MAE={mae:.4f}  residual_corr={corr:.4f}  late_MAE={late_mae:.4f}", flush=True)
        tp = ORACLE_TEST[name]
        if os.path.exists(tp):
            tarr = np.load(tp)
            if name == 'late':
                tarr_filled = fixed_test.copy()
                tarr_filled[late_mask_te] = tarr[late_mask_te]
                tarr = tarr_filled
            oracles_test[name] = tarr

names = list(oracles_oof.keys())
N = len(names)
print(f"\nAvailable oracles ({N}): {names}", flush=True)
if N == 0:
    print("No oracle models found.", flush=True)
    sys.exit(0)

STEP = 0.02
MAX_W = 0.60  # max total oracle weight

print(f"\nGlobal static grid search (STEP={STEP})...", flush=True)
best_mae = fixed_mae
best_ws = {n: 0.0 for n in names}

# Try all 1,2,3-oracle combos with global static weights
for nc in range(1, min(N+1, 4)):
    for combo in combinations(names, nc):
        cn = len(combo)
        if cn == 1:
            for w0 in np.arange(0, MAX_W+STEP/2, STEP):
                if w0 > MAX_W: continue
                bl = (1-w0)*fixed + w0*oracles_oof[combo[0]]
                m = np.mean(np.abs(bl - y))
                if m < best_mae:
                    best_mae = m
                    best_ws = {n: 0.0 for n in names}
                    best_ws[combo[0]] = round(w0, 4)
        elif cn == 2:
            for w0 in np.arange(0, MAX_W+STEP/2, STEP):
                for w1 in np.arange(0, MAX_W-w0+STEP/2, STEP):
                    if w0+w1 > MAX_W: continue
                    bl = (1-w0-w1)*fixed + w0*oracles_oof[combo[0]] + w1*oracles_oof[combo[1]]
                    m = np.mean(np.abs(bl - y))
                    if m < best_mae:
                        best_mae = m
                        best_ws = {n: 0.0 for n in names}
                        best_ws[combo[0]] = round(w0, 4)
                        best_ws[combo[1]] = round(w1, 4)
        elif cn == 3:
            for w0 in np.arange(0, MAX_W+STEP/2, STEP):
                for w1 in np.arange(0, MAX_W-w0+STEP/2, STEP):
                    for w2 in np.arange(0, MAX_W-w0-w1+STEP/2, STEP):
                        if w0+w1+w2 > MAX_W: continue
                        bl = ((1-w0-w1-w2)*fixed +
                              w0*oracles_oof[combo[0]] +
                              w1*oracles_oof[combo[1]] +
                              w2*oracles_oof[combo[2]])
                        m = np.mean(np.abs(bl - y))
                        if m < best_mae:
                            best_mae = m
                            best_ws = {n: 0.0 for n in names}
                            best_ws[combo[0]] = round(w0, 4)
                            best_ws[combo[1]] = round(w1, 4)
                            best_ws[combo[2]] = round(w2, 4)

# Build best blend OOF
wsum = sum(best_ws.values())
blend_oof = (1-wsum)*fixed + sum(best_ws[n]*oracles_oof[n] for n in names)
blend_mae = np.mean(np.abs(blend_oof - y))

w_str = ' '.join(f'{n}={best_ws[n]:.3f}' for n in names if best_ws[n] > 0)
print(f"\nBest static global blend: [{w_str}]", flush=True)
print(f"OOF MAE: {blend_mae:.4f}  (vs fixed {fixed_mae:.4f}, delta={blend_mae-fixed_mae:+.4f})", flush=True)

# 5-fold check
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
fold_deltas = []
for _, val_idx in gkf.split(np.arange(len(train_raw)), groups=groups):
    bl = blend_oof[val_idx]; f = fixed[val_idx]
    fold_deltas.append(np.mean(np.abs(bl-y[val_idx])) - np.mean(np.abs(f-y[val_idx])))
print(f"Fold deltas: {[f'{x:.4f}' for x in fold_deltas]} ({sum(x<0 for x in fold_deltas)}/5 neg)", flush=True)

# Compare to the 5-way static reference (OOF 8.3831 -> LB 9.7558)
REF_STATIC_OOF = 8.3831
xgb_w_ref, lv2_w_ref = 0.12, 0.20
ref_blend = (1-xgb_w_ref-lv2_w_ref)*fixed + xgb_w_ref*oracles_oof.get('xgb', fixed) + lv2_w_ref*oracles_oof.get('lv2', fixed)
ref_mae = np.mean(np.abs(ref_blend - y))
print(f"\nRef 5-way static (xgb=0.12,lv2=0.20): OOF={ref_mae:.4f}", flush=True)
print(f"New vs ref: delta={blend_mae-ref_mae:+.4f}", flush=True)

PREV_BEST = REF_STATIC_OOF
threshold = PREV_BEST - 0.0005  # require meaningful improvement
n_neg = sum(x < 0 for x in fold_deltas)

if blend_mae < threshold and n_neg >= 4:
    wsum = sum(best_ws.values())
    blend_test = (1-wsum)*fixed_test
    for n in names:
        if best_ws[n] > 0 and n in oracles_test:
            blend_test += best_ws[n] * oracles_test[n]
    blend_test = np.maximum(0, blend_test)
    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': blend_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_static_global_OOF{blend_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} ***", flush=True)
    print(f"*** RECOMMEND SUBMIT if this beats current best LB 9.7558 ***", flush=True)
else:
    print(f"\nNo significant improvement (threshold={threshold:.4f}, n_neg={n_neg}/5)", flush=True)
    if blend_mae < PREV_BEST:
        print(f"  OOF improved but below threshold or folds insufficient", flush=True)

print("\nDone.", flush=True)
