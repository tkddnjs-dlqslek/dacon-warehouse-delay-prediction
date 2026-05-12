"""
Final comprehensive per-position blend using ALL available oracle models.
Run after all oracle training scripts complete.
Finds best submission using N-way per-position grid optimization.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from itertools import combinations

print("Loading base data...", flush=True)
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
test_raw['row_in_sc'] = test_raw.groupby(['layout_id','scenario_id']).cumcount()

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
iter1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
iter2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
iter3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed_oof = (fw['mega33']*mega_oof + fw['rank_adj']*rank_oof +
             fw['iter_r1']*iter1_oof + fw['iter_r2']*iter2_oof + fw['iter_r3']*iter3_oof)
y_true = train_raw['avg_delay_minutes_next_30m'].values
fixed_mae = np.mean(np.abs(fixed_oof - y_true))
row_in_sc = train_raw['row_in_sc'].values
print(f"FIXED OOF MAE: {fixed_mae:.4f}", flush=True)

# Load all available oracle models
ORACLE_PATHS_OOF = {
    'xgb':       'results/oracle_seq/oof_seqC_xgb.npy',
    'lv2':       'results/oracle_seq/oof_seqC_log_v2.npy',
    'xgb_fp':    'results/oracle_seq/oof_seqC_xgb_fixedproxy.npy',
    'xgb_log2':  'results/oracle_seq/oof_seqC_xgb_log2.npy',
    'rf_log':    'results/oracle_seq/oof_seqC_rf_log.npy',
    'cumstats':  'results/oracle_seq/oof_seqC_cumstats.npy',
    'latepos':   'results/oracle_seq/oof_seqC_lgb_latepos.npy',
    'log4':      'results/oracle_seq/oof_seqC_lgb_log4.npy',
    'xgb_lag1':  'results/oracle_seq/oof_seqC_xgb_lag1.npy',
}
ORACLE_PATHS_TEST = {
    'xgb':       'results/oracle_seq/test_C_xgb.npy',
    'lv2':       'results/oracle_seq/test_C_log_v2.npy',
    'xgb_fp':    'results/oracle_seq/test_C_xgb_fixedproxy.npy',
    'xgb_log2':  'results/oracle_seq/test_C_xgb_log2.npy',
    'rf_log':    'results/oracle_seq/test_C_rf_log.npy',
    'cumstats':  'results/oracle_seq/test_C_cumstats.npy',
    'latepos':   'results/oracle_seq/test_C_lgb_latepos.npy',
    'log4':      'results/oracle_seq/test_C_lgb_log4.npy',
    'xgb_lag1':  'results/oracle_seq/test_C_xgb_lag1.npy',
}

oracles_oof  = {}
oracles_test = {}
res_fixed = y_true - fixed_oof
for name in ORACLE_PATHS_OOF:
    p = ORACLE_PATHS_OOF[name]
    if os.path.exists(p):
        arr = np.load(p)
        oracles_oof[name] = arr
        m = np.mean(np.abs(arr - y_true))
        c = np.corrcoef(res_fixed, arr - fixed_oof)[0,1]
        print(f"  {name}: MAE={m:.4f}  residual_corr={c:.4f}", flush=True)
        pt = ORACLE_PATHS_TEST[name]
        if os.path.exists(pt):
            oracles_test[name] = np.load(pt)

oracle_names = list(oracles_oof.keys())
N = len(oracle_names)
print(f"\nAvailable oracle models ({N}): {oracle_names}", flush=True)

if N == 0:
    print("No oracle models found. Exiting.", flush=True)
    sys.exit(1)

STEP = 0.04

def per_pos_grid(pos_mask, fixed_pos, y_pos, oracle_pos_dict, names):
    """Grid search over all combinations of up to 3 oracle models for this position."""
    best_m = np.mean(np.abs(fixed_pos - y_pos))
    best_w = {n: 0.0 for n in names}
    n = len(names)

    # Try all pairs and triples
    single_combos = [(i,) for i in range(n)]
    pair_combos   = [(i,j) for i in range(n) for j in range(i+1,n)]
    triple_combos = [(i,j,k) for i in range(n) for j in range(i+1,n) for k in range(j+1,n)]

    all_combos = single_combos + pair_combos + triple_combos

    for combo in all_combos:
        combo_names = [names[i] for i in combo]
        nc = len(combo)
        # Grid search for this combo
        ranges = [np.arange(0, 0.65, STEP) for _ in range(nc)]

        if nc == 1:
            for w0 in ranges[0]:
                if w0 > 0.60: continue
                bl = (1-w0)*fixed_pos + w0*oracle_pos_dict[combo_names[0]]
                m = np.mean(np.abs(bl - y_pos))
                if m < best_m:
                    best_m = m
                    best_w = {nn: 0.0 for nn in names}
                    best_w[combo_names[0]] = w0
        elif nc == 2:
            for w0 in ranges[0]:
                for w1 in np.arange(0, 0.65-w0, STEP):
                    if w0+w1 > 0.60: continue
                    bl = (1-w0-w1)*fixed_pos + w0*oracle_pos_dict[combo_names[0]] + w1*oracle_pos_dict[combo_names[1]]
                    m = np.mean(np.abs(bl - y_pos))
                    if m < best_m:
                        best_m = m
                        best_w = {nn: 0.0 for nn in names}
                        best_w[combo_names[0]] = w0
                        best_w[combo_names[1]] = w1
        elif nc == 3:
            for w0 in ranges[0]:
                for w1 in np.arange(0, 0.65-w0, STEP):
                    for w2 in np.arange(0, 0.65-w0-w1, STEP):
                        if w0+w1+w2 > 0.60: continue
                        bl = ((1-w0-w1-w2)*fixed_pos +
                              w0*oracle_pos_dict[combo_names[0]] +
                              w1*oracle_pos_dict[combo_names[1]] +
                              w2*oracle_pos_dict[combo_names[2]])
                        m = np.mean(np.abs(bl - y_pos))
                        if m < best_m:
                            best_m = m
                            best_w = {nn: 0.0 for nn in names}
                            best_w[combo_names[0]] = w0
                            best_w[combo_names[1]] = w1
                            best_w[combo_names[2]] = w2
    return best_m, best_w

print(f"\nPer-position grid optimization ({N} oracle models)...", flush=True)
per_pos_weights = {}
for pos in range(25):
    mask = row_in_sc == pos
    f_pos = fixed_oof[mask]
    y_pos = y_true[mask]
    op = {n: oracles_oof[n][mask] for n in oracle_names}
    best_m, best_w = per_pos_grid(mask, f_pos, y_pos, op, oracle_names)
    per_pos_weights[pos] = best_w
    delta = best_m - np.mean(np.abs(f_pos - y_pos))
    w_str = ' '.join(f'{n}={best_w[n]:.2f}' for n in oracle_names if best_w[n] > 0)
    print(f"  pos={pos:2d}: [{w_str if w_str else 'FIXED only'}]  delta={delta:+.4f}", flush=True)

# Full OOF blend
blend_oof = fixed_oof.copy()
for pos in range(25):
    mask = row_in_sc == pos
    w = per_pos_weights[pos]
    wsum = sum(w.values())
    bl = (1-wsum)*fixed_oof[mask]
    for n in oracle_names:
        if w[n] > 0:
            bl += w[n] * oracles_oof[n][mask]
    blend_oof[mask] = bl
blend_mae = np.mean(np.abs(blend_oof - y_true))
print(f"\nFull per-pos N-way OOF MAE: {blend_mae:.4f}  delta={blend_mae-fixed_mae:+.4f}", flush=True)

# 5-fold CV
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
fold_deltas = []
for _, val_idx in gkf.split(np.arange(len(train_raw)), groups=groups):
    bl_val = blend_oof[val_idx]; f_val = fixed_oof[val_idx]
    delta  = np.mean(np.abs(bl_val - y_true[val_idx])) - np.mean(np.abs(f_val - y_true[val_idx]))
    fold_deltas.append(delta)
print(f"Fold deltas: {[f'{x:.4f}' for x in fold_deltas]} ({sum(x<0 for x in fold_deltas)}/5 neg)", flush=True)

# Previous best baseline
PREV_BEST = 8.3800
print(f"\nComparison: new={blend_mae:.4f}  prev_best={PREV_BEST:.4f}  improvement={PREV_BEST-blend_mae:.4f}", flush=True)

if blend_mae < PREV_BEST - 0.0003 and sum(x < 0 for x in fold_deltas) >= 4:
    mega_test  = d['meta_avg_test'][te_id_to_ls]
    rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
    iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
    iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
    iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
    fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                  fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)

    test_row_sc = test_raw['row_in_sc'].values
    blend_test = fixed_test.copy()
    for pos in range(25):
        mask = test_row_sc == pos
        w = per_pos_weights[pos]
        wsum = sum(w.values())
        bl = (1-wsum)*fixed_test[mask]
        for n in oracle_names:
            if w[n] > 0 and n in oracles_test:
                bl += w[n] * oracles_test[n][mask]
        blend_test[mask] = bl
    blend_test = np.maximum(0, blend_test)

    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': blend_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_all_Nway_OOF{blend_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo significant improvement (threshold: {PREV_BEST-0.0003:.4f})", flush=True)

print("\nDone.", flush=True)
