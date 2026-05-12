"""
Master chain: waits for RF OOF file, then runs analyze → ET → log_v3 → final analysis.
Start this in background BEFORE RF completes; it will block until RF output appears.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

RF_OOF = 'results/oracle_seq/oof_seqC_rf.npy'

print("Waiting for RF oracle to complete...", flush=True)
while not os.path.exists(RF_OOF):
    time.sleep(30)
print("RF done! Starting chain...", flush=True)

steps = [
    ('analyze_with_rf', ['analyze_with_rf.py']),
    ('oracle-ET',       ['train_oracle_et.py']),
    ('oracle-log-v3',   ['train_oracle_log_v3.py']),
]

for name, args in steps:
    print(f"\n{'='*50}", flush=True)
    print(f"Starting: {name}", flush=True)
    print(f"{'='*50}", flush=True)
    ret = subprocess.run([sys.executable] + args, capture_output=False)
    print(f"{name} finished (exit={ret.returncode})", flush=True)

# Final multi-oracle per-position analysis
print("\n=== FINAL ANALYSIS ===", flush=True)
import numpy as np, pandas as pd, pickle

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

# Load all oracle models
oracles = {}
for name, path in [('xgb','results/oracle_seq/oof_seqC_xgb.npy'),
                   ('lv2','results/oracle_seq/oof_seqC_log_v2.npy'),
                   ('rf', 'results/oracle_seq/oof_seqC_rf.npy'),
                   ('et', 'results/oracle_seq/oof_seqC_et.npy'),
                   ('lv3','results/oracle_seq/oof_seqC_log_v3.npy')]:
    if os.path.exists(path):
        oracles[name] = np.load(path)
        m = np.mean(np.abs(oracles[name]-y_true))
        c = np.corrcoef(y_true-fixed_oof, oracles[name]-fixed_oof)[0,1]
        print(f"  {name}: MAE={m:.4f}  residual_corr={c:.4f}", flush=True)

oracle_names = list(oracles.keys())
print(f"\nOracle models available: {oracle_names}", flush=True)

# Per-position N-way grid optimization
from sklearn.model_selection import GroupKFold
STEP = 0.04
N = len(oracle_names)
print(f"\nPer-position {N+1}-way grid optimization...", flush=True)

per_pos_w = {}
for pos in range(25):
    mask = row_in_sc == pos
    f_pos = fixed_oof[mask]; y_pos = y_true[mask]
    oracs_pos = {n: oracles[n][mask] for n in oracle_names}
    best_m = np.mean(np.abs(f_pos - y_pos))
    best_w = tuple([0.0] * N)

    # Grid search: for N=2 → 2D, N=3 → 3D, etc.
    # Limit total oracle weight to 0.60 to avoid degenerate solutions
    def grid_search_3way(names, fpos, ypos, fixed_pos, oracle_pos):
        best_m = np.mean(np.abs(fixed_pos - ypos))
        best_w = (0., 0., 0.)
        for w0 in np.arange(0, 0.65, STEP):
            for w1 in np.arange(0, 0.65-w0, STEP):
                for w2 in np.arange(0, 0.65-w0-w1, STEP):
                    if w0+w1+w2 > 0.60: continue
                    bl = (1-w0-w1-w2)*fixed_pos + w0*oracle_pos[names[0]] + w1*oracle_pos[names[1]] + w2*oracle_pos[names[2]]
                    m = np.mean(np.abs(bl - ypos))
                    if m < best_m: best_m = m; best_w = (w0, w1, w2)
        return best_m, best_w

    if N <= 3:
        # Pad to 3
        on = oracle_names + [oracle_names[0]] * (3-N)
        op = {n: oracles[n][mask] for n in oracle_names}
        op.update({n: np.zeros(mask.sum()) for n in on if n not in oracle_names})
        best_m, best_w3 = grid_search_3way(on[:3], f_pos, y_pos, f_pos, op)
        best_w = best_w3[:N]
    else:
        # Full N-way: iterate over all subsets, take best 3-combo
        from itertools import combinations
        for combo in combinations(range(N), 3):
            n3 = [oracle_names[i] for i in combo]
            op3 = {n: oracles[n][mask] for n in n3}
            bm, bw = grid_search_3way(n3, f_pos, y_pos, f_pos, op3)
            if bm < best_m:
                best_m = bm
                ww = [0.0]*N
                for ci, wi in zip(combo, bw): ww[ci] = wi
                best_w = tuple(ww)

    per_pos_w[pos] = best_w
    delta = best_m - np.mean(np.abs(f_pos - y_pos))
    print(f"  pos={pos:2d}: {' '.join(f'{n}={w:.2f}' for n,w in zip(oracle_names,best_w))}  delta={delta:+.4f}", flush=True)

# Full blend
blend_oof = fixed_oof.copy()
for pos in range(25):
    mask = row_in_sc == pos
    w = per_pos_w[pos]
    wsum = sum(w)
    bl = (1-wsum)*fixed_oof[mask]
    for n, wi in zip(oracle_names, w):
        bl += wi * oracles[n][mask]
    blend_oof[mask] = bl
blend_mae = np.mean(np.abs(blend_oof - y_true))
print(f"\nFinal per-pos N-way OOF MAE: {blend_mae:.4f}  delta={blend_mae-fixed_mae:+.4f}", flush=True)

# 5-fold CV check
gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
fold_deltas = []
for _, val_idx in gkf.split(np.arange(len(train_raw)), groups=groups):
    bl_val = blend_oof[val_idx]
    f_val  = fixed_oof[val_idx]
    delta  = np.mean(np.abs(bl_val - y_true[val_idx])) - np.mean(np.abs(f_val - y_true[val_idx]))
    fold_deltas.append(delta)
print(f"Fold deltas: {[f'{x:.4f}' for x in fold_deltas]} ({sum(x<0 for x in fold_deltas)}/5 neg)", flush=True)

PREV_BEST = 8.3800
if blend_mae < PREV_BEST - 0.0002 and sum(x < 0 for x in fold_deltas) >= 4:
    mega_test  = d['meta_avg_test'][te_id_to_ls]
    rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
    iter1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
    iter2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
    iter3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
    fixed_test = (fw['mega33']*mega_test + fw['rank_adj']*rank_test +
                  fw['iter_r1']*iter1_test + fw['iter_r2']*iter2_test + fw['iter_r3']*iter3_test)
    oracle_tests = {}
    for name, path in [('xgb','results/oracle_seq/test_C_xgb.npy'),
                       ('lv2','results/oracle_seq/test_C_log_v2.npy'),
                       ('rf', 'results/oracle_seq/test_C_rf.npy'),
                       ('et', 'results/oracle_seq/test_C_et.npy'),
                       ('lv3','results/oracle_seq/test_C_log_v3.npy')]:
        if name in oracle_names and os.path.exists(path):
            oracle_tests[name] = np.load(path)

    test_row_sc = test_raw['row_in_sc'].values
    blend_test = fixed_test.copy()
    for pos in range(25):
        mask = test_row_sc == pos
        w = per_pos_w[pos]
        wsum = sum(w)
        bl = (1-wsum)*fixed_test[mask]
        for n, wi in zip(oracle_names, w):
            if n in oracle_tests:
                bl += wi * oracle_tests[n][mask]
        blend_test[mask] = bl
    blend_test = np.maximum(0, blend_test)

    sample_sub = pd.read_csv('sample_submission.csv')
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': blend_test})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_final_Nway_OOF{blend_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** NEW BEST! Saved: {fname} ***", flush=True)
else:
    print(f"\nNo improvement over prev best {PREV_BEST:.4f}", flush=True)

print("\nAll done!", flush=True)
