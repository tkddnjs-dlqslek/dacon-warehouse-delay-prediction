import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t   = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t   = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3; w2 = fw['iter_r2'] + dr2; w3 = fw['iter_r3'] + dr3
    fx  = wm*mega + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof, 0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_oof, oracle_test = make_pred(0.0, 0.0, 0.0, 0.64, 0.0)
best_base_oof, best_base_test = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)
base_oof = mae(oracle_oof)
best_base_v = mae(best_base_oof)
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
seen_mask = test_raw['layout_id'].isin(train_layouts)
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')

# === Part 1: xgb_monotone grid search ===
print('\n=== Part 1: xgb_monotone blending grid search ===')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
print(f'mono OOF={mae(np.clip(mono_o,0,None)):.5f}  test={mono_t.mean():.3f}  unseen={mono_t[unseen_mask].mean():.3f}')

best_mono_w = None; best_mono_oof = base_oof
for w in np.arange(0.05, 0.35, 0.01):
    b_oof = np.clip((1-w)*best_base_oof + w*np.clip(mono_o,0,None), 0, None)
    b_test = np.clip((1-w)*best_base_test + w*np.clip(mono_t,0,None), 0, None)
    if mae(b_oof) < best_mono_oof:
        best_mono_oof = mae(b_oof)
        best_mono_w = w

print(f'Best mono weight: w={best_mono_w:.2f}  OOF={best_mono_oof:.5f} ({best_mono_oof-base_oof:+.6f})')
b_oof_mono = np.clip((1-best_mono_w)*best_base_oof + best_mono_w*np.clip(mono_o,0,None), 0, None)
b_test_mono = np.clip((1-best_mono_w)*best_base_test + best_mono_w*np.clip(mono_t,0,None), 0, None)
print(f'  test_mean={b_test_mono.mean():.3f}  unseen={b_test_mono[unseen_mask].mean():.3f}  seen={b_test_mono[seen_mask].mean():.3f}')

# Also try oracle_NEW + mono
for w in [0.05, 0.10, 0.12, 0.15, 0.20, 0.25]:
    b_oof = np.clip((1-w)*oracle_oof + w*np.clip(mono_o,0,None), 0, None)
    b_test = np.clip((1-w)*oracle_test + w*np.clip(mono_t,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  oracle+mono w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')

# === Part 2: All available OOF oracle_seq files (correct shape matching) ===
print('\n=== Part 2: All oracle_seq OOF files paired with test files ===')
oracle_dir = 'results/oracle_seq/'
oof_files = sorted(glob.glob(os.path.join(oracle_dir, 'oof_*.npy')))
results = []
for fp in oof_files:
    fn = os.path.basename(fp)
    arr = np.load(fp)
    if arr.shape[0] != len(train_raw): continue  # must be training-size
    test_fn = fn.replace('oof_seqC_', 'test_C_')
    test_path = os.path.join(oracle_dir, test_fn)
    if not os.path.exists(test_path): continue
    t_arr = np.load(test_path)
    if t_arr.shape[0] != len(test_raw): continue  # must be test-size
    oof_mae_v = mae(np.clip(arr,0,None))
    t_unseen = t_arr[unseen_mask].mean()
    t_seen = t_arr[seen_mask].mean()
    results.append((fn, oof_mae_v, t_arr.mean(), t_unseen, t_seen, t_unseen/t_seen if t_seen>0 else 0, arr, t_arr))

results.sort(key=lambda x: x[1])  # sort by OOF ASC (best first)
print(f'{"filename":35s}  {"OOF":>9}  {"test":>8}  {"t_unseen":>9}  {"t_seen":>9}  {"u/s":>7}')
for fn, oof_v, tm, tu, ts, ratio, _, _ in results:
    print(f'  {fn:35s}  {oof_v:>9.5f}  {tm:>8.3f}  {tu:>9.3f}  {ts:>9.3f}  {ratio:>7.3f}')

# === Part 3: Best OOF oracle_seq blended with best_base ===
print('\n=== Part 3: Best oracle_seq files + best_base blend ===')
print(f'{"name":50s}  {"OOF":>9}  {"ΔOOF":>8}  {"test":>8}  {"unseen":>9}  Marker')
best_combos = []
for fn, oof_v_comp, tm_comp, tu_comp, ts_comp, ratio_comp, oof_arr, test_arr in results:
    for w in [0.05, 0.10, 0.12, 0.15, 0.20]:
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(oof_arr,0,None), 0, None)
        b_test = np.clip((1-w)*best_base_test + w*np.clip(test_arr,0,None), 0, None)
        d_oof = mae(b_oof) - base_oof
        if d_oof < 0:
            name = f'{fn[:30]}_w{w:.2f}'
            best_combos.append((mae(b_oof), name, d_oof, b_test.mean(), b_test[unseen_mask].mean(), b_oof, b_test))

best_combos.sort()
print('OOF-improving combos:')
for oof_v, name, d_oof, tm, tu, _, _ in best_combos[:15]:
    print(f'  {name:50s}  {oof_v:>9.5f}  {d_oof:+.6f}  {tm:>8.3f}  {tu:>9.3f} *')

# === Part 4: Three-way blend (best_base + mono + another file) ===
print('\n=== Part 4: Three-way blends ===')
if best_combos:
    best_v2_oof, best_v2_name, best_v2_d, best_v2_tm, best_v2_tu, best_v2_oof_arr, best_v2_test_arr = best_combos[0]
    print(f'Best two-way combo: {best_v2_name}  OOF={best_v2_oof:.5f}')
    # Add a third component
    for fn, oof_v_comp, tm_comp, tu_comp, ts_comp, ratio_comp, oof_arr, test_arr in results[:5]:
        if fn.replace('oof_seqC_', '') in best_v2_name: continue
        for w3 in [0.03, 0.05, 0.08]:
            w_base = 1.0 - w3
            b3_oof = np.clip(w_base*best_v2_oof_arr + w3*np.clip(oof_arr,0,None), 0, None)
            b3_test = np.clip(w_base*best_v2_test_arr + w3*np.clip(test_arr,0,None), 0, None)
            d_oof = mae(b3_oof) - base_oof
            if d_oof < best_v2_d - 0.0001:
                print(f'  +{fn[:25]:25s} w={w3:.2f}: OOF={mae(b3_oof):.5f} ({d_oof:+.6f})  test={b3_test.mean():.3f} *')

# === Part 5: KNN layout bias correction (CV-validated) ===
print('\n=== Part 5: KNN layout bias correction ===')
train_raw['oracle_pred'] = oracle_oof
train_layout_stats = train_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization', 'mean'),
    inflow_mean=('order_inflow_15m', 'mean'),
    y_mean=('avg_delay_minutes_next_30m', 'mean'),
    oracle_mean=('oracle_pred', 'mean'),
    n=('ID', 'count')
).reset_index()
train_layout_stats['bias'] = train_layout_stats['y_mean'] - train_layout_stats['oracle_mean']

X_tr = train_layout_stats[['pack_mean', 'inflow_mean']].values
y_bias_tr = train_layout_stats['bias'].values
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)

test_layout_stats = test_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization', 'mean'),
    inflow_mean=('order_inflow_15m', 'mean'),
    n=('ID', 'count')
).reset_index()
X_test_all = scaler.transform(test_layout_stats[['pack_mean', 'inflow_mean']].values)

for k in [3, 5, 7, 10]:
    knn_corr_alpha = oracle_oof.copy()
    for lid in train_layout_stats['layout_id'].unique():
        m_hold = train_layout_stats['layout_id'] != lid
        knn = KNeighborsRegressor(n_neighbors=k, weights='distance')
        knn.fit(X_tr_sc[m_hold], y_bias_tr[m_hold])
        row = train_layout_stats[train_layout_stats['layout_id'] == lid]
        X_cv = scaler.transform(row[['pack_mean', 'inflow_mean']].values)
        pb = knn.predict(X_cv)[0]
        m_rows = train_raw['layout_id'] == lid
        knn_corr_alpha[m_rows] = np.clip(oracle_oof[m_rows] + 0.5*pb, 0, None)
    print(f'  KNN k={k} alpha=0.5: OOF={mae(knn_corr_alpha):.5f} ({mae(knn_corr_alpha)-base_oof:+.6f})')

# Best alpha for KNN k=5
best_knn_oof = base_oof; best_alpha_knn = 0.0
for alpha in np.arange(0.1, 1.1, 0.1):
    knn_c = oracle_oof.copy()
    for lid in train_layout_stats['layout_id'].unique():
        m_hold = train_layout_stats['layout_id'] != lid
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(X_tr_sc[m_hold], y_bias_tr[m_hold])
        row = train_layout_stats[train_layout_stats['layout_id'] == lid]
        X_cv = scaler.transform(row[['pack_mean', 'inflow_mean']].values)
        pb = knn.predict(X_cv)[0]
        m_rows = train_raw['layout_id'] == lid
        knn_c[m_rows] = np.clip(oracle_oof[m_rows] + alpha*pb, 0, None)
    if mae(knn_c) < best_knn_oof:
        best_knn_oof = mae(knn_c)
        best_alpha_knn = alpha
    print(f'  KNN k=5 alpha={alpha:.1f}: OOF={mae(knn_c):.5f} ({mae(knn_c)-base_oof:+.6f})')

print(f'\nBest KNN: alpha={best_alpha_knn:.1f}  OOF={best_knn_oof:.5f}')

# Apply KNN to test
knn_full = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_full.fit(X_tr_sc, y_bias_tr)
test_bias_pred = knn_full.predict(X_test_all)
test_bias_map = dict(zip(test_layout_stats['layout_id'], test_bias_pred))
print(f'Test KNN bias: SEEN={np.mean([test_bias_map[l] for l in test_layout_stats.loc[test_layout_stats["layout_id"].isin(train_layouts),"layout_id"]]):.3f}  UNSEEN={np.mean([test_bias_map[l] for l in test_layout_stats.loc[~test_layout_stats["layout_id"].isin(train_layouts),"layout_id"]]):.3f}')

# === Part 6: Save best new submissions ===
print('\n=== Part 6: Save best submissions ===')
sub = pd.read_csv('sample_submission.csv')

# 1. best_base + xgb_monotone (best OOF finder)
w_mono = best_mono_w
b_oof_m = np.clip((1-w_mono)*best_base_oof + w_mono*np.clip(mono_o,0,None), 0, None)
b_test_m = np.clip((1-w_mono)*best_base_test + w_mono*np.clip(mono_t,0,None), 0, None)
print(f'\nbest_base+mono w={w_mono:.2f}: OOF={mae(b_oof_m):.5f}  test={b_test_m.mean():.3f}  unseen={b_test_m[unseen_mask].mean():.3f}')
sub['avg_delay_minutes_next_30m'] = b_test_m
fname = f'submission_bb_mono_w{w_mono:.2f}_OOF{mae(b_oof_m):.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}')

# 2. Save top OOF-improving combos
for oof_v, name, d_oof, tm, tu, oof_arr, test_arr in best_combos[:3]:
    sub['avg_delay_minutes_next_30m'] = test_arr
    clean_name = name.replace('oof_seqC_', '').replace('.npy', '').replace(' ', '_')[:50]
    fname = f'submission_{clean_name}_OOF{oof_v:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}  OOF={oof_v:.5f} ({d_oof:+.6f})  test={tm:.3f}')

# 3. oracle_NEW + xgb_monotone at best weight for oracle base
best_w_oracle_mono = None; best_oof_om = base_oof
for w in np.arange(0.05, 0.30, 0.01):
    b = np.clip((1-w)*oracle_oof + w*np.clip(mono_o,0,None), 0, None)
    if mae(b) < best_oof_om:
        best_oof_om = mae(b)
        best_w_oracle_mono = w

if best_w_oracle_mono:
    b_oof_om = np.clip((1-best_w_oracle_mono)*oracle_oof + best_w_oracle_mono*np.clip(mono_o,0,None), 0, None)
    b_test_om = np.clip((1-best_w_oracle_mono)*oracle_test + best_w_oracle_mono*np.clip(mono_t,0,None), 0, None)
    print(f'\noracle+mono w={best_w_oracle_mono:.2f}: OOF={mae(b_oof_om):.5f}  test={b_test_om.mean():.3f}  unseen={b_test_om[unseen_mask].mean():.3f}')
    sub['avg_delay_minutes_next_30m'] = b_test_om
    fname = f'submission_oracle_mono_w{best_w_oracle_mono:.2f}_OOF{mae(b_oof_om):.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}')
else:
    print('oracle+mono: no OOF improvement found')

print('\nDone.')
