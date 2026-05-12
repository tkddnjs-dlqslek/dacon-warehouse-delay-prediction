import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
from sklearn.model_selection import GroupKFold

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
    wm = fw['mega33'] - dr2 - dr3
    w2 = fw['iter_r2'] + dr2
    w3 = fw['iter_r3'] + dr3
    fx  = wm*mega + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0 - wf; wxgb = 0.12*w_rem/0.36; wlv2 = 0.16*w_rem/0.36; wrem = 0.08*w_rem/0.36
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

# === Part 1: Multiplicative per-layout calibration (CV validated) ===
print('\n=== Part 1: Multiplicative per-layout calibration ===')
train_raw['oracle_pred'] = oracle_oof

def apply_mult_calib(preds_oof, alpha=1.0):
    """CV-validated multiplicative per-layout calibration"""
    corr = preds_oof.copy()
    for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        tr_data = train_raw.iloc[tr_idx]
        val_data = train_raw.iloc[val_idx]
        # Compute per-layout ratio from training fold
        layout_ratios = {}
        for lid, grp in tr_data.groupby('layout_id'):
            oof_mean = preds_oof[tr_idx[tr_data.index.get_loc(grp.index[0]):tr_data.index.get_loc(grp.index[-1])+1]].mean() if False else \
                       preds_oof[np.array([i for i in tr_idx if train_raw.iloc[i]['layout_id'] == lid])].mean()
            y_mean = grp['avg_delay_minutes_next_30m'].mean()
            if oof_mean > 0:
                layout_ratios[lid] = y_mean / oof_mean
        # Apply to val fold (same layout — note: GroupKFold means val layouts are NOT in train fold)
        # So this doesn't actually apply to val rows for the same layouts
        # Instead, apply to val fold layouts using global ratio from training fold
        global_ratio = np.mean(list(layout_ratios.values()))
        for i in val_idx:
            lid = train_raw.iloc[i]['layout_id']
            ratio = layout_ratios.get(lid, global_ratio)
            corr[i] = np.clip(preds_oof[i] * (1 + alpha*(ratio-1)), 0, None)
    return corr

# Since GroupKFold puts each layout in exactly ONE fold,
# the val fold's layouts are never in the training fold.
# So multiplicative calibration using training fold's ratio won't apply to val layouts.
# Instead, compute global multiplicative calibration.
print('Testing global multiplicative calibration:')
for alpha in [0.2, 0.3, 0.5, 0.7, 1.0]:
    global_ratio = y_true.mean() / np.clip(oracle_oof, 1e-8, None).mean()
    corr = np.clip(oracle_oof * (1 + alpha*(global_ratio-1)), 0, None)
    print(f'  alpha={alpha:.1f} (global ratio={global_ratio:.4f}): OOF={mae(corr):.5f} ({mae(corr)-base_oof:+.6f})')

# Per-bin (by prediction level) multiplicative calibration
print('\nPer-prediction-level multiplicative calibration:')
pred_bins = np.percentile(oracle_oof, [20, 40, 60, 80])
for nbins in [4, 5, 6]:
    pred_q = np.percentile(oracle_oof, np.linspace(0, 100, nbins+1)[1:-1])
    bin_ratios = []
    for i in range(nbins):
        lo = pred_q[i-1] if i > 0 else -np.inf
        hi = pred_q[i] if i < nbins-1 else np.inf
        m = (oracle_oof >= lo) & (oracle_oof < hi)
        if m.sum() == 0: continue
        pred_m = np.clip(oracle_oof[m], 1e-8, None).mean()
        y_m = y_true[m].mean()
        bin_ratios.append((lo, hi, y_m/pred_m, m.sum()))
    # Apply
    corr = oracle_oof.copy()
    for lo, hi, ratio, n in bin_ratios:
        m = (oracle_oof >= lo) & (oracle_oof < hi)
        corr[m] = np.clip(oracle_oof[m] * ratio, 0, None)
    print(f'  {nbins}-bin pred-level mult calib: OOF={mae(corr):.5f} ({mae(corr)-base_oof:+.6f})')

# === Part 2: xgb_monotone blending ===
print('\n=== Part 2: xgb_monotone oracle_seq file blending ===')
oracle_dir = 'results/oracle_seq/'
# Find xgb_monotone
mono_test = os.path.join(oracle_dir, 'test_C_xgb_monotone.npy')
mono_oof = os.path.join(oracle_dir, 'oof_seqC_xgb_monotone.npy')
if os.path.exists(mono_test):
    mt = np.load(mono_test)
    print(f'test_C_xgb_monotone: test_mean={mt.mean():.3f}  unseen={mt[unseen_mask].mean():.3f}  seen={mt[seen_mask].mean():.3f}')
    if os.path.exists(mono_oof):
        mo = np.load(mono_oof)
        print(f'oof_seqC_xgb_monotone: train_mean={mo.mean():.3f}  OOF={mae(np.clip(mo,0,None)):.5f}')
        for w in [0.05, 0.10, 0.15, 0.20]:
            b_oof = np.clip((1-w)*best_base_oof + w*np.clip(mo,0,None), 0, None)
            b_test = np.clip((1-w)*best_base_test + w*np.clip(mt,0,None), 0, None)
            marker = '*' if mae(b_oof) < base_oof else ''
            print(f'  best_base+mono w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')
    else:
        print(f'OOF file not found: {mono_oof}')
else:
    print(f'Test file not found: {mono_test}')

# List all OOF files and find which ones are available
print('\n=== Available OOF oracle_seq files ===')
oof_files = sorted(glob.glob(os.path.join(oracle_dir, 'oof_*.npy')))
print(f'Found {len(oof_files)} OOF files')
oof_info = []
for fp in oof_files:
    arr = np.load(fp)
    if arr.shape[0] != len(train_raw): continue
    fn = os.path.basename(fp)
    oof_mae = mae(np.clip(arr, 0, None))
    # Find test counterpart
    test_fn = fn.replace('oof_seqC_', 'test_C_')
    test_path = os.path.join(oracle_dir, test_fn)
    if os.path.exists(test_path):
        t_arr = np.load(test_path)
        t_unseen = t_arr[unseen_mask].mean()
        t_seen = t_arr[seen_mask].mean()
        oof_info.append((fn, oof_mae, t_arr.mean(), t_unseen, t_seen, t_unseen/t_seen))

oof_info.sort(key=lambda x: x[5], reverse=True)  # sort by unseen/seen ratio
print(f'{"filename":35s}  {"train_OOF":>10}  {"test_mean":>10}  {"t_unseen":>9}  {"t_seen":>9}  {"u/s":>7}')
for fn, oof_m, tm, tu, ts, ratio in oof_info[:20]:
    print(f'{fn:35s}  {oof_m:>10.5f}  {tm:>10.3f}  {tu:>9.3f}  {ts:>9.3f}  {ratio:>7.3f}')

# === Part 3: Blend with top OOF-improving oracle_seq files ===
print('\n=== Part 3: Best OOF oracle_seq files blended with best_base ===')
for fn, oof_m, tm, tu, ts, ratio in sorted(oof_info, key=lambda x: x[1])[:10]:  # sort by OOF ASC
    test_fn = fn.replace('oof_seqC_', 'test_C_')
    test_path = os.path.join(oracle_dir, test_fn)
    mo = np.load(os.path.join(oracle_dir, fn))
    mt = np.load(test_path)
    for w in [0.05, 0.10]:
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(mo,0,None), 0, None)
        b_test = np.clip((1-w)*best_base_test + w*np.clip(mt,0,None), 0, None)
        if mae(b_oof) < base_oof:
            print(f'  *{fn[:30]:30s} w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

# === Part 4: Global under-prediction calibration ===
print('\n=== Part 4: Global under-prediction analysis ===')
# Oracle systematically under-predicts: mean OOF = 15.3 but mean y = 18.9?
print(f'Oracle OOF mean = {oracle_oof.mean():.3f}  y_true mean = {y_true.mean():.3f}')
print(f'Oracle OOF clipped mean = {np.clip(oracle_oof,0,None).mean():.3f}')
print(f'Under-prediction gap: {y_true.mean() - np.clip(oracle_oof,0,None).mean():.3f}')
print(f'Test: oracle_test mean = {oracle_test.mean():.3f}')

# What if we add a constant shift?
for delta in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
    corr = np.clip(oracle_oof + delta, 0, None)
    print(f'  +{delta:.1f} constant: OOF={mae(corr):.5f} ({mae(corr)-base_oof:+.6f})')

print()
# Apply constant shift to test
for delta in [1.0, 2.0, 3.0]:
    corr_test = oracle_test + delta
    print(f'  Test +{delta:.1f}: test_mean={corr_test.mean():.3f}')

# === Part 5: Per-layout-bucket calibration (CV validated for real) ===
print('\n=== Part 5: Per-layout-bucket calibration using similar layouts ===')
# Group layouts by pack_utilization and inflow buckets
train_layout_stats = train_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization', 'mean'),
    inflow_mean=('order_inflow_15m', 'mean'),
    y_mean=('avg_delay_minutes_next_30m', 'mean'),
    oracle_mean=('oracle_pred', 'mean'),
    n=('ID', 'count')
).reset_index()
train_layout_stats['ratio'] = train_layout_stats['y_mean'] / train_layout_stats['oracle_mean'].clip(1e-8)
train_layout_stats['bias'] = train_layout_stats['y_mean'] - train_layout_stats['oracle_mean']

# For test layouts: find K nearest training layouts by pack+inflow
test_layout_stats = test_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization', 'mean'),
    inflow_mean=('order_inflow_15m', 'mean'),
    n=('ID', 'count')
).reset_index()

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor

# Fit KNN bias predictor using training layout features
X_tr = train_layout_stats[['pack_mean', 'inflow_mean']].values
y_bias_tr = train_layout_stats['bias'].values
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)

# Leave-one-layout-out CV
knn_corr_oof = oracle_oof.copy()
for lid in train_layout_stats['layout_id'].unique():
    # Hold out this layout
    m_hold = train_layout_stats['layout_id'] != lid
    X_train_cv = X_tr_sc[m_hold]
    y_bias_cv = y_bias_tr[m_hold]
    knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
    knn.fit(X_train_cv, y_bias_cv)
    # Predict bias for held-out layout
    row = train_layout_stats[train_layout_stats['layout_id'] == lid]
    X_test_cv = scaler.transform(row[['pack_mean', 'inflow_mean']].values)
    pred_bias = knn.predict(X_test_cv)[0]
    # Apply to OOF predictions for this layout
    m_rows = train_raw['layout_id'] == lid
    for alpha in [1.0]:  # we'll test alpha separately
        knn_corr_oof[m_rows] = np.clip(oracle_oof[m_rows] + alpha * pred_bias, 0, None)

print(f'KNN layout bias correction (LOOCV): OOF={mae(knn_corr_oof):.5f} ({mae(knn_corr_oof)-base_oof:+.6f})')

# Also test smaller alpha
for alpha in [0.2, 0.3, 0.5, 0.7, 1.0]:
    knn_corr_alpha = oracle_oof.copy()
    for lid in train_layout_stats['layout_id'].unique():
        m_hold = train_layout_stats['layout_id'] != lid
        X_train_cv = X_tr_sc[m_hold]
        y_bias_cv = y_bias_tr[m_hold]
        knn = KNeighborsRegressor(n_neighbors=5, weights='distance')
        knn.fit(X_train_cv, y_bias_cv)
        row = train_layout_stats[train_layout_stats['layout_id'] == lid]
        X_test_cv = scaler.transform(row[['pack_mean', 'inflow_mean']].values)
        pred_bias = knn.predict(X_test_cv)[0]
        m_rows = train_raw['layout_id'] == lid
        knn_corr_alpha[m_rows] = np.clip(oracle_oof[m_rows] + alpha * pred_bias, 0, None)
    print(f'  alpha={alpha:.1f}: OOF={mae(knn_corr_alpha):.5f} ({mae(knn_corr_alpha)-base_oof:+.6f})')

# Apply KNN correction to test (using all training data)
knn_full = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_full.fit(X_tr_sc, y_bias_tr)
X_test_all = scaler.transform(test_layout_stats[['pack_mean', 'inflow_mean']].values)
test_bias_pred = knn_full.predict(X_test_all)
test_bias_map = dict(zip(test_layout_stats['layout_id'], test_bias_pred))
print(f'\nKNN predicted bias for test layouts:')
print(f'  SEEN: mean={np.mean([test_bias_map[l] for l in test_layout_stats.loc[test_layout_stats["layout_id"].isin(train_layouts),"layout_id"]]):.3f}')
print(f'  UNSEEN: mean={np.mean([test_bias_map[l] for l in test_layout_stats.loc[~test_layout_stats["layout_id"].isin(train_layouts),"layout_id"]]):.3f}')

for alpha in [0.3, 0.5, 1.0]:
    corr = oracle_test.copy()
    for lid, bias in test_bias_map.items():
        m = test_raw['layout_id'] == lid
        corr[m] = np.clip(oracle_test[m] + alpha*bias, 0, None)
    print(f'  KNN test alpha={alpha:.1f}: test_mean={corr.mean():.3f} ({corr.mean()-oracle_test.mean():+.4f})  unseen={corr[unseen_mask].mean():.3f}  seen={corr[seen_mask].mean():.3f}')

# === Part 6: Summary of best submissions ===
print('\n=== Part 6: Best submission summary ===')
sub = pd.read_csv('sample_submission.csv')
candidates = []

# A: oracle_NEW
candidates.append(('oracle_NEW', oracle_oof, oracle_test, base_oof))

# B: best_base
candidates.append(('best_base_m34_rw_wf072_cb12', best_base_oof, best_base_test, best_base_v))

# C: rw_wf070 (from check_fold2_layouts)
rw70_oof, rw70_test = make_pred(0.0, -0.04, -0.02, 0.70, 0.0)
candidates.append(('rw_wf070', rw70_oof, rw70_test, mae(rw70_oof)))

# D: best_base + KNN test correction
for alpha in [0.3, 0.5]:
    corr_knn = oracle_test.copy()
    # Use best_base as base
    corr_knn = best_base_test.copy()
    for lid, bias in test_bias_map.items():
        m = test_raw['layout_id'] == lid
        corr_knn[m] = np.clip(best_base_test[m] + alpha*bias, 0, None)
    candidates.append((f'best_base_KNN_a{alpha:.1f}', best_base_oof, corr_knn, best_base_v))

# Also look at KNN on oracle_oof for OOF score
for alpha in [0.3, 0.5]:
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
    # For test
    corr_knn_t = best_base_test.copy()
    for lid, bias in test_bias_map.items():
        m = test_raw['layout_id'] == lid
        corr_knn_t[m] = np.clip(best_base_test[m] + alpha*bias, 0, None)
    # Use KNN oof to get proper OOF score
    knn_oof_mae = mae(knn_c)
    candidates.append((f'oracle_KNN_a{alpha:.1f}', knn_c, corr_knn_t, knn_oof_mae))

print(f'{"config":40s}  {"OOF":>9}  {"ΔOOF":>8}  {"test_mean":>10}  {"unseen":>9}  {"seen":>9}')
for name, oof_arr, test_arr, oof_v in candidates:
    d_oof = oof_v - base_oof
    print(f'  {name:40s}  {oof_v:.5f}  {d_oof:+.6f}  {test_arr.mean():.3f}  {test_arr[unseen_mask].mean():.3f}  {test_arr[seen_mask].mean():.3f}')

# Save the KNN-corrected submissions
for name, oof_arr, test_arr, oof_v in candidates[3:]:
    if 'KNN' in name:
        sub['avg_delay_minutes_next_30m'] = test_arr
        fname = f'submission_{name.replace("/","_").replace("+","_")}_OOF{oof_v:.5f}.csv'
        sub.to_csv(fname, index=False)
        print(f'Saved: {fname}')

print('\nDone.')
