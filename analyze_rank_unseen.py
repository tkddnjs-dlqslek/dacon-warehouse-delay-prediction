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

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}  unseen={oracle_test[unseen_mask].mean():.3f}  seen={oracle_test[seen_mask].mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}  unseen={best_base_test[unseen_mask].mean():.3f}  seen={best_base_test[seen_mask].mean():.3f}')

# === Part 1: Rank component analysis ===
print('\n=== Part 1: rank_adj deep dive ===')
print(f'rank_oof: mean={rank_oof.mean():.3f}  OOF={mae(rank_oof):.5f}')
print(f'rank_test: mean={rank_test.mean():.3f}  unseen={rank_test[unseen_mask].mean():.3f}  seen={rank_test[seen_mask].mean():.3f}')

# Per-component analysis for unseen vs seen
components = [
    ('mega33_oof/test', mega33_oof, mega33_test),
    ('mega34_oof/test', mega34_oof, mega34_test),
    ('rank_adj', rank_oof, rank_test),
    ('iter_r1', r1_oof, r1_test),
    ('iter_r2', r2_oof, r2_test),
    ('iter_r3', r3_oof, r3_test),
    ('oracle_xgb', xgb_o, xgb_t),
    ('oracle_lv2', lv2_o, lv2_t),
    ('oracle_rem', rem_o, rem_t),
    ('cb', cb_oof, cb_test),
]
print(f'\n{"component":15s}  {"train_mean":>11}  {"test_mean":>10}  {"test_unseen":>12}  {"test_seen":>10}  {"ratio_u/s":>9}')
for name, oo, ot in components:
    train_m = oo.mean()
    test_m = ot.mean()
    unseen_m = ot[unseen_mask].mean()
    seen_m = ot[seen_mask].mean()
    ratio = unseen_m / seen_m if seen_m > 0 else float('inf')
    print(f'{name:15s}  {train_m:>11.3f}  {test_m:>10.3f}  {unseen_m:>12.3f}  {seen_m:>10.3f}  {ratio:>9.3f}')

# === Part 2: Increase rank_adj specifically for unseen test rows ===
print('\n=== Part 2: rank_adj boost for unseen test only (unseen-layout targeted) ===')
for w_rank_extra in [0.02, 0.05, 0.10, 0.15, 0.20]:
    # Add rank at extra weight only to unseen rows
    corr_bb = best_base_test.copy()
    corr_bb[unseen_mask] = np.clip(best_base_test[unseen_mask] + w_rank_extra * rank_test[unseen_mask], 0, None)
    print(f'  w_rank_unseen={w_rank_extra:.2f}: test_mean={corr_bb.mean():.3f} ({corr_bb.mean()-oracle_test.mean():+.4f})  unseen={corr_bb[unseen_mask].mean():.3f} ({corr_bb[unseen_mask].mean()-oracle_test[unseen_mask].mean():+.4f})')

# === Part 3: SEEN train/test prediction gap ===
print('\n=== Part 3: SEEN layout train/test prediction gap ===')
# For each SEEN test layout: compare oracle_oof (training OOF) vs oracle_test
seen_test_layouts = test_raw.loc[seen_mask, 'layout_id'].unique()
train_oof_by_layout = {}
test_pred_by_layout = {}
train_y_by_layout = {}

for lid in seen_test_layouts:
    tr_m = train_raw['layout_id'] == lid
    te_m = test_raw['layout_id'] == lid
    if tr_m.sum() > 0:
        train_oof_by_layout[lid] = oracle_oof[tr_m].mean()
        train_y_by_layout[lid] = y_true[tr_m].mean()
    if te_m.sum() > 0:
        test_pred_by_layout[lid] = oracle_test[te_m.values].mean()

# Compare train OOF vs test prediction for SEEN layouts
train_oof_means = np.array([train_oof_by_layout[l] for l in seen_test_layouts if l in train_oof_by_layout])
test_pred_means = np.array([test_pred_by_layout[l] for l in seen_test_layouts if l in test_pred_by_layout and l in train_oof_by_layout])
train_y_means = np.array([train_y_by_layout[l] for l in seen_test_layouts if l in train_y_by_layout and l in test_pred_by_layout])

print(f'SEEN layouts (n={len(train_oof_means)}):')
print(f'  Training OOF mean: {train_oof_means.mean():.3f}  (model under-predicts → lower than actuals)')
print(f'  Training y mean: {train_y_means.mean():.3f}')
print(f'  Test prediction mean: {test_pred_means.mean():.3f}')
print(f'  Train OOF / Train y ratio: {(train_oof_means/train_y_means).mean():.4f}')
gap = train_y_means.mean() - train_oof_means.mean()
print(f'  Training OOF gap (y - pred): {gap:.3f}')
print(f'  Train y vs test pred gap: {train_y_means.mean() - test_pred_means.mean():.3f}')

# === Part 4: Isotonic regression for inflow correction ===
print('\n=== Part 4: Isotonic regression for inflow correction ===')
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold

valid_mask = train_raw['order_inflow_15m'].notna()
inflow_valid = train_raw.loc[valid_mask, 'order_inflow_15m'].values
oracle_valid = oracle_oof[valid_mask]
y_valid = y_true[valid_mask]
resid_valid = y_valid - np.clip(oracle_valid, 0, None)

# Fit isotonic regression on inflow → residual
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(inflow_valid, resid_valid)
resid_predicted = iso.predict(inflow_valid)

# Check if this correction helps on training
corr_iso_train = np.clip(oracle_oof.copy(), 0, None)
inflow_all = train_raw['order_inflow_15m'].fillna(0).values
corr_iso_train = np.clip(oracle_oof + iso.predict(inflow_all), 0, None)
print(f'Isotonic correction (all rows): OOF={mae(corr_iso_train):.5f} ({mae(corr_iso_train)-base_oof:+.6f})')

# Apply only to unseen test
test_inflow_all = test_raw['order_inflow_15m'].fillna(0).values
iso_test_correction = iso.predict(test_inflow_all)
for alpha in [0.2, 0.3, 0.5]:
    corr_iso = best_base_test.copy()
    corr_iso[unseen_mask] = np.clip(best_base_test[unseen_mask] + alpha * iso_test_correction[unseen_mask], 0, None)
    print(f'  Isotonic alpha={alpha:.1f} unseen-only: test_mean={corr_iso.mean():.3f} ({corr_iso.mean()-oracle_test.mean():+.4f})')

# === Part 5: All available oracle_seq files ===
print('\n=== Part 5: All oracle_seq files — unseen test comparison ===')
oracle_dir = 'results/oracle_seq/'
all_files = sorted(glob.glob(os.path.join(oracle_dir, '*.npy')))
print(f'Found {len(all_files)} oracle_seq files')
file_info = []
for fp in all_files:
    fn = os.path.basename(fp)
    arr = np.load(fp)
    if arr.shape[0] == len(train_raw):  # OOF file
        pass
    elif arr.shape[0] == len(test_raw):  # test file
        unseen_m = arr[unseen_mask].mean()
        seen_m = arr[seen_mask].mean()
        file_info.append((fn, arr.mean(), unseen_m, seen_m, unseen_m/seen_m if seen_m>0 else 0))

file_info.sort(key=lambda x: -x[4])  # sort by unseen/seen ratio
print(f'{"filename":40s}  {"test_mean":>10}  {"unseen":>9}  {"seen":>9}  {"u/s_ratio":>9}')
for fn, tm, um, sm, ratio in file_info[:20]:
    print(f'  {fn:40s}  {tm:>10.3f}  {um:>9.3f}  {sm:>9.3f}  {ratio:>9.3f}')

# === Part 6: Find oracle_seq file with highest unseen/seen ratio ===
print('\n=== Part 6: Best unseen-ratio oracle_seq file blended with best_base ===')
if file_info:
    best_ratio_file = file_info[0][0]
    # Match with OOF file
    oof_cand = best_ratio_file.replace('test_', 'oof_')
    print(f'Best ratio file: {best_ratio_file} (ratio={file_info[0][4]:.3f})')
    # Try to find the OOF counterpart
    for fn_oof, tm_o, um_o, sm_o, ratio_o in file_info:
        if 'oof' in fn_oof.lower():
            print(f'Best OOF file by ratio: {fn_oof} (u/s={ratio_o:.3f}  unseen={um_o:.3f})')
            break

    # For top 5 high-ratio test files, try blending
    for fn, tm, um, sm, ratio in file_info[:10]:
        test_arr = np.load(os.path.join(oracle_dir, fn))
        if test_arr.shape[0] != len(test_raw): continue
        # Find OOF counterpart
        oof_name = fn.replace('test_', 'oof_')
        oof_path = os.path.join(oracle_dir, oof_name)
        if not os.path.exists(oof_path): continue
        oof_arr = np.load(oof_path)
        if oof_arr.shape[0] != len(train_raw): continue
        # Try blending OOF/test with best_base
        for w in [0.05, 0.10]:
            b_oof = np.clip((1-w)*best_base_oof + w*np.clip(oof_arr,0,None), 0, None)
            b_test = np.clip((1-w)*best_base_test + w*np.clip(test_arr,0,None), 0, None)
            marker = '*' if mae(b_oof) < base_oof else ''
            print(f'  {fn[:30]:30s} w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')

# === Part 7: Generate submission combining SEEN calibration + UNSEEN inflow correction ===
print('\n=== Part 7: Combined SEEN calibration + UNSEEN inflow correction ===')
# Compute per-SEEN-layout bias from training OOF
seen_layout_bias = {}
train_raw['oracle_pred_full'] = oracle_oof
train_raw['resid_full'] = y_true - np.clip(oracle_oof, 0, None)
for lid in seen_test_layouts:
    m = train_raw['layout_id'] == lid
    if m.sum() > 0:
        seen_layout_bias[lid] = train_raw.loc[m, 'resid_full'].mean()

# Inflow bins from training
valid_mask2 = train_raw['order_inflow_15m'].notna() & train_raw['pack_utilization'].notna()
inflow_v2 = train_raw.loc[valid_mask2, 'order_inflow_15m'].values
resid_v2 = y_true[valid_mask2] - np.clip(oracle_oof[valid_mask2], 0, None)
q_vals2 = np.percentile(inflow_v2, [25, 50, 75, 90, 95])
q25t, q50t, q75t, q90t, q95t = q_vals2
bins2 = [(-np.inf, q25t), (q25t, q50t), (q50t, q75t), (q75t, q90t), (q90t, q95t), (q95t, np.inf)]
bin_resids2 = []
for lo, hi in bins2:
    m = (inflow_v2 >= lo) & (inflow_v2 < hi)
    bin_resids2.append(resid_v2[m].mean() if m.sum() > 0 else 0.0)

test_inflow_all2 = test_raw['order_inflow_15m'].fillna(0).values

sub = pd.read_csv('sample_submission.csv')
# Best_base as starting point
for alpha_seen, alpha_unseen in [(0.2, 0.3), (0.3, 0.3), (0.2, 0.5), (0.3, 0.5)]:
    corr = best_base_test.copy()
    # Apply SEEN calibration
    for lid, bias in seen_layout_bias.items():
        m = test_raw['layout_id'] == lid
        corr[m] = np.clip(corr[m] + alpha_seen * bias, 0, None)
    # Apply UNSEEN inflow correction
    for (lo, hi), r in zip(bins2, bin_resids2):
        m = unseen_mask & (test_inflow_all2 >= lo) & (test_inflow_all2 < hi)
        corr[m] = np.clip(best_base_test[m] + alpha_unseen * r, 0, None)
    print(f'  seen_a={alpha_seen:.1f}+unseen_a={alpha_unseen:.1f}: test_mean={corr.mean():.3f} ({corr.mean()-oracle_test.mean():+.4f})  seen={corr[seen_mask].mean():.3f}  unseen={corr[unseen_mask].mean():.3f}')

# Save a few
for alpha_seen, alpha_unseen in [(0.2, 0.3), (0.3, 0.5)]:
    corr = best_base_test.copy()
    for lid, bias in seen_layout_bias.items():
        m = test_raw['layout_id'] == lid
        corr[m] = np.clip(corr[m] + alpha_seen * bias, 0, None)
    for (lo, hi), r in zip(bins2, bin_resids2):
        m = unseen_mask & (test_inflow_all2 >= lo) & (test_inflow_all2 < hi)
        corr[m] = np.clip(best_base_test[m] + alpha_unseen * r, 0, None)
    sub['avg_delay_minutes_next_30m'] = corr
    fname = f'submission_combo_seen{alpha_seen:.1f}_unseen{alpha_unseen:.1f}_OOF{best_base_v:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'Saved: {fname}  test_mean={corr.mean():.3f}')

print('\nDone.')
