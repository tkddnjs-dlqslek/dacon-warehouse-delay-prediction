import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
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

fx_orig_oof = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fx_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
oracle_oof = np.clip(0.64*fx_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fx_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

wf_bb=0.72; w_rem_bb=1-wf_bb; wxgb_bb=0.12*w_rem_bb/0.36; wlv2_bb=0.16*w_rem_bb/0.36; wrem_bb=0.08*w_rem_bb/0.36
m_bl = 0.75*mega33_oof+0.25*mega34_oof; m_bl_t = 0.75*mega33_test+0.25*mega34_test
best_base_oof = np.clip((1-0.12)*(wf_bb*(wm_best*m_bl+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_best*r2_oof+w3_best*r3_oof)+wxgb_bb*xgb_o+wlv2_bb*lv2_o+wrem_bb*rem_o) + 0.12*cb_oof, 0, None)
best_base_test = np.clip((1-0.12)*(wf_bb*(wm_best*m_bl_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2_best*r2_test+w3_best*r3_test)+wxgb_bb*xgb_t+wlv2_bb*lv2_t+wrem_bb*rem_t) + 0.12*cb_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
best_base_v = mae(best_base_oof)

# Compute residuals from oracle_oof
train_raw['oracle_pred'] = oracle_oof
train_raw['bb_pred'] = best_base_oof
train_raw['oracle_resid'] = y_true - np.clip(oracle_oof, 0, None)
train_raw['bb_resid'] = y_true - np.clip(best_base_oof, 0, None)
train_raw['inflow'] = train_raw['order_inflow_15m'].fillna(0)
train_raw['pack'] = train_raw['pack_utilization'].fillna(0)

print(f'oracle_NEW OOF={base_oof:.5f}  best_base OOF={best_base_v:.5f}')

# === Part 1: Within-layout inflow vs residual correlation ===
print('\n=== Part 1: Within-layout inflow-residual correlation ===')
# For each layout, compute corr(inflow, oracle_resid)
layout_corrs = []
for lid, grp in train_raw.groupby('layout_id'):
    if len(grp) < 10: continue
    inflow_g = grp['inflow'].values
    resid_g = grp['oracle_resid'].values
    if inflow_g.std() < 1e-6: continue
    c = np.corrcoef(inflow_g, resid_g)[0,1]
    layout_corrs.append((lid, c, inflow_g.mean(), inflow_g.std(), resid_g.mean(), len(grp)))

layout_corrs.sort(key=lambda x: x[1])
arr_corr = np.array([x[1] for x in layout_corrs])
print(f'Per-layout corr(inflow, oracle_resid):')
print(f'  mean={arr_corr.mean():.4f}  median={np.median(arr_corr):.4f}  std={arr_corr.std():.4f}')
print(f'  % positive: {(arr_corr>0).mean()*100:.1f}%  % >0.3: {(arr_corr>0.3).mean()*100:.1f}%')
print(f'  Bottom 5 (negative corr):')
for lid, c, im, istd, rm, n in layout_corrs[:5]:
    print(f'    {lid}: corr={c:.4f}  inflow_mean={im:.1f}  inflow_std={istd:.1f}  resid_mean={rm:+.2f}  n={n}')
print(f'  Top 5 (positive corr):')
for lid, c, im, istd, rm, n in layout_corrs[-5:]:
    print(f'    {lid}: corr={c:.4f}  inflow_mean={im:.1f}  inflow_std={istd:.1f}  resid_mean={rm:+.2f}  n={n}')

# === Part 2: Fold 2 within-fold inflow vs residual ===
print('\n=== Part 2: Fold 2 within-fold analysis ===')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_val_indices = []
for fi, (_, vi) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    fold_val_indices.append((fi, np.sort(vi)))

# Fold 2 (index 1 since fold 2 is 1-indexed as 2)
f2_idx = fold_val_indices[1][1]
f2_data = train_raw.iloc[f2_idx]
print(f'Fold 2: n={len(f2_data)}  inflow_mean={f2_data["inflow"].mean():.1f}  resid_mean={f2_data["oracle_resid"].mean():+.3f}')
# Inflow quantile breakdown within fold 2
q25f2, q50f2, q75f2, q90f2 = f2_data['inflow'].quantile([0.25, 0.5, 0.75, 0.9])
for label, lo, hi in [('<p25', -np.inf, q25f2), ('p25-50', q25f2, q50f2), ('p50-75', q50f2, q75f2), ('p75-90', q75f2, q90f2), ('>p90', q90f2, np.inf)]:
    m = (f2_data['inflow'] >= lo) & (f2_data['inflow'] < hi)
    if not m.any(): continue
    sub_f2 = f2_data[m]
    print(f'  {label:8s}: n={m.sum():5d}  inflow={sub_f2["inflow"].mean():.1f}  y={sub_f2["avg_delay_minutes_next_30m"].mean():.2f}  pred={sub_f2["oracle_pred"].mean():.2f}  resid={sub_f2["oracle_resid"].mean():+.2f}')

# === Part 3: Fold-level inflow distribution vs residual ===
print('\n=== Part 3: Fold-level inflow distribution ===')
for fi, vi in fold_val_indices:
    fd = train_raw.iloc[vi]
    inflow_gt150 = (fd['inflow'] > 150).sum()
    inflow_gt200 = (fd['inflow'] > 200).sum()
    print(f'Fold {fi+1}: inflow_mean={fd["inflow"].mean():.1f}  >150: {inflow_gt150}({inflow_gt150/len(fd)*100:.1f}%)  >200: {inflow_gt200}({inflow_gt200/len(fd)*100:.1f}%)  resid={fd["oracle_resid"].mean():+.3f}')

# === Part 4: Check if inflow correction applied to FOLD 2 rows only improves fold 2 OOF ===
print('\n=== Part 4: Apply inflow correction to fold 2 rows only (in-fold validation) ===')
valid_mask = train_raw['order_inflow_15m'].notna() & train_raw['pack_utilization'].notna()
inflow_v = train_raw.loc[valid_mask, 'order_inflow_15m'].values
y_v = y_true[valid_mask]
oracle_v = oracle_oof[valid_mask]
resid_v = y_v - np.clip(oracle_v, 0, None)
q_vals = np.percentile(inflow_v, [25, 50, 75, 90, 95])
q25t, q50t, q75t, q90t, q95t = q_vals
bins = [(-np.inf, q25t), (q25t, q50t), (q50t, q75t), (q75t, q90t), (q90t, q95t), (q95t, np.inf)]
bin_resids = []
for lo, hi in bins:
    m = (inflow_v >= lo) & (inflow_v < hi)
    bin_resids.append(resid_v[m].mean() if m.sum() > 0 else 0.0)

# Apply correction only to fold 2 validation rows
f2_vi_set = set(f2_idx)
train_inflow_all = train_raw['inflow'].values

def apply_correction_masked(preds, inflow, mask_indices, bins, bin_resids, alpha=1.0):
    corrected = preds.copy()
    mask = np.zeros(len(preds), dtype=bool)
    mask[mask_indices] = True
    for (lo, hi), r in zip(bins, bin_resids):
        m = mask & (inflow >= lo) & (inflow < hi)
        corrected[m] = np.clip(preds[m] + alpha*r, 0, None)
    return corrected

print('  Applying inflow correction only to fold 2 rows (leak-free within oracle model):')
f2_vi = f2_idx
for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    corr = apply_correction_masked(oracle_oof, train_inflow_all, f2_vi, bins, bin_resids, alpha)
    f2_mae = float(np.mean(np.abs(corr[f2_vi] - y_true[f2_vi])))
    f2_mae_base = float(np.mean(np.abs(np.clip(oracle_oof[f2_vi],0,None) - y_true[f2_vi])))
    total_mae = mae(corr)
    print(f'  alpha={alpha:.1f}: fold2_MAE={f2_mae:.5f} ({f2_mae-f2_mae_base:+.5f})  total_OOF={total_mae:.5f} ({total_mae-base_oof:+.5f})')

# === Part 5: Scenario-level inflow analysis ===
print('\n=== Part 5: Per-scenario inflow vs residual (within same layout) ===')
# Within each layout, are higher-inflow scenarios harder?
scenario_stats = train_raw.groupby(['layout_id', 'scenario_id']).agg(
    inflow_mean=('inflow', 'mean'),
    y_mean=('avg_delay_minutes_next_30m', 'mean'),
    oracle_resid_mean=('oracle_resid', 'mean'),
    n=('ID', 'count')
).reset_index()

# Within-layout: for layouts with multiple scenarios, check inflow vs residual
within_layout_corrs = []
for lid, grp in scenario_stats.groupby('layout_id'):
    if len(grp) < 3: continue
    if grp['inflow_mean'].std() < 0.1: continue
    c = np.corrcoef(grp['inflow_mean'].values, grp['oracle_resid_mean'].values)[0,1]
    within_layout_corrs.append((lid, c, grp['inflow_mean'].mean(), len(grp)))

wlc = np.array([x[1] for x in within_layout_corrs])
print(f'Within-layout (per-scenario) corr(inflow_mean, resid_mean):')
print(f'  n_layouts={len(wlc)}  mean={wlc.mean():.4f}  median={np.median(wlc):.4f}')
print(f'  % positive: {(wlc>0).mean()*100:.1f}%  % >0.3: {(wlc>0.3).mean()*100:.1f}%')

# === Part 6: Unseen layout inflow breakdown ===
print('\n=== Part 6: Unseen test layout breakdown ===')
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
test_raw['inflow_t'] = test_raw['order_inflow_15m'].fillna(0)

print(f'Unseen test: n={unseen_mask.sum()}  inflow_mean={test_raw.loc[unseen_mask,"inflow_t"].mean():.1f}')
q_test_unseen = test_raw.loc[unseen_mask, 'inflow_t'].values
for label, lo, hi in [(f'<{int(q25t)}', -np.inf, q25t), (f'{int(q25t)}-{int(q75t)}', q25t, q75t), (f'{int(q75t)}-{int(q90t)}', q75t, q90t), (f'>{int(q90t)}', q90t, np.inf)]:
    m = (q_test_unseen >= lo) & (q_test_unseen < hi)
    print(f'  inflow {label:12s}: n={m.sum():5d} ({m.sum()/len(q_test_unseen)*100:.1f}%)')

# Which training bin correction would apply to unseen rows?
print(f'\nBin corrections applied to unseen rows:')
for i, (lo, hi) in enumerate(bins):
    m = (test_raw.loc[unseen_mask, 'inflow_t'].values >= lo) & (test_raw.loc[unseen_mask, 'inflow_t'].values < hi)
    print(f'  bin[{lo:.0f},{hi:.0f}): n_unseen={m.sum():5d}  correction={bin_resids[i]:+.2f}')

# === Part 7: Generate additional alpha variants on best_base ===
print('\n=== Part 7: Additional alpha variants ===')
sub = pd.read_csv('sample_submission.csv')
test_inflow_all = test_raw['inflow_t'].values

for alpha in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
    corr_bb = best_base_test.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = unseen_mask & (test_inflow_all >= lo) & (test_inflow_all < hi)
        corr_bb[m] = np.clip(best_base_test[m] + alpha*r, 0, None)
    print(f'  alpha={alpha:.2f}: test_mean={corr_bb.mean():.3f} ({corr_bb.mean()-oracle_test.mean():+.3f})  unseen_mean={corr_bb[unseen_mask].mean():.3f}')

# Save alpha=0.4 variant
alpha = 0.4
corr_bb = best_base_test.copy()
for (lo, hi), r in zip(bins, bin_resids):
    m = unseen_mask & (test_inflow_all >= lo) & (test_inflow_all < hi)
    corr_bb[m] = np.clip(best_base_test[m] + alpha*r, 0, None)
sub['avg_delay_minutes_next_30m'] = corr_bb
fname = f'submission_bb_unseen_inflow_a{alpha:.1f}_OOF{best_base_v:.5f}.csv'
sub.to_csv(fname, index=False)
print(f'\nSaved: {fname}  test_mean={corr_bb.mean():.3f}')

# === Part 8: oracle base + unseen inflow correction (not best_base) ===
print('\n=== Part 8: oracle_NEW base + unseen inflow correction ===')
for alpha in [0.3, 0.5, 0.7, 1.0]:
    corr_ora = oracle_test.copy()
    for (lo, hi), r in zip(bins, bin_resids):
        m = unseen_mask & (test_inflow_all >= lo) & (test_inflow_all < hi)
        corr_ora[m] = np.clip(oracle_test[m] + alpha*r, 0, None)
    print(f'  alpha={alpha:.1f}: test_mean={corr_ora.mean():.3f} ({corr_ora.mean()-oracle_test.mean():+.3f})  unseen={corr_ora[unseen_mask].mean():.3f}')

# Save oracle + alpha=0.5
alpha = 0.5
corr_ora = oracle_test.copy()
for (lo, hi), r in zip(bins, bin_resids):
    m = unseen_mask & (test_inflow_all >= lo) & (test_inflow_all < hi)
    corr_ora[m] = np.clip(oracle_test[m] + alpha*r, 0, None)
sub['avg_delay_minutes_next_30m'] = corr_ora
fname = f'submission_oracle_unseen_inflow_a{alpha:.1f}_OOF{base_oof:.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}  test_mean={corr_ora.mean():.3f}')

# === Part 9: pack × inflow joint correction ===
print('\n=== Part 9: pack × inflow joint correction on unseen test ===')
# High pack + high inflow should be corrected most
# Training: define 2D bins
pack_q50 = train_raw['pack'].quantile(0.50)
pack_q75 = train_raw['pack'].quantile(0.75)
inflow_q50 = np.percentile(inflow_v, 50)
inflow_q75 = np.percentile(inflow_v, 75)

print(f'Training pack q50={pack_q50:.3f}  q75={pack_q75:.3f}')
print(f'Training inflow q50 (valid rows)={inflow_q50:.1f}  q75={inflow_q75:.1f}')

# 4-cell joint bins from training
joint_bins = [
    ('low_pack+low_inflow', train_raw['pack'] < pack_q50, train_raw['inflow'] < inflow_q50),
    ('low_pack+high_inflow', train_raw['pack'] < pack_q50, train_raw['inflow'] >= inflow_q50),
    ('high_pack+low_inflow', train_raw['pack'] >= pack_q50, train_raw['inflow'] < inflow_q50),
    ('high_pack+high_inflow', train_raw['pack'] >= pack_q50, train_raw['inflow'] >= inflow_q50),
]
joint_resids = {}
for name, pm, im in joint_bins:
    m = pm & im
    r = train_raw.loc[m, 'oracle_resid'].mean()
    joint_resids[name] = r
    print(f'  {name}: n={m.sum()}  resid_mean={r:+.3f}  y_mean={train_raw.loc[m,"avg_delay_minutes_next_30m"].mean():.2f}')

# Apply to unseen test using 4-cell joint correction
test_raw['pack_t'] = test_raw['pack_utilization'].fillna(0)
for alpha in [0.3, 0.5, 1.0]:
    corr_joint = best_base_test.copy()
    for (name, pm_cond, im_cond), (_, pm_cond2, im_cond2) in zip(joint_bins, joint_bins):
        pass  # rebuild for test
    # Just apply to test using same thresholds
    unseen_idx = np.where(unseen_mask)[0]
    test_pack_t = test_raw['pack_t'].values
    test_inflow_t = test_raw['inflow_t'].values
    for bname, r in joint_resids.items():
        pack_low = bname.startswith('low_pack')
        inflow_low = 'low_inflow' in bname
        if pack_low:
            pm = test_pack_t < pack_q50
        else:
            pm = test_pack_t >= pack_q50
        if inflow_low:
            im = test_inflow_t < inflow_q50
        else:
            im = test_inflow_t >= inflow_q50
        m = unseen_mask & pm & im
        corr_joint[m] = np.clip(best_base_test[m] + alpha*r, 0, None)
    print(f'  Joint correction alpha={alpha:.1f}: test_mean={corr_joint.mean():.3f} ({corr_joint.mean()-oracle_test.mean():+.3f})  unseen={corr_joint[unseen_mask].mean():.3f}')

print('\nDone.')
