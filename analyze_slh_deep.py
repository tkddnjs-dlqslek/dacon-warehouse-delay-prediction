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
xgb_comb_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgb_comb_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]

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
val_indices = [(fi, np.sort(vi)) for fi, (_, vi) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups))]

# Best blends
bb_slh08 = np.clip(0.92*best_base_oof + 0.08*np.clip(slh_o,0,None), 0, None)
bb_slh08_t = np.clip(0.92*best_base_test + 0.08*np.clip(slh_t,0,None), 0, None)
four_way_oof = np.clip(0.74*best_base_oof + 0.08*np.clip(slh_o,0,None) + 0.10*np.clip(xgb_comb_o,0,None) + 0.08*np.clip(mono_o,0,None), 0, None)
four_way_test = np.clip(0.74*best_base_test + 0.08*np.clip(slh_t,0,None) + 0.10*np.clip(xgb_comb_t,0,None) + 0.08*np.clip(mono_t,0,None), 0, None)

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')
print(f'bb+slh_w08: OOF={mae(bb_slh08):.5f}  test={bb_slh08_t.mean():.3f}')
print(f'4way_best:  OOF={mae(four_way_oof):.5f}  test={four_way_test.mean():.3f}')

# === Part 1: SLH residual analysis ===
print('\n=== Part 1: SLH residual analysis ===')
slh_resid = y_true - np.clip(slh_o, 0, None)
oracle_resid = y_true - np.clip(oracle_oof, 0, None)
train_raw['slh_pred'] = np.clip(slh_o, 0, None)
train_raw['oracle_pred'] = oracle_oof
train_raw['slh_resid'] = slh_resid
train_raw['oracle_resid'] = oracle_resid
train_raw['inflow'] = train_raw['order_inflow_15m'].fillna(0)
train_raw['pack'] = train_raw['pack_utilization'].fillna(0)

print(f'SLH: mean_pred={np.clip(slh_o,0,None).mean():.3f}  oracle: mean_pred={np.clip(oracle_oof,0,None).mean():.3f}  y_true: mean={y_true.mean():.3f}')
print(f'SLH residual: mean={slh_resid.mean():.3f}  std={slh_resid.std():.3f}')
print(f'Oracle residual: mean={oracle_resid.mean():.3f}  std={oracle_resid.std():.3f}')

# By inflow bins
q_vals = np.percentile(train_raw.loc[train_raw['order_inflow_15m'].notna(), 'inflow'].values, [25, 50, 75, 90, 95])
print(f'\nInflow bin analysis:')
print(f'{"bin":12s}  {"n":>6}  {"y":>7}  {"oracle":>8}  {"slh":>8}  {"oracle_r":>9}  {"slh_r":>7}  {"diff":>7}')
for label, lo, hi in [('<p25', -np.inf, q_vals[0]), ('p25-50', q_vals[0], q_vals[1]),
                       ('p50-75', q_vals[1], q_vals[2]), ('p75-90', q_vals[2], q_vals[3]),
                       ('>p90', q_vals[3], np.inf)]:
    m = (train_raw['inflow'] >= lo) & (train_raw['inflow'] < hi)
    sub = train_raw[m]
    print(f'  {label:10s}: {m.sum():>6d}  {sub["avg_delay_minutes_next_30m"].mean():>7.2f}  {sub["oracle_pred"].mean():>8.2f}  {sub["slh_pred"].mean():>8.2f}  {sub["oracle_resid"].mean():>+9.2f}  {sub["slh_resid"].mean():>+7.2f}  {sub["slh_resid"].mean()-sub["oracle_resid"].mean():>+7.2f}')

# By pack bins
print(f'\nPack bin analysis:')
pack_q = np.percentile(train_raw['pack'].values, [25, 50, 75, 90])
for label, lo, hi in [('<p25', -np.inf, pack_q[0]), ('p25-50', pack_q[0], pack_q[1]),
                       ('p50-75', pack_q[1], pack_q[2]), ('p75-90', pack_q[2], pack_q[3]),
                       ('>p90', pack_q[3], np.inf)]:
    m = (train_raw['pack'] >= lo) & (train_raw['pack'] < hi)
    sub = train_raw[m]
    print(f'  {label:10s}: {m.sum():>6d}  y={sub["avg_delay_minutes_next_30m"].mean():>7.2f}  oracle_r={sub["oracle_resid"].mean():>+7.2f}  slh_r={sub["slh_resid"].mean():>+7.2f}  diff={sub["slh_resid"].mean()-sub["oracle_resid"].mean():>+7.2f}')

# === Part 2: Where does slh improve oracle? ===
print('\n=== Part 2: SLH improvement over oracle (per fold) ===')
for fi, vi in val_indices:
    oracle_fv = float(np.mean(np.abs(np.clip(oracle_oof[vi],0,None) - y_true[vi])))
    slh_fv = float(np.mean(np.abs(np.clip(slh_o[vi],0,None) - y_true[vi])))
    bb_slh_fv = float(np.mean(np.abs(bb_slh08[vi] - y_true[vi])))
    pack_mean = train_raw.iloc[vi]['pack_utilization'].mean()
    inflow_mean = train_raw.iloc[vi]['inflow'].mean()
    n = len(vi)
    print(f'Fold {fi+1}: n={n}  pack={pack_mean:.4f}  inflow={inflow_mean:.1f}  oracle={oracle_fv:.5f}  slh={slh_fv:.5f} ({slh_fv-oracle_fv:+.5f})  bb+slh={bb_slh_fv:.5f} ({bb_slh_fv-oracle_fv:+.5f})')

# === Part 3: SLH helps in high-y scenarios ===
print('\n=== Part 3: SLH performance by y_true quantile ===')
q_y = np.percentile(y_true, [25, 50, 75, 90, 95])
for label, lo, hi in [('<p25', -np.inf, q_y[0]), ('p25-50', q_y[0], q_y[1]),
                       ('p50-75', q_y[1], q_y[2]), ('p75-90', q_y[2], q_y[3]),
                       ('p90-95', q_y[3], q_y[4]), ('>p95', q_y[4], np.inf)]:
    m = (y_true >= lo) & (y_true < hi)
    oracle_mae_sub = float(np.mean(np.abs(np.clip(oracle_oof[m],0,None) - y_true[m])))
    slh_mae_sub = float(np.mean(np.abs(np.clip(slh_o[m],0,None) - y_true[m])))
    bb_slh_mae = float(np.mean(np.abs(bb_slh08[m] - y_true[m])))
    print(f'  {label:8s}: n={m.sum():6d}  y={y_true[m].mean():.2f}  oracle_mae={oracle_mae_sub:.3f}  slh_mae={slh_mae_sub:.3f} ({slh_mae_sub-oracle_mae_sub:+.3f})  bb+slh={bb_slh_mae:.3f} ({bb_slh_mae-oracle_mae_sub:+.3f})')

# === Part 4: Combine best blend + unseen inflow correction ===
print('\n=== Part 4: Combine best blend + unseen inflow correction ===')
valid_mask = train_raw['order_inflow_15m'].notna() & train_raw['pack_utilization'].notna()
inflow_v = train_raw.loc[valid_mask, 'order_inflow_15m'].values
oracle_v = oracle_oof[valid_mask]
y_v = y_true[valid_mask]
resid_v = y_v - np.clip(oracle_v, 0, None)
q_vals_bins = np.percentile(inflow_v, [25, 50, 75, 90, 95])
bins = [(-np.inf, q_vals_bins[0]), (q_vals_bins[0], q_vals_bins[1]), (q_vals_bins[1], q_vals_bins[2]),
        (q_vals_bins[2], q_vals_bins[3]), (q_vals_bins[3], q_vals_bins[4]), (q_vals_bins[4], np.inf)]
bin_resids = [resid_v[(inflow_v >= lo) & (inflow_v < hi)].mean() if ((inflow_v >= lo) & (inflow_v < hi)).sum() > 0 else 0.0 for lo, hi in bins]

test_inflow = test_raw['order_inflow_15m'].fillna(0).values

# Apply inflow correction to unseen test rows on top of best blend
for base_name, base_test in [('oracle', oracle_test), ('bb_slh08', bb_slh08_t), ('4way', four_way_test)]:
    for alpha in [0.2, 0.3, 0.5]:
        corr = base_test.copy()
        for (lo, hi), r in zip(bins, bin_resids):
            m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
            corr[m] = np.clip(base_test[m] + alpha*r, 0, None)
        print(f'  {base_name}+inflow_a{alpha:.1f}: test_mean={corr.mean():.3f}  unseen={corr[unseen_mask].mean():.3f}')

# === Part 5: Check remaining cascade variants ===
print('\n=== Part 5: Remaining cascade and specialist files ===')
spec_dirs = ['results/cascade/', 'results/cluster_spec/', 'results/bucket_specialist/', 'results/pack_spec/']
all_spec = []
for d in spec_dirs:
    if not os.path.exists(d): continue
    for fp in sorted(glob.glob(os.path.join(d, '*_oof.npy'))):
        fn = os.path.basename(fp)
        test_fp = fp.replace('_oof.', '_test.')
        if not os.path.exists(test_fp): continue
        oo = np.load(fp)
        ot = np.load(test_fp)
        if oo.shape[0] not in [len(train_raw), len(train_ls)]: continue
        if ot.shape[0] not in [len(test_raw), len(test_ls)]: continue
        if oo.shape[0] == len(train_ls): oo = oo[id2]
        if ot.shape[0] == len(test_ls): ot = ot[te_id2]
        oof_v = mae(np.clip(oo,0,None))
        if oof_v > 20: continue  # skip clearly wrong files
        # Try blend with 4way as base
        for w in np.arange(0.02, 0.15, 0.01):
            b_oof = np.clip((1-w)*four_way_oof + w*np.clip(oo,0,None), 0, None)
            b_test = np.clip((1-w)*four_way_test + w*np.clip(ot,0,None), 0, None)
            if mae(b_oof) < mae(four_way_oof) - 0.001:
                print(f'  * {d}{fn:35s} w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')
                break

# Also check layer2
layer2_oof_path = 'results/layer2/layer2_oof.npy'
layer2_test_path = 'results/layer2/layer2_test.npy'
if os.path.exists(layer2_oof_path):
    l2_o = np.load(layer2_oof_path)
    l2_t = np.load(layer2_test_path)
    print(f'\nlayer2: shape_oof={l2_o.shape}  shape_test={l2_t.shape}')
    if l2_o.shape[0] in [len(train_raw), len(train_ls)] and l2_t.shape[0] in [len(test_raw), len(test_ls)]:
        if l2_o.shape[0] == len(train_ls): l2_o = l2_o[id2]
        if l2_t.shape[0] == len(test_ls): l2_t = l2_t[te_id2]
        print(f'  OOF={mae(np.clip(l2_o,0,None)):.5f}  test={l2_t.mean():.3f}  unseen={l2_t[unseen_mask].mean():.3f}')
        for w in [0.05, 0.10]:
            b_oof = np.clip((1-w)*four_way_oof + w*np.clip(l2_o,0,None), 0, None)
            b_test = np.clip((1-w)*four_way_test + w*np.clip(l2_t,0,None), 0, None)
            print(f'  4way+layer2 w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

# === Part 6: Generate combo submissions (best blend + unseen inflow) ===
print('\n=== Part 6: Save best submissions with inflow combo ===')
sub = pd.read_csv('sample_submission.csv')

# Save bb_slh08 + unseen inflow a=0.3 (conservative)
alpha=0.3
corr = bb_slh08_t.copy()
for (lo, hi), r in zip(bins, bin_resids):
    m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
    corr[m] = np.clip(bb_slh08_t[m] + alpha*r, 0, None)
sub['avg_delay_minutes_next_30m'] = corr
fname = f'submission_bb_slh08_inflow_a{alpha:.1f}_OOF{mae(bb_slh08):.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}  test_mean={corr.mean():.3f}  unseen={corr[unseen_mask].mean():.3f}')

# Save 4way + unseen inflow a=0.3
corr4 = four_way_test.copy()
for (lo, hi), r in zip(bins, bin_resids):
    m = unseen_mask & (test_inflow >= lo) & (test_inflow < hi)
    corr4[m] = np.clip(four_way_test[m] + alpha*r, 0, None)
sub['avg_delay_minutes_next_30m'] = corr4
fname = f'submission_4way_inflow_a{alpha:.1f}_OOF{mae(four_way_oof):.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}  test_mean={corr4.mean():.3f}  unseen={corr4[unseen_mask].mean():.3f}')

print('\n=== FINAL FULL SUMMARY ===')
print(f'oracle_NEW (LB=9.7527): OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:              OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')
print(f'bb+slh_w0.08:           OOF={mae(bb_slh08):.5f}  test={bb_slh08_t.mean():.3f}  unseen={bb_slh08_t[unseen_mask].mean():.3f}')
print(f'4way_best:              OOF={mae(four_way_oof):.5f}  test={four_way_test.mean():.3f}  unseen={four_way_test[unseen_mask].mean():.3f}')

print('\nDone.')
