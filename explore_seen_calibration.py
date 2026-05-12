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
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'best_base (m34_rw_wf072+cb12): OOF={best_base_v:.5f}  test_mean={best_base_test.mean():.3f}')

# === Part 1: High-wf (pure FIXED) exploration ===
print('\n=== Part 1: High-wf exploration (reduce oracle_seq dependence) ===')
print(f'{"config":35s}  {"OOF":>9}  {"ΔOOF":>9}  {"test_mean":>10}  {"Δtest":>7}')
for wf in [0.64, 0.68, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90, 1.00]:
    oo, ot = make_pred(0.0, 0.0, 0.0, wf, 0.0)
    d_oof = mae(oo)-base_oof
    print(f'  orig_wf{wf:.2f}:                     {mae(oo):.5f}  {d_oof:+.6f}  {ot.mean():.3f}  {ot.mean()-oracle_test.mean():+.4f}')

print()
for wf in [0.72, 0.75, 0.80, 0.85, 0.90, 1.00]:
    oo, ot = make_pred(0.25, -0.04, -0.02, wf, 0.12)
    d_oof = mae(oo)-base_oof
    print(f'  m34_rw_wf{wf:.2f}_cb12:              {mae(oo):.5f}  {d_oof:+.6f}  {ot.mean():.3f}  {ot.mean()-oracle_test.mean():+.4f}')

# === Part 2: Component mean contributions ===
print('\n=== Part 2: Component test_mean analysis ===')
fx_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
print(f'FIXED (orig weights): test_mean={fx_test.mean():.3f}')
fx_rw_test = wm_best*(0.75*mega33_test+0.25*mega34_test) + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test
print(f'FIXED_rw (best): test_mean={fx_rw_test.mean():.3f}')
print(f'oracle_seq xgb: test_mean={xgb_t.mean():.3f}')
print(f'oracle_seq lv2: test_mean={lv2_t.mean():.3f}')
print(f'oracle_seq rem: test_mean={rem_t.mean():.3f}')
print(f'cb_test: test_mean={cb_test.mean():.3f}')
print(f'rank_test: test_mean={rank_test.mean():.3f}')
print(f'mega33_test: test_mean={mega33_test.mean():.3f}')
print(f'mega34_test: test_mean={mega34_test.mean():.3f}')
print()
print(f'At wf=1.0 (pure FIXED_rw + cb12): test_mean = {(0.88*fx_rw_test + 0.12*cb_test).mean():.3f}')
print(f'At wf=0.9: test_mean estimate = {(0.9*fx_rw_test + 0.028*xgb_t + 0.038*lv2_t + 0.019*rem_t + 0.12*cb_test).mean():.3f}')

# === Part 3: SEEN layout calibration ===
print('\n=== Part 3: SEEN layout per-layout bias calibration ===')
train_layouts = set(train_raw['layout_id'].unique())
test_seen_mask = test_raw['layout_id'].isin(train_layouts)
test_unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
print(f'SEEN test rows: {test_seen_mask.sum()}  UNSEEN test rows: {test_unseen_mask.sum()}')

# For SEEN layouts: compute per-layout OOF bias in training
train_raw['oracle_pred'] = oracle_oof
train_raw['resid'] = y_true - np.clip(oracle_oof, 0, None)

seen_layout_bias = {}
for lid in test_raw.loc[test_seen_mask, 'layout_id'].unique():
    m = train_raw['layout_id'] == lid
    if m.sum() < 5: continue
    bias = train_raw.loc[m, 'resid'].mean()
    seen_layout_bias[lid] = bias

biases = np.array(list(seen_layout_bias.values()))
print(f'\nSEEN layout training bias statistics:')
print(f'  mean={biases.mean():.3f}  std={biases.std():.3f}  min={biases.min():.3f}  max={biases.max():.3f}')
print(f'  % positive: {(biases>0).mean()*100:.1f}%  % >2: {(biases>2).mean()*100:.1f}%')

# Apply per-layout bias correction to SEEN test rows
for alpha in [0.2, 0.3, 0.5, 0.7, 1.0]:
    corr = oracle_test.copy()
    for lid, bias in seen_layout_bias.items():
        m = test_raw['layout_id'] == lid
        corr[m] = np.clip(oracle_test[m] + alpha*bias, 0, None)
    seen_change = corr[test_seen_mask].mean() - oracle_test[test_seen_mask].mean()
    print(f'  alpha={alpha:.1f}: test_mean={corr.mean():.3f} ({corr.mean()-oracle_test.mean():+.4f})  seen_change={seen_change:+.4f}  unseen_unchanged={corr[test_unseen_mask].mean():.3f}')

# Cross-validate the per-layout bias correction using GroupKFold
print('\n=== CV validation of per-layout bias correction ===')
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
layout_corr_oof = oracle_oof.copy()

for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    tr_data = train_raw.iloc[tr_idx]
    val_data = train_raw.iloc[val_idx]
    # Compute per-layout bias from training fold
    fold_biases = tr_data.groupby('layout_id')['resid'].mean().to_dict()
    # Apply to val fold
    for lid, bias in fold_biases.items():
        m = val_data['layout_id'] == lid
        if not m.any(): continue
        val_idx_lid = val_idx[m.values]
        layout_corr_oof[val_idx_lid] = np.clip(oracle_oof[val_idx_lid] + bias, 0, None)

print(f'Layout bias correction (alpha=1.0): OOF={mae(layout_corr_oof):.5f} ({mae(layout_corr_oof)-base_oof:+.6f})')

# Try partial alpha
for alpha in [0.2, 0.3, 0.5, 0.7, 1.0]:
    layout_corr_oof_a = oracle_oof.copy()
    for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        tr_data = train_raw.iloc[tr_idx]
        val_data = train_raw.iloc[val_idx]
        fold_biases = tr_data.groupby('layout_id')['resid'].mean().to_dict()
        for lid, bias in fold_biases.items():
            m = val_data['layout_id'] == lid
            if not m.any(): continue
            val_idx_lid = val_idx[m.values]
            layout_corr_oof_a[val_idx_lid] = np.clip(oracle_oof[val_idx_lid] + alpha*bias, 0, None)
    print(f'  alpha={alpha:.1f}: OOF={mae(layout_corr_oof_a):.5f} ({mae(layout_corr_oof_a)-base_oof:+.6f})')

# === Part 4: rank_adj component deep dive ===
print('\n=== Part 4: rank_adj component analysis ===')
print(f'rank_oof: mean={rank_oof.mean():.3f}  std={rank_oof.std():.3f}  OOF={mae(rank_oof):.5f}')
print(f'rank_test: mean={rank_test.mean():.3f}  std={rank_test.std():.3f}')
print(f'rank_test[unseen]: mean={rank_test[test_unseen_mask].mean():.3f}')
print(f'rank_test[seen]: mean={rank_test[test_seen_mask].mean():.3f}')

# Check if adding more rank_adj weight helps
for dr_rank in [0.0, 0.02, 0.04, 0.06, 0.08]:
    # Increase rank_adj weight by shifting from mega33
    wm_r = fw['mega33'] - dr_rank - best_dr2 - best_dr3
    wr = fw['rank_adj'] + dr_rank
    w2_r = fw['iter_r2'] + best_dr2
    w3_r = fw['iter_r3'] + best_dr3
    m_bl = 0.75*mega33_oof + 0.25*mega34_oof
    m_bl_t = 0.75*mega33_test + 0.25*mega34_test
    wf = 0.72; w_rem = 1-wf; wxgb = 0.12*w_rem/0.36; wlv2 = 0.16*w_rem/0.36; wrem = 0.08*w_rem/0.36
    fx_rw = wm_r*m_bl + wr*rank_oof + fw['iter_r1']*r1_oof + w2_r*r2_oof + w3_r*r3_oof
    fx_rw_t = wm_r*m_bl_t + wr*rank_test + fw['iter_r1']*r1_test + w2_r*r2_test + w3_r*r3_test
    oo = np.clip((1-0.12)*(wf*fx_rw + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o) + 0.12*cb_oof, 0, None)
    ot = np.clip((1-0.12)*(wf*fx_rw_t + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t) + 0.12*cb_test, 0, None)
    marker = '*' if mae(oo) < base_oof else ''
    print(f'  dr_rank={dr_rank:.2f}: OOF={mae(oo):.5f} ({mae(oo)-base_oof:+.6f})  test={ot.mean():.3f} ({ot.mean()-oracle_test.mean():+.4f}) {marker}')

# === Part 5: Scenario-level analysis — do certain scenario_ids have systematic bias? ===
print('\n=== Part 5: Scenario-level bias analysis ===')
train_raw['oracle_pred'] = oracle_oof
train_raw['resid'] = y_true - np.clip(oracle_oof, 0, None)

# Check if scenario_id is numeric/ordinal and correlates with residual
if 'scenario_id' in train_raw.columns:
    scen_stats = train_raw.groupby('scenario_id').agg(
        n=('ID','count'),
        y_mean=('avg_delay_minutes_next_30m','mean'),
        pred_mean=('oracle_pred','mean'),
        resid_mean=('resid','mean'),
        inflow_mean=('order_inflow_15m','mean')
    ).reset_index()
    scen_stats = scen_stats.sort_values('resid_mean')
    print(f'Number of unique scenario_ids: {len(scen_stats)}')
    print(f'Scenario residual: min={scen_stats["resid_mean"].min():.3f}  max={scen_stats["resid_mean"].max():.3f}  std={scen_stats["resid_mean"].std():.3f}')
    print(f'\nTop 10 hardest scenarios (highest resid):')
    for _, row in scen_stats.tail(10).iterrows():
        print(f'  {str(row["scenario_id"]):25s}: n={int(row["n"]):5d}  y={row["y_mean"]:.2f}  pred={row["pred_mean"]:.2f}  resid={row["resid_mean"]:+.2f}  inflow={row["inflow_mean"]:.1f}')
    print(f'\nTop 10 easiest scenarios (lowest resid):')
    for _, row in scen_stats.head(10).iterrows():
        print(f'  {str(row["scenario_id"]):25s}: n={int(row["n"]):5d}  y={row["y_mean"]:.2f}  pred={row["pred_mean"]:.2f}  resid={row["resid_mean"]:+.2f}  inflow={row["inflow_mean"]:.1f}')

    # Check test scenario distribution
    test_scen = set(test_raw['scenario_id'].unique())
    train_scen = set(train_raw['scenario_id'].unique())
    unseen_scen = test_scen - train_scen
    print(f'\nTest scenarios total: {len(test_scen)}  Unseen: {len(unseen_scen)}')

    # For each test scenario, are bias patterns consistent?
    common_scen = test_scen & train_scen
    print(f'Common scenarios: {len(common_scen)}')
    if common_scen:
        common_biases = scen_stats[scen_stats['scenario_id'].isin(common_scen)]['resid_mean'].values
        print(f'Common scenario biases: mean={common_biases.mean():.3f}  std={common_biases.std():.3f}')

        # Apply scenario-level correction
        scen_bias_map = dict(zip(scen_stats['scenario_id'], scen_stats['resid_mean']))
        for alpha in [0.3, 0.5, 1.0]:
            corr = oracle_test.copy()
            for sid, bias in scen_bias_map.items():
                m = test_raw['scenario_id'] == sid
                if not m.any(): continue
                corr[m] = np.clip(oracle_test[m] + alpha*bias, 0, None)
            print(f'  Scenario bias alpha={alpha:.1f}: test_mean={corr.mean():.3f} ({corr.mean()-oracle_test.mean():+.4f})')

        # CV validation of scenario bias
        scen_corr_oof = oracle_oof.copy()
        for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
            tr_data = train_raw.iloc[tr_idx]
            val_data = train_raw.iloc[val_idx]
            scen_biases_fold = tr_data.groupby('scenario_id')['resid'].mean().to_dict()
            for sid, bias in scen_biases_fold.items():
                m = val_data['scenario_id'] == sid
                if not m.any(): continue
                val_idx_sid = val_idx[m.values]
                scen_corr_oof[val_idx_sid] = np.clip(oracle_oof[val_idx_sid] + bias, 0, None)
        print(f'  Scenario bias CV (alpha=1.0): OOF={mae(scen_corr_oof):.5f} ({mae(scen_corr_oof)-base_oof:+.6f})')

# === Part 6: Pack-level scenario analysis ===
print('\n=== Part 6: High-pack test layout breakdown ===')
test_pack_mean = test_raw.groupby('layout_id')['pack_utilization'].mean()
high_pack_test = test_pack_mean[test_pack_mean > 0.6]
print(f'Test layouts with pack_mean > 0.6: {len(high_pack_test)}')
for lid, pm in high_pack_test.sort_values(ascending=False).items():
    is_seen = lid in train_layouts
    m = test_raw['layout_id'] == lid
    n = m.sum()
    inflow_m = test_raw.loc[m, 'order_inflow_15m'].mean()
    oracle_m = oracle_test[m].mean()
    print(f'  {lid}: pack={pm:.4f}  n={n}  inflow={inflow_m:.1f}  oracle={oracle_m:.2f}  {"SEEN" if is_seen else "UNSEEN"}')

print('\nDone.')
