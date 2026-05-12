import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
residuals_train = y_true - fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*60)
print("Scenario-Level Residual Analysis")
print("="*60)

# Per-scenario OOF, y_true, residual
train_raw['_oof'] = fw4_oo
train_raw['_resid'] = residuals_train
sc_stats = train_raw.groupby(['layout_id','scenario_id']).agg(
    oof_mean=('_oof','mean'),
    ytrue_mean=('avg_delay_minutes_next_30m','mean'),
    resid_mean=('_resid','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    n=('_oof','count')
).reset_index()

print(f"\n  Training scenarios: {len(sc_stats)}")
print(f"  scenario oof_mean: {sc_stats['oof_mean'].describe()}")

from scipy.stats import pearsonr
r_oof, _ = pearsonr(sc_stats['oof_mean'], sc_stats['resid_mean'])
r_inf, _ = pearsonr(sc_stats['inflow_mean'], sc_stats['resid_mean'])
r_ytrue, _ = pearsonr(sc_stats['ytrue_mean'], sc_stats['resid_mean'])
print(f"\n  Correlations with scenario residual:")
print(f"  r(oof_mean, resid): {r_oof:.4f}")
print(f"  r(inflow_mean, resid): {r_inf:.4f}")
print(f"  r(ytrue_mean, resid): {r_ytrue:.4f}")

# Scenario oof bins
print(f"\n  Scenario oof_mean bins vs residual:")
bins = [0, 10, 15, 20, 25, 30, 100]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (sc_stats['oof_mean'] >= lo) & (sc_stats['oof_mean'] < hi)
    if m.sum() > 0:
        r = sc_stats.loc[m, 'resid_mean'].mean()
        n = m.sum()
        print(f"  [{lo:3d},{hi:3d}): n={n:5d}  resid={r:+.4f}")

# For test: compute scenario-level oracle_NEW predictions
test_raw['_oN'] = oracle_new_t
te_sc_stats = test_raw.groupby(['layout_id','scenario_id']).agg(
    oN_mean=('_oN','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    n=('_oN','count')
).reset_index()
te_sc_stats['is_unseen'] = ~te_sc_stats['layout_id'].isin(train_layouts)
print(f"\n  Test scenarios: {len(te_sc_stats)}")
print(f"  Unseen test: {te_sc_stats['is_unseen'].sum()}")
print(f"  Unseen test oN_mean: {te_sc_stats.loc[te_sc_stats['is_unseen'],'oN_mean'].describe()}")

# What correction would the linear bias (slope) give for each unseen scenario?
# bias = 0.3799 * oof_mean - 2.8055
te_sc_stats['linear_bias_corr'] = 0.3799 * te_sc_stats['oN_mean'] - 2.8055
te_sc_stats['linear_bias_corr'] = te_sc_stats['linear_bias_corr'].clip(lower=0)

unseen_sc = te_sc_stats[te_sc_stats['is_unseen']].copy()
print(f"\n  Unseen test: linear bias correction per scenario:")
print(f"  mean={unseen_sc['linear_bias_corr'].mean():.4f}  std={unseen_sc['linear_bias_corr'].std():.4f}")
print(f"  range: [{unseen_sc['linear_bias_corr'].min():.3f}, {unseen_sc['linear_bias_corr'].max():.3f}]")

# Apply per-scenario linear bias correction
# For each unseen test row, correction = linear_bias(scenario_oof_mean)
sc_to_corr = {(row['layout_id'], row['scenario_id']): row['linear_bias_corr']
              for _, row in unseen_sc.iterrows()}

corr_sc = np.zeros(len(test_raw))
for (lid, sid), c in sc_to_corr.items():
    m = (test_raw['layout_id'] == lid) & (test_raw['scenario_id'] == sid)
    corr_sc[m.values] = c

ct_sc = oracle_new_t.copy()
ct_sc[unseen_mask] = oracle_new_t[unseen_mask] + corr_sc[unseen_mask]
ct_sc = np.clip(ct_sc, 0, None)
du = ct_sc[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Per-scenario linear bias correction: D={du:+.4f}  seen={ct_sc[seen_mask].mean():.3f}  "
      f"unseen={ct_sc[unseen_mask].mean():.3f}")
fname = 'FINAL_NEW_oN_scLinBias_OOF8.3825.csv'
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_sc
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# Scale to 5.5
scale = 5.5 / du
corr_sc_sc = corr_sc * scale
ct_sc_sc = oracle_new_t.copy()
ct_sc_sc[unseen_mask] = oracle_new_t[unseen_mask] + corr_sc_sc[unseen_mask]
ct_sc_sc = np.clip(ct_sc_sc, 0, None)
du2 = ct_sc_sc[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Per-scenario linear bias scaled(5.5): D={du2:+.4f}  seen={ct_sc_sc[seen_mask].mean():.3f}  "
      f"unseen={ct_sc_sc[unseen_mask].mean():.3f}")
fname2 = 'FINAL_NEW_oN_scLinBias_s5p5_OOF8.3825.csv'
sub2 = sub_tmpl.copy(); sub2['avg_delay_minutes_next_30m'] = ct_sc_sc
sub2.to_csv(fname2, index=False)
print(f"  Saved: {fname2}")

# ================================================================
# Ensemble of multiple scaled-to-5.5 corrections
# ================================================================
print(f"\n{'='*60}")
print("Ensemble of multiple corrections (all scaled to 5.5)")
print(f"{'='*60}")

corr_files = {
    'flat5p5':     'FINAL_NEW_oN_udelta5p5_OOF8.3825.csv',
    'tsHiScaled':  'FINAL_NEW_oN_tsHiScaled5p5_OOF8.3825.csv',
    'physOffset':  'FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv',
    'tsInflow2D':  'FINAL_NEW_oN_tsInflow2D_s5p5_OOF8.3825.csv',
    'physTs5p5':   'FINAL_NEW_oN_physTs_5p5_OOF8.3825.csv',
    'scLinBias_s': 'FINAL_NEW_oN_scLinBias_s5p5_OOF8.3825.csv',
}

preds = {}
for name, fname in corr_files.items():
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        preds[name] = df['avg_delay_minutes_next_30m'].values
        print(f"  Loaded: {name}  unseen_mean={preds[name][unseen_mask].mean():.4f}")
    except:
        print(f"  MISSING: {name}")

for k in list(range(2, len(preds)+1)):
    names = list(preds.keys())[:k]
    ens = np.mean([preds[n] for n in names], axis=0)
    du = ens[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std = np.array([preds[n][unseen_mask] for n in names]).std(axis=0).mean()
    print(f"  Ensemble({k}): D={du:+.4f}  unseen={ens[unseen_mask].mean():.3f}  "
          f"mean_std={std:.4f}  {'+'.join(names)}")

# Save best ensemble (all 6)
all_names = list(preds.keys())
ens_all = np.mean([preds[n] for n in all_names], axis=0)
du_all = ens_all[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname_ens = 'FINAL_NEW_oN_ens6scaled5p5_OOF8.3825.csv'
sub_ens = sub_tmpl.copy(); sub_ens['avg_delay_minutes_next_30m'] = ens_all
sub_ens.to_csv(fname_ens, index=False)
print(f"\n  Ensemble(6) saved: {fname_ens}  D={du_all:+.4f}  unseen={ens_all[unseen_mask].mean():.3f}")

print("\nDone.")
