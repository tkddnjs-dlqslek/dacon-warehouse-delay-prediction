import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

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

inflow_col = 'order_inflow_15m'
train_raw['_resid'] = residuals_train
layout_grp = train_raw.groupby('layout_id')
layout_oof_mean = pd.Series({lid: fw4_oo[(train_raw['layout_id']==lid).values].mean()
                              for lid in train_raw['layout_id'].unique()})
layout_inflow_mean = layout_grp[inflow_col].mean()
layout_resid_mean = layout_grp['_resid'].mean()

lids_tr = layout_resid_mean.index
oof_vals = layout_oof_mean[lids_tr].values
infl_vals = layout_inflow_mean[lids_tr].values
X_tr = np.column_stack([oof_vals, infl_vals, oof_vals**2, infl_vals**2, oof_vals*infl_vals])
y_resid_lv = layout_resid_mean[lids_tr].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
test_layout_inflow = test_raw.groupby('layout_id')[inflow_col].mean()
test_layout_oN = pd.Series({lid: oracle_new_t[(test_raw['layout_id']==lid).values].mean()
                             for lid in test_raw['layout_id'].unique()})

oN_u = test_layout_oN[unseen_lids].values
infl_u = test_layout_inflow[unseen_lids].values
X_te_u = np.column_stack([oN_u, infl_u, oN_u**2, infl_u**2, oN_u*infl_u])

print("Layout-level regression: predict per-layout correction for unseen test")
print(f"Training: {len(X_tr)} layouts, unseen test: {len(X_te_u)} layouts")

sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_u_s = sc.transform(X_te_u)

for alpha in [0.01, 0.1, 1, 10, 100]:
    reg = Ridge(alpha=alpha)
    reg.fit(X_tr_s, y_resid_lv)
    pred_u = reg.predict(X_te_u_s)
    print(f"  alpha={alpha:6.2f}: pred_u mean={pred_u.mean():+.4f}  std={pred_u.std():.4f}  "
          f"min={pred_u.min():+.4f}  max={pred_u.max():+.4f}  R2={reg.score(X_tr_s, y_resid_lv):.4f}")

# Use alpha=1
reg = Ridge(alpha=1)
reg.fit(X_tr_s, y_resid_lv)
pred_u_best = reg.predict(X_te_u_s)

print(f"\n--- 1-feature models ---")
for feat_name, x1, xt in [('oof_mean', oof_vals.reshape(-1,1), oN_u.reshape(-1,1)),
                            ('inflow_mean', infl_vals.reshape(-1,1), infl_u.reshape(-1,1))]:
    sc1 = StandardScaler(); x1_s = sc1.fit_transform(x1); xt_s = sc1.transform(xt)
    r = Ridge(alpha=1); r.fit(x1_s, y_resid_lv)
    p = r.predict(xt_s)
    print(f"  {feat_name}: R2={r.score(x1_s, y_resid_lv):.4f}  pred_u={p.mean():+.4f}  std={p.std():.4f}")

print(f"\n--- Apply per-layout regression correction (alpha=1, 5 features) ---")
lid_to_pred = {lid: pred_u_best[i] for i, lid in enumerate(unseen_lids)}
order = np.argsort(pred_u_best)
print(f"  Layout-level correction stats: mean={pred_u_best.mean():+.4f}  std={pred_u_best.std():.4f}")
print(f"  Range: [{pred_u_best.min():.3f}, {pred_u_best.max():.3f}]")
print(f"  5 lowest:")
for i in order[:5]:
    lid = unseen_lids[i]
    print(f"    {lid:12s}: oN={test_layout_oN[lid]:.2f}  inflow={test_layout_inflow[lid]:.1f}  corr={pred_u_best[i]:+.3f}")
print(f"  5 highest:")
for i in order[-5:]:
    lid = unseen_lids[i]
    print(f"    {lid:12s}: oN={test_layout_oN[lid]:.2f}  inflow={test_layout_inflow[lid]:.1f}  corr={pred_u_best[i]:+.3f}")

ct = oracle_new_t.copy()
for lid in unseen_lids:
    m = (test_raw['layout_id'] == lid).values
    ct[m] = oracle_new_t[m] + lid_to_pred[lid]
ct = np.clip(ct, 0, None)
du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Per-layout regression: D={du:+.4f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
fname = 'FINAL_NEW_oN_lvRegCorr_OOF8.3825.csv'
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# ================================================================
# Correlation analysis: what predicts per-layout residual?
# ================================================================
print(f"\n--- Pearson correlations with layout residual ---")
from scipy.stats import pearsonr
# Additional layout-level features
feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot','_resid')]
layout_feat_means = layout_grp[feat_cols].mean()
lv_resid_arr = layout_resid_mean.values

corrs = []
for col in feat_cols:
    v = layout_feat_means[col].values
    m = np.isfinite(v)
    if m.sum() > 50:
        r, p = pearsonr(v[m], lv_resid_arr[m])
        corrs.append((abs(r), col, r, p))

corrs.sort(reverse=True)
print(f"  Top 20 features by |r| with layout-level residual:")
for absR, col, r, p in corrs[:20]:
    print(f"  {col:40s}: r={r:+.4f}  p={p:.4f}")

print("\nDone.")
