import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
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
layout_resid_mean = layout_grp['_resid'].mean()

# Physical features (layout-level means)
phys_feats = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps',
              'packaging_material_cost', 'order_inflow_15m']
lids_tr = layout_resid_mean.index
y_resid_lv = layout_resid_mean.values

print("="*60)
print("Physical Feature Correction (pack_util, conveyor, truck_wait)")
print("="*60)

# Layout-level means for training and test
tr_lv_feats = layout_grp[phys_feats].mean()
te_lv_feats = test_raw.groupby('layout_id')[phys_feats].mean()

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
seen_lids_te = test_raw[seen_mask]['layout_id'].unique()

X_tr_lv = tr_lv_feats.loc[lids_tr].values
X_u_lv  = te_lv_feats.loc[unseen_lids].values
X_s_lv  = te_lv_feats.loc[seen_lids_te].values

# OOD check for physical features
print(f"\n  Training vs Unseen layout-level feature stats:")
for i, feat in enumerate(phys_feats):
    tr_min, tr_max = X_tr_lv[:,i].min(), X_tr_lv[:,i].max()
    u_mean, u_std = X_u_lv[:,i].mean(), X_u_lv[:,i].std()
    n_ood = ((X_u_lv[:,i] < tr_min) | (X_u_lv[:,i] > tr_max)).sum()
    r, _ = pearsonr(X_tr_lv[:,i], y_resid_lv)
    print(f"  {feat:35s}: r={r:+.4f}  tr=[{tr_min:.3f},{tr_max:.3f}]  "
          f"u_mean={u_mean:.3f}  n_OOD={n_ood}")

# Fit Ridge regression using only top-3 physical features (no oracle mean)
print(f"\n--- Ridge regression (pure physical features, no oracle_mean) ---")
sc = StandardScaler()

# 3-feature: pack_util, outbound_truck_wait, conveyor_speed
X3_tr = sc.fit_transform(X_tr_lv[:, :3])
X3_u  = sc.transform(X_u_lv[:, :3])
X3_s  = sc.transform(X_s_lv[:, :3])

for alpha in [1, 10, 100, 1000]:
    reg = Ridge(alpha=alpha)
    reg.fit(X3_tr, y_resid_lv)
    p_u = reg.predict(X3_u)
    p_s = reg.predict(X3_s)
    print(f"  alpha={alpha:5d}: R2={reg.score(X3_tr, y_resid_lv):.4f}  "
          f"pred_u={p_u.mean():+.4f}±{p_u.std():.4f}  pred_s={p_s.mean():+.4f}±{p_s.std():.4f}")

# 5-feature
sc5 = StandardScaler()
X5_tr = sc5.fit_transform(X_tr_lv)
X5_u  = sc5.transform(X_u_lv)
X5_s  = sc5.transform(X_s_lv)
for alpha in [1, 10, 100, 1000]:
    reg = Ridge(alpha=alpha)
    reg.fit(X5_tr, y_resid_lv)
    p_u = reg.predict(X5_u)
    p_s = reg.predict(X5_s)
    print(f"  (5feat) alpha={alpha:5d}: R2={reg.score(X5_tr, y_resid_lv):.4f}  "
          f"pred_u={p_u.mean():+.4f}±{p_u.std():.4f}  pred_s={p_s.mean():+.4f}±{p_s.std():.4f}")

# Best: alpha=100 3-feature
sc_best = StandardScaler()
X3_tr_b = sc_best.fit_transform(X_tr_lv[:, :3])
X3_u_b  = sc_best.transform(X_u_lv[:, :3])
reg_best = Ridge(alpha=100)
reg_best.fit(X3_tr_b, y_resid_lv)
pred_u_phys = reg_best.predict(X3_u_b)

print(f"\n  Alpha=100 3-feature prediction per unseen layout:")
for i in np.argsort(pred_u_phys):
    lid = unseen_lids[i]
    print(f"  {lid:12s}: pack_util={X_u_lv[i,0]:.4f}  truck_wait={X_u_lv[i,1]:.4f}  "
          f"conveyor={X_u_lv[i,2]:.4f}  corr={pred_u_phys[i]:+.3f}")

# Apply correction
ct_phys = oracle_new_t.copy()
lid_to_corr = {lid: pred_u_phys[i] for i, lid in enumerate(unseen_lids)}
for lid in unseen_lids:
    m = (test_raw['layout_id'] == lid).values
    ct_phys[m] = oracle_new_t[m] + lid_to_corr[lid]
ct_phys = np.clip(ct_phys, 0, None)
du = ct_phys[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  Physical feature correction (alpha=100): D={du:+.4f}  "
      f"seen={ct_phys[seen_mask].mean():.3f}  unseen={ct_phys[unseen_mask].mean():.3f}")
fname = 'FINAL_NEW_oN_physFeat_OOF8.3825.csv'
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_phys
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# ================================================================
# Leave-one-layout-out validation of physical feature Ridge
# ================================================================
print(f"\n--- LOOO validation (3 physical features) ---")
looo_preds = np.zeros(len(lids_tr))
for i in range(len(lids_tr)):
    X_o = np.delete(X_tr_lv[:, :3], i, axis=0)
    y_o = np.delete(y_resid_lv, i)
    sc_l = StandardScaler(); Xo_s = sc_l.fit_transform(X_o)
    Xi_s = sc_l.transform(X_tr_lv[i:i+1, :3])
    r = Ridge(alpha=100); r.fit(Xo_s, y_o)
    looo_preds[i] = r.predict(Xi_s)[0]

looo_mae = np.mean(np.abs(looo_preds - y_resid_lv))
print(f"  LOOO MAE: {looo_mae:.4f}  (vs mean residual: {y_resid_lv.mean():.4f})")
print(f"  LOOO pred mean: {looo_preds.mean():+.4f}  std: {looo_preds.std():.4f}")

print("\nDone.")
