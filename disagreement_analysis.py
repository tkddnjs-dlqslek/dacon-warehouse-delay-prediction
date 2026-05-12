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

# Physical feature Ridge (3-feat) for unseen layouts
phys_feats = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps']
train_raw['_resid'] = residuals_train
layout_grp_tr = train_raw.groupby('layout_id')
layout_resid_mean = layout_grp_tr['_resid'].mean()
lids_tr = layout_resid_mean.index
y_resid_lv = layout_resid_mean.values
X_tr_lv = layout_grp_tr[phys_feats].mean().loc[lids_tr].values
te_lv_feats = test_raw.groupby('layout_id')[phys_feats].mean()
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
X_u_lv = te_lv_feats.loc[unseen_lids].values
sc = StandardScaler(); X_tr_s = sc.fit_transform(X_tr_lv); X_u_s = sc.transform(X_u_lv)
reg = Ridge(alpha=100); reg.fit(X_tr_s, y_resid_lv)
phys3_raw = reg.predict(X_u_s)
phys3_offset = 5.5 - phys3_raw.mean()
phys3_corr = phys3_raw + phys3_offset
lid_to_phys3 = dict(zip(unseen_lids, phys3_corr))

# Load iso_lvL75 corrections per layout
iso_df = pd.read_csv('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv').set_index('ID').reindex(id_order).reset_index()
iso_t = iso_df['avg_delay_minutes_next_30m'].values
test_raw['_iso_corr'] = iso_t - oracle_new_t

# Compute per-layout iso correction mean
iso_by_layout = test_raw[unseen_mask].groupby('layout_id')['_iso_corr'].mean()
lid_to_iso = iso_by_layout.to_dict()

# Physical features per layout (for unseen test)
te_phys = test_raw[unseen_mask].groupby('layout_id')[
    phys_feats + ['order_inflow_15m']].mean()
te_phys['iso_corr'] = te_phys.index.map(lid_to_iso)
te_phys['phys3_corr'] = te_phys.index.map(lid_to_phys3)
te_phys['diff'] = te_phys['iso_corr'] - te_phys['phys3_corr']
te_phys['oracle_mean'] = test_raw[unseen_mask].groupby('layout_id')['_iso_corr'].transform('count') \
    .groupby(test_raw[unseen_mask]['layout_id']).first().reindex(te_phys.index)
# Actually compute oracle_new mean per layout
oN_by_layout = test_raw.copy()
oN_by_layout['_oN'] = oracle_new_t
oN_by_layout = oN_by_layout[unseen_mask].groupby('layout_id')['_oN'].mean()
te_phys['oN_mean'] = oN_by_layout

print("="*70)
print("Disagreement Analysis: iso_lvL75 vs physOffset5p5")
print("="*70)

# Training layout physical features (for context)
tr_phys = layout_grp_tr[phys_feats + ['order_inflow_15m']].mean()
print(f"\n  Training layout physical feature ranges:")
for f in phys_feats + ['order_inflow_15m']:
    print(f"  {f:35s}: train=[{tr_phys[f].min():.3f}, {tr_phys[f].max():.3f}]  "
          f"unseen=[{te_phys[f].min():.3f}, {te_phys[f].max():.3f}]")

print(f"\n  Physical feature regression coefficients (scaled):")
for feat, coef in zip(phys_feats, reg.coef_):
    print(f"  {feat:35s}: {coef:+.4f}")
print(f"  Intercept: {reg.intercept_:+.4f}")

# Top disagreement: iso >> phys
print(f"\n  Layouts where iso >> phys (iso says hard, phys says easy):")
top_iso = te_phys.nlargest(10, 'diff')[
    ['pack_utilization','outbound_truck_wait_min','conveyor_speed_mps',
     'order_inflow_15m','oN_mean','iso_corr','phys3_corr','diff']]
print(top_iso.round(3).to_string())

print(f"\n  Layouts where phys >> iso (phys says hard, iso says easy):")
top_phys = te_phys.nsmallest(10, 'diff')[
    ['pack_utilization','outbound_truck_wait_min','conveyor_speed_mps',
     'order_inflow_15m','oN_mean','iso_corr','phys3_corr','diff']]
print(top_phys.round(3).to_string())

# Correlation between oN_mean and corrections
from scipy.stats import pearsonr
r_on_iso, _ = pearsonr(te_phys['oN_mean'], te_phys['iso_corr'])
r_on_phys, _ = pearsonr(te_phys['oN_mean'], te_phys['phys3_corr'])
r_pu_iso, _ = pearsonr(te_phys['pack_utilization'], te_phys['iso_corr'])
r_pu_phys, _ = pearsonr(te_phys['pack_utilization'], te_phys['phys3_corr'])
print(f"\n  Correlations (layout-level):")
print(f"  r(oN_mean, iso_corr)   = {r_on_iso:.4f}")
print(f"  r(oN_mean, phys_corr)  = {r_on_phys:.4f}")
print(f"  r(pack_util, iso_corr) = {r_pu_iso:.4f}")
print(f"  r(pack_util, phys_corr) = {r_pu_phys:.4f}")

# Check: in training data, does high oracle_mean correlate with high residual?
tr_sc_stats = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof' if '_oof' in train_raw.columns else '_resid', 'mean'),
    resid_mean=('_resid','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
train_raw['_oof'] = fw4_oo
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'),
    resid_mean=('_resid','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean')
).reset_index()

r_oof_resid, _ = pearsonr(tr_lv['oof_mean'], tr_lv['resid_mean'])
r_pu_resid, _ = pearsonr(tr_lv['pu'], tr_lv['resid_mean'])
print(f"\n  Training layout-level validations:")
print(f"  r(oof_mean, resid_mean) = {r_oof_resid:.4f}  ← basis for iso_lvL75")
print(f"  r(pack_util, resid_mean) = {r_pu_resid:.4f}  ← basis for physOffset5p5")

# Check: for training, which predictor of residual is better? oof_mean or phys features?
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, LeaveOneOut
import warnings
warnings.filterwarnings('ignore')

X_oof = tr_lv[['oof_mean']].values
X_pu  = tr_lv[['pu','tw','conv']].values
y_lr  = tr_lv['resid_mean'].values

loo = LeaveOneOut()
sc2 = StandardScaler()
X_pu_s = sc2.fit_transform(X_pu)

lr_oof  = Ridge(alpha=1.0)
lr_phys = Ridge(alpha=100)

# LOOO scores
oof_preds  = np.array([Ridge(alpha=1.0).fit(np.delete(X_oof,i,0), np.delete(y_lr,i)).predict(X_oof[i:i+1])[0]
                        for i in range(len(y_lr))])
phys_preds = np.array([Ridge(alpha=100).fit(
    StandardScaler().fit_transform(np.delete(X_pu,i,0)), np.delete(y_lr,i))
    .predict(StandardScaler().fit_transform(np.delete(X_pu,i,0))[-1:]  # wrong indexing fix:
    if False else StandardScaler().fit(np.delete(X_pu,i,0)).transform(X_pu[i:i+1]))[0]
                        for i in range(len(y_lr))])

oof_mae  = np.mean(np.abs(y_lr - oof_preds))
phys_mae = np.mean(np.abs(y_lr - phys_preds))
print(f"\n  LOOO MAE on TRAINING layouts:")
print(f"  oof_mean (iso basis): {oof_mae:.4f}")
print(f"  phys3feat (phys basis): {phys_mae:.4f}")

# Combined predictor
X_both = np.column_stack([X_oof, X_pu_s])
both_preds = []
for i in range(len(y_lr)):
    X_tr_both = np.delete(X_both, i, 0); y_tr_both = np.delete(y_lr, i)
    sc3 = StandardScaler(); X_tr_both_s = sc3.fit_transform(X_tr_both)
    X_te_both_s = sc3.transform(X_both[i:i+1])
    lr = Ridge(alpha=10); lr.fit(X_tr_both_s, y_tr_both)
    both_preds.append(lr.predict(X_te_both_s)[0])
both_mae = np.mean(np.abs(y_lr - np.array(both_preds)))
print(f"  oof_mean + phys3feat (combined): {both_mae:.4f}")

print(f"\n  → The lower LOOO MAE method should give better per-layout corrections")
print(f"\nDone.")
