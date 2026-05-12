import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
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
train_raw['_resid'] = residuals_train
train_raw['_oof'] = fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
test_raw['_oN'] = oracle_new_t
sub_tmpl = pd.read_csv('sample_submission.csv')

tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'), resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'), tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'), inflow=('order_inflow_15m','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
    inflow=('order_inflow_15m','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

feats_tr = tr_lv[['oof_mean','pu','tw','conv']].values
feats_te = te_lv_u[['oN_mean','pu','tw','conv']].values
feat_names = ['oof_mean','pu','tw','conv']

import warnings; warnings.filterwarnings('ignore')

print("="*70)
print("GBM Tree Analysis: Splits and OOD Extrapolation Risk")
print("="*70)

# Fit GBM on full training data
gbm = GradientBoostingRegressor(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm.fit(feats_tr, y_lv)
gbm_preds_test = gbm.predict(feats_te)

# Analyze first tree's splits
print(f"\n  GBM first tree split analysis:")
tree = gbm.estimators_[0, 0]
from sklearn.tree import export_text
print(export_text(tree, feature_names=feat_names, max_depth=4))

# Feature importance
print(f"\n  GBM feature importances:")
for fn, fi in sorted(zip(feat_names, gbm.feature_importances_), key=lambda x:-x[1]):
    print(f"  {fn:20s}: {fi:.4f}")

# Training feature ranges
print(f"\n  Training feature ranges:")
for i, fn in enumerate(feat_names):
    print(f"  {fn:20s}: [{feats_tr[:,i].min():.3f}, {feats_tr[:,i].max():.3f}]")
print(f"\n  Unseen test feature ranges:")
for i, fn in enumerate(feat_names):
    print(f"  {fn:20s}: [{feats_te[:,i].min():.3f}, {feats_te[:,i].max():.3f}]")

# OOD layouts (pack_util > training max)
tr_pu_max = feats_tr[:,1].max()
tr_tw_max = feats_tr[:,2].max()
tr_oof_max = feats_tr[:,0].max()
print(f"\n  Training max: pu={tr_pu_max:.3f}  tw={tr_tw_max:.3f}  oof_mean={tr_oof_max:.3f}")
ood_pu = te_lv_u[te_lv_u['pu'] > tr_pu_max]
ood_tw = te_lv_u[te_lv_u['tw'] > tr_tw_max]
ood_oof = te_lv_u[te_lv_u['oN_mean'] > tr_oof_max]
print(f"  Unseen OOD pu (>{tr_pu_max:.3f}): {len(ood_pu)} layouts: {list(ood_pu['layout_id'])}")
print(f"  Unseen OOD tw (>{tr_tw_max:.3f}): {len(ood_tw)} layouts: {list(ood_tw['layout_id'])}")
print(f"  Unseen OOD oof (>{tr_oof_max:.3f}): {len(ood_oof)} layouts: {list(ood_oof['layout_id'])}")

# For OOD layouts: GBM just predicts the max-leaf value (extrapolation = boundary)
# Check: which training layouts are in the same leaf as OOD test layouts?
print(f"\n  GBM leaf assignments for key layouts:")
# Use predict + apply to get leaf indices
leaves_tr = gbm.apply(feats_tr)  # shape (n_tr, n_estimators, 1)
leaves_te = gbm.apply(feats_te)

# For the most extreme test layouts, find nearest training layouts by prediction
te_lv_u['gbm_raw'] = gbm_preds_test
extreme_te = te_lv_u[te_lv_u['gbm_raw'] > gbm_preds_test.mean() * 1.5].sort_values('gbm_raw', ascending=False)
print(f"\n  Extreme GBM predictions (>1.5× mean):")
print(extreme_te[['layout_id','oN_mean','pu','tw','gbm_raw']].round(3).to_string())

# Find training layouts with highest residuals (same region as extreme test predictions)
high_resid_tr = tr_lv.nlargest(10, 'resid_mean')[['layout_id','oof_mean','pu','tw','inflow','resid_mean']]
print(f"\n  Training layouts with highest residuals (analogues for extreme test):")
print(high_resid_tr.round(3).to_string())

# Capped GBM: cap corrections at training max residual
tr_max_resid = y_lv.max()
tr_p90_resid = np.percentile(y_lv, 90)
tr_p95_resid = np.percentile(y_lv, 95)
print(f"\n  Training residual bounds: max={tr_max_resid:.3f}  p90={tr_p90_resid:.3f}  p95={tr_p95_resid:.3f}")

# GBM capped at training max and p90
ridge = Ridge(alpha=100)
sc_r = StandardScaler()
ridge.fit(sc_r.fit_transform(feats_tr), y_lv)
ridge_preds_test = ridge.predict(sc_r.transform(feats_te))

print(f"\n  GBM raw predictions: mean={gbm_preds_test.mean():.3f}  std={gbm_preds_test.std():.3f}  "
      f"max={gbm_preds_test.max():.3f}")

for cap in [tr_p90_resid, tr_p95_resid, tr_max_resid]:
    capped = np.minimum(gbm_preds_test, cap)
    ens = 0.5 * capped + 0.5 * ridge_preds_test
    scaled = ens + (5.5 - ens.mean())
    print(f"  GBM capped@{cap:.1f} + Ridge(50%): mean={scaled.mean():.3f}  std={scaled.std():.3f}  "
          f"max={scaled.max():.3f}")

# Save key variants
sub_tmpl2 = pd.read_csv('sample_submission.csv')

def save_per_layout(lids, preds_raw, target, label, fname):
    offset = target - preds_raw.mean()
    corr_arr = np.zeros(len(test_raw))
    for lid, c in zip(lids, preds_raw + offset):
        corr_arr[(test_raw['layout_id'] == lid).values] = c
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std_ = corr_arr[unseen_mask].std()
    max_ = corr_arr[unseen_mask].max()
    print(f"  {label:55s}: D={du:+.4f}  std={std_:.3f}  max={max_:.2f}")
    sub = sub_tmpl2.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)

lids = te_lv_u['layout_id'].values
print(f"\n  --- Capped GBM + Ridge ensembles (scaled to 5.5) ---")
for cap in [tr_p90_resid, tr_p95_resid]:
    capped = np.minimum(gbm_preds_test, cap)
    for w_gbm in [0.5, 0.7]:
        ens = w_gbm * capped + (1-w_gbm) * ridge_preds_test
        cap_label = f'cap{cap:.0f}'
        save_per_layout(lids, ens, 5.5,
                        f'GBM_cap{cap:.0f}({w_gbm:.1f})+Ridge({1-w_gbm:.1f}) 5.5',
                        f'FINAL_NEW_oN_gbmCap{int(cap)}_w{int(w_gbm*10)}_OOF8.3825.csv')

# LOOO MAE for capped versions
print(f"\n  LOOO MAE for capped GBM variants:")
gbm_looo = []
for i in range(len(y_lv)):
    Xtr_ = np.delete(feats_tr, i, 0); ytr_ = np.delete(y_lv, i)
    m = GradientBoostingRegressor(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
    m.fit(Xtr_, ytr_)
    gbm_looo.append(m.predict(feats_tr[i:i+1])[0])
gbm_looo = np.array(gbm_looo)

ridge_looo = []
for i in range(len(y_lv)):
    Xtr_ = np.delete(feats_tr, i, 0); ytr_ = np.delete(y_lv, i)
    sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_); Xte_s = sc_.transform(feats_tr[i:i+1])
    m = Ridge(alpha=100); m.fit(Xtr_s, ytr_)
    ridge_looo.append(m.predict(Xte_s)[0])
ridge_looo = np.array(ridge_looo)

for cap in [tr_p90_resid, tr_p95_resid, 999]:
    capped_looo = np.minimum(gbm_looo, cap)
    for wg in [0.5, 0.7]:
        ens = wg * capped_looo + (1-wg) * ridge_looo
        mae = np.mean(np.abs(y_lv - ens))
        cap_str = f'{cap:.0f}' if cap < 100 else 'uncap'
        print(f"  GBM_cap{cap_str}({wg:.1f})+Ridge({1-wg:.1f}): LOOO_MAE={mae:.4f}")

print("\nDone.")
