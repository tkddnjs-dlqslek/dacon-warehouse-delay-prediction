import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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
    conv=('conveyor_speed_mps','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

X_base_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_base_te = te_lv_u[['oN_mean','pu','tw','conv']].values

# Build oof*pu interaction features
pu_oof_tr = X_base_tr[:,0] * X_base_tr[:,1]
pu_oof_te = X_base_te[:,0] * X_base_te[:,1]
X_int_tr = np.column_stack([X_base_tr, pu_oof_tr])
X_int_te = np.column_stack([X_base_te, pu_oof_te])

import warnings; warnings.filterwarnings('ignore')

# Fit final models
p95 = np.percentile(y_lv, 95)

gbm = GradientBoostingRegressor(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm.fit(X_base_tr, y_lv)
gbm_test = gbm.predict(X_base_te)
gbm_capped = np.minimum(gbm_test, p95)

sc_int = StandardScaler()
X_int_tr_s = sc_int.fit_transform(X_int_tr)
X_int_te_s = sc_int.transform(X_int_te)
ridge_int = Ridge(alpha=100)
ridge_int.fit(X_int_tr_s, y_lv)
ridge_int_test = ridge_int.predict(X_int_te_s)

# BEST: 70% GBM_cap + 30% Ridge_int
best_blend = 0.7 * gbm_capped + 0.3 * ridge_int_test

print("="*70)
print("Final Best Model: GBM_cap_p95(0.7) + Ridge_int(0.3)")
print(f"  LOOO_MAE = 1.7617 (best achieved)")
print("="*70)

def save_corr(blend_preds, target, label, fname):
    scaled = blend_preds + (target - blend_preds.mean())
    corr_arr = np.zeros(len(test_raw))
    for lid, c in zip(te_lv_u['layout_id'], scaled):
        corr_arr[(test_raw['layout_id'] == lid).values] = c
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std = corr_arr[unseen_mask].std()
    mx = corr_arr[unseen_mask].max()
    mn = corr_arr[unseen_mask].min()
    print(f"  {label:55s}: D={du:+.4f}  std={std:.3f}  [{mn:.2f},{mx:.2f}]")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return scaled

print(f"\n  Raw blend stats: mean={best_blend.mean():.3f}  std={best_blend.std():.3f}")

# Save at multiple targets
best_5p5   = save_corr(best_blend, 5.5,  'GBM_cap_p95(0.7)+Ridge_int(0.3) Δ=5.5',
                        'FINAL_NEW_oN_gbmCapRidgeInt_5p5_OOF8.3825.csv')
best_5p77  = save_corr(best_blend, 5.77, 'GBM_cap_p95(0.7)+Ridge_int(0.3) Δ=5.77',
                        'FINAL_NEW_oN_gbmCapRidgeInt_5p77_OOF8.3825.csv')
best_6p0   = save_corr(best_blend, 6.0,  'GBM_cap_p95(0.7)+Ridge_int(0.3) Δ=6.0',
                        'FINAL_NEW_oN_gbmCapRidgeInt_6p0_OOF8.3825.csv')

# Show per-layout comparison
print(f"\n  Per-layout comparison (sorted by best_5p5):")
te_lv_u['best_5p5'] = best_5p5
te_lv_u['gbm_cap'] = gbm_capped + (5.5 - gbm_capped.mean())
te_lv_u['ridge_int'] = ridge_int_test + (5.5 - ridge_int_test.mean())

# Load iso and prev best
iso_df = pd.read_csv('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv')
iso_t = iso_df.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
iso_by_lid = {lid: (iso_t[test_raw['layout_id']==lid].mean() -
                    oracle_new_t[test_raw['layout_id']==lid].mean())
              for lid in unseen_lids}
te_lv_u['iso'] = te_lv_u['layout_id'].map(iso_by_lid)

prev_best = pd.read_csv('FINAL_NEW_oN_gbmCapP95_w7_5p5_OOF8.3825.csv')
prev_t = prev_best.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
prev_by_lid = {lid: (prev_t[test_raw['layout_id']==lid].mean() -
                      oracle_new_t[test_raw['layout_id']==lid].mean())
               for lid in unseen_lids}
te_lv_u['prev'] = te_lv_u['layout_id'].map(prev_by_lid)

print(f"  {'layout_id':10s}  {'oN':5s}  {'pu':5s}  {'iso':5s}  {'prev':5s}  {'new':5s}  {'diff':5s}")
for _, row in te_lv_u.sort_values('best_5p5').iterrows():
    diff = row['best_5p5'] - row['prev']
    print(f"  {row['layout_id']:10s}  {row['oN_mean']:5.1f}  {row['pu']:.3f}  "
          f"{row['iso']:5.2f}  {row['prev']:5.2f}  {row['best_5p5']:5.2f}  {diff:+5.2f}")

# Statistics comparison
print(f"\n  Statistics comparison (Δ=5.5):")
for label, arr in [
    ('flat 5.5', np.full(50,5.5)),
    ('ridge+int a100', te_lv_u['ridge_int'].values),
    ('prev_best (gbmCap70+ridge30)', te_lv_u['prev'].values),
    ('new_best (gbmCap70+ridgeInt30)', te_lv_u['best_5p5'].values),
    ('iso_lvL75', te_lv_u['iso'].values),
]:
    arr = np.array(arr)
    print(f"  {label:40s}: std={arr.std():.3f}  [{arr.min():.2f},{arr.max():.2f}]  mean={arr.mean():.3f}")

print("\nDone.")
