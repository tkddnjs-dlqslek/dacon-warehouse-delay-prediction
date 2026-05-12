import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings; warnings.filterwarnings('ignore')
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
train_raw['_resid'] = residuals_train; train_raw['_oof'] = fw4_oo

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

X_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_te = te_lv_u[['oN_mean','pu','tw','conv']].values
y = y_lv
p95 = np.percentile(y, 95)

# Fit GBM
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_m = GradientBoostingRegressor(**gbm_kw); gbm_m.fit(X_tr, y)
gbm_tp = np.minimum(gbm_m.predict(X_te), p95)

# Fit GP Matern52
kern = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
sc_gp = StandardScaler()
gp_m = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=5)
gp_m.fit(sc_gp.fit_transform(X_tr), y)
gp_tp = gp_m.predict(sc_gp.transform(X_te))

# Best blend: GBM(0.1)+GP(0.9)
blend_tp = 0.1*gbm_tp + 0.9*gp_tp
print("="*70)
print("FINAL BEST: GBM(0.1)+GP_Matern52(0.9) — LOOO=1.7139")
print("="*70)
print(f"\n  GBM test: mean={gbm_tp.mean():.3f}  max={gbm_tp.max():.3f}")
print(f"  GP  test: mean={gp_tp.mean():.3f}  max={gp_tp.max():.3f}")
print(f"  Blend test: mean={blend_tp.mean():.3f}  max={blend_tp.max():.3f}  std={blend_tp.std():.3f}")

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
    mx = corr_arr[unseen_mask].max(); mn = corr_arr[unseen_mask].min()
    print(f"  {label:65s}: D={du:+.4f}  std={std:.3f}  [{mn:.2f},{mx:.2f}]")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return scaled

print(f"\n--- Saving GP Blend Submissions ---")
save_corr(blend_tp, 5.5,  'GBM(0.1)+GP_M52(0.9) Δ=5.5   LOOO=1.7139', 'FINAL_NEW_oN_gbm1gp9_5p5_OOF8.3825.csv')
save_corr(blend_tp, 5.77, 'GBM(0.1)+GP_M52(0.9) Δ=5.77  LOOO=1.7139', 'FINAL_NEW_oN_gbm1gp9_5p77_OOF8.3825.csv')

# Also save GP standalone and GP(0.2) for comparison
save_corr(gp_tp,    5.5,  'GP_M52 standalone    Δ=5.5   LOOO=1.7148', 'FINAL_NEW_oN_gpM52_5p5_OOF8.3825.csv')
blend2 = 0.2*gbm_tp + 0.8*gp_tp
save_corr(blend2,   5.5,  'GBM(0.2)+GP_M52(0.8) Δ=5.5   LOOO=1.7144', 'FINAL_NEW_oN_gbm2gp8_5p5_OOF8.3825.csv')

# Per-layout comparison
print(f"\n--- Per-Layout (sorted by GP blend, Δ=5.5) ---")
blend_s = blend_tp + (5.5 - blend_tp.mean())
gp_s = gp_tp + (5.5 - gp_tp.mean())
gbm_s = gbm_tp + (5.5 - gbm_tp.mean())

prev_ri = pd.read_csv('FINAL_NEW_oN_gbmCapRidgeInt_5p5_OOF8.3825.csv')
prev_ri_t = prev_ri.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
prev_by_lid = {lid: (prev_ri_t[test_raw['layout_id']==lid].mean() -
                     oracle_new_t[test_raw['layout_id']==lid].mean())
               for lid in unseen_lids}

te_lv_u['gp'] = gp_s; te_lv_u['blend'] = blend_s; te_lv_u['gbm'] = gbm_s
te_lv_u['prev_ri'] = te_lv_u['layout_id'].map(prev_by_lid)

print(f"  {'layout_id':10s}  {'pu':5s}  {'gbm':6s}  {'gp':6s}  {'blend':6s}  {'prev_ri':7s}")
for _, row in te_lv_u.sort_values('blend').iterrows():
    print(f"  {row['layout_id']:10s}  {row['pu']:.3f}  {row['gbm']:6.2f}  {row['gp']:6.2f}  "
          f"{row['blend']:6.2f}  {row['prev_ri']:7.2f}")

print(f"\n  SUMMARY:")
print(f"  {'model':35s}  {'LOOO':7s}  {'std':5s}  {'max':5s}  {'min':5s}")
for name, arr in [
    ('flat_5.5', np.full(50, 5.5)),
    ('prev_best (GBM(0.7)+RidgeInt(0.3))', te_lv_u['prev_ri'].values),
    ('GBM_cap_p95 standalone', gbm_s.values),
    ('GP_Matern52 standalone (LOOO=1.7148)', gp_s.values),
    ('GBM(0.1)+GP(0.9) (LOOO=1.7139)', blend_s.values),
]:
    arr = np.array(arr)
    print(f"  {name:35s}  {'---':7s}  {arr.std():.3f}  {arr.max():.2f}  {arr.min():.2f}")

print("\nDone.")
