import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
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
train_raw['_resid'] = y_true - fw4_oo; train_raw['_oof'] = fw4_oo

tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'), resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'), tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
).reset_index()
y = tr_lv['resid_mean'].values
lids = tr_lv['layout_id'].values
X_tr = tr_lv[['oof_mean','pu','tw','conv']].values
p95 = np.percentile(y, 95)

gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
kern = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)

print("Computing LOOO predictions for all 250 layouts...")
gbm_looo, gp_looo = [], []
for i in range(len(y)):
    # GBM
    m = GradientBoostingRegressor(**gbm_kw)
    m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
    gbm_looo.append(min(m.predict(X_tr[i:i+1])[0], p95))
    # GP
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(X_tr, i, 0))
    Xte_s = sc_.transform(X_tr[i:i+1])
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2)
    gp.fit(Xtr_s, np.delete(y, i))
    gp_looo.append(gp.predict(Xte_s)[0])
    if (i+1) % 50 == 0: print(f"  {i+1}/250 done...")

gbm_looo = np.array(gbm_looo); gp_looo = np.array(gp_looo)

mae_gbm = np.mean(np.abs(y - gbm_looo))
mae_gp = np.mean(np.abs(y - gp_looo))
print(f"\nMAE: GBM={mae_gbm:.4f}  GP={mae_gp:.4f}  (ΔLOOO={mae_gbm-mae_gp:.4f})")

# Per-layout error analysis
err_gbm = np.abs(y - gbm_looo)
err_gp = np.abs(y - gp_looo)
delta_err = err_gbm - err_gp  # positive = GP wins, negative = GBM wins

print(f"\nLayouts where GP wins (|error_gbm - error_gp| > 0.5):")
idx = np.where(delta_err > 0.5)[0]
idx = idx[np.argsort(delta_err[idx])[::-1]][:15]
print(f"  {'layout':10s}  {'pu':5s}  {'y_true':7s}  {'gbm_pred':8s}  {'gp_pred':7s}  {'err_gbm':8s}  {'err_gp':7s}  {'delta':6s}")
for i in idx:
    print(f"  {lids[i]:10s}  {X_tr[i,1]:.3f}  {y[i]:7.3f}  {gbm_looo[i]:8.3f}  {gp_looo[i]:7.3f}  "
          f"{err_gbm[i]:8.3f}  {err_gp[i]:7.3f}  {delta_err[i]:+6.3f}")

print(f"\nLayouts where GBM wins (err_gbm - err_gp < -0.5):")
idx2 = np.where(delta_err < -0.5)[0]
idx2 = idx2[np.argsort(delta_err[idx2])[:5]]
print(f"  {'layout':10s}  {'pu':5s}  {'y_true':7s}  {'gbm_pred':8s}  {'gp_pred':7s}  {'err_gbm':8s}  {'err_gp':7s}  {'delta':6s}")
for i in idx2:
    print(f"  {lids[i]:10s}  {X_tr[i,1]:.3f}  {y[i]:7.3f}  {gbm_looo[i]:8.3f}  {gp_looo[i]:7.3f}  "
          f"{err_gbm[i]:8.3f}  {err_gp[i]:7.3f}  {delta_err[i]:+6.3f}")

# Fine-grained blend search
print(f"\nFine blend: GBM_cap + GP:")
for wg in np.arange(0.0, 1.05, 0.1):
    wgp = 1 - wg
    bl = wg*gbm_looo + wgp*gp_looo
    mae = np.mean(np.abs(y - bl))
    print(f"  GBM({wg:.1f})+GP({wgp:.1f}): LOOO={mae:.4f}")

print("\nDone.")
