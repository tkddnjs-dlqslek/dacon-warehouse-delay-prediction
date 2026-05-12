import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import HuberRegressor
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
train_raw['_resid'] = y_true - fw4_oo; train_raw['_oof'] = fw4_oo

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
y = tr_lv['resid_mean'].values
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

X_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_te = te_lv_u[['oN_mean','pu','tw','conv']].values
p95 = np.percentile(y, 95)

# Build interaction features
pu_oof_tr = X_tr[:,0]*X_tr[:,1]; pu_oof_te = X_te[:,0]*X_te[:,1]
X_int_tr = np.column_stack([X_tr, pu_oof_tr])
X_int_te = np.column_stack([X_te, pu_oof_te])

kern_iso = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
kern_int = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)

def looo_gp(X_tr_, X_te_, y_, kern, nr=3):
    looo = []
    for i in range(len(y_)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr_, i, 0))
        Xte_s = sc_.transform(X_tr_[i:i+1])
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=nr)
        gp.fit(Xtr_s, np.delete(y_, i))
        looo.append(gp.predict(Xte_s)[0])
    sc_f = StandardScaler()
    gp_f = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=nr)
    gp_f.fit(sc_f.fit_transform(X_tr_), y_)
    return np.array(looo), gp_f.predict(sc_f.transform(X_te_))

# Precompute all components
print("Computing GP_int LOOO (5 features: oof, pu, tw, conv, pu*oof)...")
gp_int_looo, gp_int_tp = looo_gp(X_int_tr, X_int_te, y, kern_int)
mae_gp_int = np.mean(np.abs(y - gp_int_looo))
print(f"  GP_int Matern52: LOOO={mae_gp_int:.4f}  test_mean={gp_int_tp.mean():.3f}")

print("Computing GP_base LOOO (4 features)...")
gp_looo, gp_tp = looo_gp(X_tr, X_te, y, kern_iso)
mae_gp = np.mean(np.abs(y - gp_looo))
print(f"  GP_base Matern52: LOOO={mae_gp:.4f}  test_mean={gp_tp.mean():.3f}")

# GBM
print("Computing GBM LOOO...")
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_looo = []
for i in range(len(y)):
    m = GradientBoostingRegressor(**gbm_kw)
    m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
    gbm_looo.append(min(m.predict(X_tr[i:i+1])[0], p95))
gbm_looo = np.array(gbm_looo)
gbm_m = GradientBoostingRegressor(**gbm_kw); gbm_m.fit(X_tr, y)
gbm_tp = np.minimum(gbm_m.predict(X_te), p95)

# Huber
print("Computing Huber LOOO...")
huber_looo = []
for i in range(len(y)):
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(X_int_tr, i, 0))
    Xte_s = sc_.transform(X_int_tr[i:i+1])
    m = HuberRegressor(epsilon=2.0, alpha=0.0001, max_iter=1000)
    m.fit(Xtr_s, np.delete(y, i))
    huber_looo.append(m.predict(Xte_s)[0])
huber_looo = np.array(huber_looo)
sc_h = StandardScaler(); huber_m = HuberRegressor(epsilon=2.0, alpha=0.0001, max_iter=1000)
huber_m.fit(sc_h.fit_transform(X_int_tr), y); huber_tp = huber_m.predict(sc_h.transform(X_int_te))

print("="*70)
print("Weight Sweeps for GP_int (5-feature)")
print("="*70)

# GP_int + GBM
print(f"\n--- GP_int + GBM_cap ---")
best_blend = (99, 0, 0, None)
for wgi in np.arange(0.5, 1.05, 0.1):
    wg = 1 - wgi
    bl = wgi*gp_int_looo + wg*gbm_looo
    mae = np.mean(np.abs(y - bl))
    tp_bl = wgi*gp_int_tp + wg*gbm_tp
    print(f"  GP_int({wgi:.1f})+GBM({wg:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f}  max={tp_bl.max():.2f}")
    if mae < best_blend[0]:
        best_blend = (mae, wgi, wg, tp_bl)

# GP_int + Huber
print(f"\n--- GP_int + HuberInt ---")
best_blend2 = (99, 0, 0, None)
for wgi in np.arange(0.5, 1.05, 0.1):
    wh = 1 - wgi
    bl = wgi*gp_int_looo + wh*huber_looo
    mae = np.mean(np.abs(y - bl))
    tp_bl = wgi*gp_int_tp + wh*huber_tp
    print(f"  GP_int({wgi:.1f})+Huber({wh:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f}  max={tp_bl.max():.2f}")
    if mae < best_blend2[0]:
        best_blend2 = (mae, wgi, wh, tp_bl)

# GP_int + GP_base
print(f"\n--- GP_int + GP_base ---")
for wgi in [0.6, 0.7, 0.8, 0.9]:
    wb = 1 - wgi
    bl = wgi*gp_int_looo + wb*gp_looo
    mae = np.mean(np.abs(y - bl))
    tp_bl = wgi*gp_int_tp + wb*gp_tp
    print(f"  GP_int({wgi:.1f})+GP_base({wb:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f}  max={tp_bl.max():.2f}")

# 3-way: GP_int + GP_base + GBM
print(f"\n--- 3-way: GP_int + GP_base + GBM ---")
best3 = (99, 0, 0, 0, None)
for wgi in [0.5, 0.6, 0.7]:
    for wgb in [0.1, 0.2, 0.3]:
        wg_gbm = 1 - wgi - wgb
        if wg_gbm < 0: continue
        bl = wgi*gp_int_looo + wgb*gp_looo + wg_gbm*gbm_looo
        mae = np.mean(np.abs(y - bl))
        tp_bl = wgi*gp_int_tp + wgb*gp_tp + wg_gbm*gbm_tp
        if mae < best3[0]:
            best3 = (mae, wgi, wgb, wg_gbm, tp_bl)
            print(f"  GP_int({wgi})+GP_base({wgb})+GBM({wg_gbm:.1f}): LOOO={mae:.4f}  max={tp_bl.max():.2f} ←")

print(f"\n  Overall best so far:")
print(f"  GP_int standalone: LOOO={mae_gp_int:.4f}")
print(f"  GP_base+GBM(0.1): LOOO=1.7139")
if best3[0] < 99:
    print(f"  3-way best: LOOO={best3[0]:.4f}")

# Save best GP_int model
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
    print(f"  {label:70s}: D={du:+.4f}  std={std:.3f}  [{mn:.2f},{mx:.2f}]")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return scaled

print(f"\n--- Saving GP_int best submissions ---")
save_corr(gp_int_tp, 5.5, f'GP_int_M52 standalone Δ=5.5 LOOO={mae_gp_int:.4f}',
          'FINAL_NEW_oN_gpIntM52_5p5_OOF8.3825.csv')

# Best GP_int + GBM blend
if best_blend[3] is not None:
    wgi, wg = best_blend[1], best_blend[2]
    save_corr(best_blend[3], 5.5, f'GP_int({wgi:.1f})+GBM({wg:.1f}) Δ=5.5 LOOO={best_blend[0]:.4f}',
              f'FINAL_NEW_oN_gpInt{int(wgi*10)}gbm{int(wg*10)}_5p5_OOF8.3825.csv')

print("\nDone.")
