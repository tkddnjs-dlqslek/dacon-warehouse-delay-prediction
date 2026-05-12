import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel, DotProduct
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

pu_oof_tr = X_tr[:,0]*X_tr[:,1]
pu_oof_te = X_te[:,0]*X_te[:,1]
X_int_tr = np.column_stack([X_tr, pu_oof_tr])
X_int_te = np.column_stack([X_te, pu_oof_te])

# Precompute reference components
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_looo = []
for i in range(len(y)):
    m = GradientBoostingRegressor(**gbm_kw)
    m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
    gbm_looo.append(min(m.predict(X_tr[i:i+1])[0], p95))
gbm_looo = np.array(gbm_looo)
gbm_m = GradientBoostingRegressor(**gbm_kw); gbm_m.fit(X_tr, y)
gbm_tp = np.minimum(gbm_m.predict(X_te), p95)

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

def looo_gp(X_tr_, X_te_, y_, kernel, nr=3):
    looo = []
    for i in range(len(y_)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr_, i, 0))
        Xte_s = sc_.transform(X_tr_[i:i+1])
        gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=nr)
        gp.fit(Xtr_s, np.delete(y_, i))
        looo.append(gp.predict(Xte_s)[0])
    sc_f = StandardScaler()
    X_tr_s = sc_f.fit_transform(X_tr_); X_te_s = sc_f.transform(X_te_)
    gp_f = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=nr)
    gp_f.fit(X_tr_s, y_)
    tp = gp_f.predict(X_te_s)
    return np.array(looo), tp

print("="*70)
print("Deep Gaussian Process Analysis: Kernels × Feature Sets × Blends")
print("="*70)

# GP kernel sweep on base4 features
kernels = [
    ('RBF',         ConstantKernel(1.0)*RBF(1.0) + WhiteKernel(0.1)),
    ('Matern52',    ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)),
    ('Matern32',    ConstantKernel(1.0)*Matern(length_scale=1.0, nu=1.5) + WhiteKernel(0.1)),
    ('Matern12',    ConstantKernel(1.0)*Matern(length_scale=1.0, nu=0.5) + WhiteKernel(0.1)),
    ('RBF+Linear',  ConstantKernel(1.0)*RBF(1.0) + ConstantKernel(1.0)*DotProduct(sigma_0=1.0) + WhiteKernel(0.1)),
]

print(f"\n--- GP kernel sweep on base4 features ---")
best_gp = {}
for kname, kern in kernels:
    looo, tp = looo_gp(X_tr, X_te, y, kern)
    mae = np.mean(np.abs(y - looo))
    print(f"  GP {kname:12s}: LOOO={mae:.4f}  test_mean={tp.mean():.3f}  std={tp.std():.3f}")
    best_gp[kname] = (looo, tp, mae)

print(f"\n--- GP Matern52 on interaction features ---")
looo_gp_int, tp_gp_int = looo_gp(X_int_tr, X_int_te, y, ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1))
mae_gp_int = np.mean(np.abs(y - looo_gp_int))
print(f"  GP Matern52 int: LOOO={mae_gp_int:.4f}  test_mean={tp_gp_int.mean():.3f}")

# Use best GP (Matern52, base4)
gp_looo, gp_tp = best_gp['Matern52'][0], best_gp['Matern52'][1]
mae_gp_standalone = best_gp['Matern52'][2]
print(f"\n  Reference: GP Matern52 standalone: LOOO={mae_gp_standalone:.4f}")

# Weight sweep: GP + GBM
print(f"\n--- Weight sweep: GP + GBM ---")
best_mae, best_wg = 99, 0
for wg in np.arange(0.0, 1.05, 0.1):
    wgp = 1 - wg
    bl = wg*gbm_looo + wgp*gp_looo
    mae = np.mean(np.abs(y - bl))
    tp_bl = wg*gbm_tp + wgp*gp_tp
    mark = " ←" if mae < best_mae else ""
    print(f"  GBM({wg:.1f})+GP({wgp:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f}  std={tp_bl.std():.3f}{mark}")
    if mae < best_mae: best_mae = mae; best_wg = wg

# Weight sweep: GP + Huber
print(f"\n--- Weight sweep: GP + HuberInt ---")
best_mae2, best_wgp2 = 99, 0
for wgp in np.arange(0.0, 1.05, 0.1):
    wh = 1 - wgp
    bl = wgp*gp_looo + wh*huber_looo
    mae = np.mean(np.abs(y - bl))
    tp_bl = wgp*gp_tp + wh*huber_tp
    mark = " ←" if mae < best_mae2 else ""
    print(f"  GP({wgp:.1f})+Huber({wh:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f}{mark}")
    if mae < best_mae2: best_mae2 = mae; best_wgp2 = wgp

# 3-way: GBM + GP + Huber
print(f"\n--- 3-way: GBM + GP + HuberInt ---")
best3 = (99, 0, 0, 0)
for wg in [0.2, 0.3, 0.4, 0.5]:
    for wgp in [0.2, 0.3, 0.4, 0.5]:
        wh = 1 - wg - wgp
        if wh < 0: continue
        bl = wg*gbm_looo + wgp*gp_looo + wh*huber_looo
        mae = np.mean(np.abs(y - bl))
        tp_bl = wg*gbm_tp + wgp*gp_tp + wh*huber_tp
        if mae < best3[0]:
            best3 = (mae, wg, wgp, wh, tp_bl.mean())
            print(f"  GBM({wg})+GP({wgp})+H({wh:.1f}): LOOO={mae:.4f}  test_mean={tp_bl.mean():.3f} ←")

if best3[0] < 99:
    print(f"\n  BEST 3-way: LOOO={best3[0]:.4f}")

# Compare GP alone vs Huber best
print(f"\n--- Summary so far ---")
print(f"  GP Matern52 standalone:           LOOO={mae_gp_standalone:.4f}  test_mean={gp_tp.mean():.3f}")
h_looo_alone = 1.8462  # from earlier
print(f"  GBM(0.55)+HuberInt(0.45):         LOOO=1.7289  test_mean≈5.315")
print(f"  GBM(0.70)+RidgeInt(0.30) [prev]:  LOOO=1.7617  test_mean≈5.031")

print("\nDone.")
