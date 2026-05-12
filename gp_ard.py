import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel, RBF
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

# Precompute GBM
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_looo = []
for i in range(len(y)):
    m = GradientBoostingRegressor(**gbm_kw)
    m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
    gbm_looo.append(min(m.predict(X_tr[i:i+1])[0], p95))
gbm_looo = np.array(gbm_looo)
gbm_m = GradientBoostingRegressor(**gbm_kw); gbm_m.fit(X_tr, y)
gbm_tp = np.minimum(gbm_m.predict(X_te), p95)

print("="*70)
print("GP Kernel Comparison: isotropic vs ARD (4 features)")
print("="*70)

kernels = [
    ('Matern52_iso',    ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)),
    ('Matern52_ARD',    ConstantKernel(1.0)*Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(0.1)),
    ('RBF_iso',         ConstantKernel(1.0)*RBF(length_scale=1.0) + WhiteKernel(0.1)),
    ('RBF_ARD',         ConstantKernel(1.0)*RBF(length_scale=np.ones(4)) + WhiteKernel(0.1)),
    ('Matern32_ARD',    ConstantKernel(1.0)*Matern(length_scale=np.ones(4), nu=1.5) + WhiteKernel(0.1)),
]

best_looo, best_name, best_gp_looo = 99, '', None
for kname, kern in kernels:
    looo = []
    for i in range(len(y)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr, i, 0))
        Xte_s = sc_.transform(X_tr[i:i+1])
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=3)
        gp.fit(Xtr_s, np.delete(y, i))
        looo.append(gp.predict(Xte_s)[0])
    looo = np.array(looo)
    mae = np.mean(np.abs(y - looo))
    bl_looo = 0.1*gbm_looo + 0.9*looo
    bl_mae = np.mean(np.abs(y - bl_looo))

    sc_f = StandardScaler()
    gp_f = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=3)
    gp_f.fit(sc_f.fit_transform(X_tr), y)
    tp = gp_f.predict(sc_f.transform(X_te))
    blend_tp = 0.1*gbm_tp + 0.9*tp

    print(f"  {kname:20s}: LOOO={mae:.4f}  gbm1+gp9={bl_mae:.4f}  test_mean={blend_tp.mean():.3f}  max={tp.max():.2f}")
    if bl_mae < best_looo:
        best_looo = bl_mae; best_name = kname; best_gp_looo = looo; best_gp_tp = tp

print(f"\n  Best kernel for GBM(0.1)+GP(0.9): {best_name}, LOOO={best_looo:.4f}")

# Also check the optimized kernel's learned length scales
print(f"\n  Fitting final ARD Matern52 on all data to inspect length scales...")
kern_ard = ConstantKernel(1.0)*Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(0.1)
sc_f = StandardScaler()
gp_ard = GaussianProcessRegressor(kernel=kern_ard, normalize_y=True, n_restarts_optimizer=5)
gp_ard.fit(sc_f.fit_transform(X_tr), y)
print(f"  Learned kernel: {gp_ard.kernel_}")
# Extract length scales from the optimized kernel
try:
    l_scales = gp_ard.kernel_.k1.k2.length_scale
    feats = ['oof_mean', 'pu', 'tw', 'conv']
    for f, l in zip(feats, l_scales):
        print(f"    {f}: length_scale={l:.3f} (small = more important)")
except:
    print("  Could not extract per-feature length scales")

print("\nDone.")
