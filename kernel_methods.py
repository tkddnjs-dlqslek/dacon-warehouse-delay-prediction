import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
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

X_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_te = te_lv_u[['oN_mean','pu','tw','conv']].values
y = y_lv
p95 = np.percentile(y, 95)

# Interaction term
pu_oof_tr = X_tr[:,0]*X_tr[:,1]
pu_oof_te = X_te[:,0]*X_te[:,1]
X_int_tr = np.column_stack([X_tr, pu_oof_tr])
X_int_te = np.column_stack([X_te, pu_oof_te])

# Reference baselines
def looo_model(X_tr, X_te, y, model_fn, cap=None, scale=True):
    looo = []
    for i in range(len(y)):
        Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y, i)
        if scale:
            sc = StandardScaler(); Xtr_s = sc.fit_transform(Xtr_)
            Xte_s = sc.transform(X_tr[i:i+1])
        else:
            Xtr_s, Xte_s = Xtr_, X_tr[i:i+1]
        m = model_fn(); m.fit(Xtr_s, ytr_)
        p = m.predict(Xte_s)[0]
        if cap: p = min(p, cap)
        looo.append(p)
    if scale:
        sc_f = StandardScaler(); X_tr_s = sc_f.fit_transform(X_tr); X_te_s = sc_f.transform(X_te)
    else:
        X_tr_s, X_te_s = X_tr, X_te
    m_f = model_fn(); m_f.fit(X_tr_s, y)
    tp = m_f.predict(X_te_s)
    if cap is not None: tp = np.minimum(tp, cap)
    return np.mean(np.abs(y - np.array(looo))), tp, np.array(looo)

# Pre-compute reference models
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
_, gbm_tp, gbm_looo = looo_model(X_tr, X_te, y, lambda: GradientBoostingRegressor(**gbm_kw), cap=p95, scale=False)

sc_ri = StandardScaler(); X_int_tr_s = sc_ri.fit_transform(X_int_tr)
ri_looo = []
for i in range(len(y)):
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(X_int_tr, i, 0))
    Xte_s = sc_.transform(X_int_tr[i:i+1])
    m = Ridge(alpha=100); m.fit(Xtr_s, np.delete(y, i))
    ri_looo.append(m.predict(Xte_s)[0])
ri_looo = np.array(ri_looo)
ri_f = Ridge(alpha=100); ri_f.fit(sc_ri.transform(X_int_tr), y)
ridgeint_tp = ri_f.predict(sc_ri.transform(X_int_te))
mae_current = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*ri_looo)))
print(f"Current best (GBM(0.7)+RidgeInt(0.3)): LOOO={mae_current:.4f}")

print("="*70)
print("Kernel / Non-Parametric Methods at Layout Level (n=250)")
print("="*70)

# SVR variants
print(f"\n--- SVR ---")
for C, gamma, eps in [(1,0.1,0.1),(1,'scale',0.1),(10,'scale',0.05),(10,0.5,0.05),(5,0.2,0.1)]:
    kw = dict(C=C, gamma=gamma, epsilon=eps)
    mae, tp, looo = looo_model(X_tr, X_te, y, lambda: SVR(**kw), cap=p95)
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    print(f"  SVR C={C},g={gamma},e={eps}: LOOO={mae:.4f}  blend_gbm70={bl:.4f}  test_mean={tp.mean():.3f}")

# Gaussian Process
print(f"\n--- Gaussian Process ---")
kernels = [
    ('RBF', ConstantKernel(1.0)*RBF(1.0) + WhiteKernel(0.1)),
    ('Matern32', ConstantKernel(1.0)*Matern(length_scale=1.0, nu=1.5) + WhiteKernel(0.1)),
    ('Matern52', ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)),
]
for kname, kern in kernels:
    def make_gp(k=kern): return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2)
    mae, tp, looo = looo_model(X_tr, X_te, y, make_gp)
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    print(f"  GP {kname:10s}: LOOO={mae:.4f}  blend_gbm70={bl:.4f}  test_mean={tp.mean():.3f}")

# ExtraTrees
print(f"\n--- ExtraTrees ---")
for d, n in [(3,100),(4,200),(3,200),(5,100)]:
    kw = dict(max_depth=d, n_estimators=n, random_state=42)
    mae, tp, looo = looo_model(X_tr, X_te, y, lambda: ExtraTreesRegressor(**kw), cap=p95, scale=False)
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    print(f"  ET d={d},n={n}: LOOO={mae:.4f}  blend_gbm70={bl:.4f}  test_mean={tp.mean():.3f}")

# Kernel Ridge Regression
from sklearn.kernel_ridge import KernelRidge
print(f"\n--- Kernel Ridge ---")
for alpha, gamma in [(0.1,0.1),(1.0,0.1),(0.5,0.5),(1.0,1.0),(0.1,1.0)]:
    kw = dict(kernel='rbf', alpha=alpha, gamma=gamma)
    mae, tp, looo = looo_model(X_tr, X_te, y, lambda: KernelRidge(**kw))
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    print(f"  KRR a={alpha},g={gamma}: LOOO={mae:.4f}  blend_gbm70={bl:.4f}  test_mean={tp.mean():.3f}")

print("\nDone.")
