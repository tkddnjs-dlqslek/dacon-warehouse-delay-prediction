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
    conv=('conveyor_speed_mps','mean'), pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'), agv=('agv_task_success_rate','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

X_base_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_base_te = te_lv_u[['oN_mean','pu','tw','conv']].values

pu_oof_tr = X_base_tr[:,0] * X_base_tr[:,1]
pu_oof_te = X_base_te[:,0] * X_base_te[:,1]
X_int_tr = np.column_stack([X_base_tr, pu_oof_tr])
X_int_te = np.column_stack([X_base_te, pu_oof_te])

import warnings; warnings.filterwarnings('ignore')
p95 = np.percentile(y_lv, 95)

print("="*70)
print("GBM Hyperparameter Search + XGBoost at Layout Level")
print("="*70)

# Reference: GBM(d=3,n=30) = 1.7993
# Best blend so far: GBM_cap(0.7)+Ridge_int(0.3) = 1.7617

# GBM variants
gbm_configs = [
    ('GBM d=3,n=30,lr=0.1',  dict(max_depth=3, n_estimators=30,  learning_rate=0.1,  random_state=42)),
    ('GBM d=4,n=30,lr=0.1',  dict(max_depth=4, n_estimators=30,  learning_rate=0.1,  random_state=42)),
    ('GBM d=4,n=50,lr=0.05', dict(max_depth=4, n_estimators=50,  learning_rate=0.05, random_state=42)),
    ('GBM d=3,n=50,lr=0.05', dict(max_depth=3, n_estimators=50,  learning_rate=0.05, random_state=42)),
    ('GBM d=3,n=100,lr=0.02',dict(max_depth=3, n_estimators=100, learning_rate=0.02, random_state=42)),
    ('GBM d=2,n=50,lr=0.1',  dict(max_depth=2, n_estimators=50,  learning_rate=0.1,  random_state=42)),
    ('GBM d=5,n=20,lr=0.1',  dict(max_depth=5, n_estimators=20,  learning_rate=0.1,  random_state=42)),
]

print(f"\n  GBM hyperparameter sweep (oof+phys3 features):")
gbm_results = []
for name, kw in gbm_configs:
    preds_looo = []
    for i in range(len(y_lv)):
        Xtr_ = np.delete(X_base_tr, i, 0); ytr_ = np.delete(y_lv, i)
        m = GradientBoostingRegressor(**kw); m.fit(Xtr_, ytr_)
        preds_looo.append(m.predict(X_base_tr[i:i+1])[0])
    gbm_m = GradientBoostingRegressor(**kw); gbm_m.fit(X_base_tr, y_lv)
    test_preds = gbm_m.predict(X_base_te)
    mae = np.mean(np.abs(y_lv - np.array(preds_looo)))
    print(f"  {name:30s}: LOOO={mae:.4f}  test_mean={test_preds.mean():.3f}  std={test_preds.std():.3f}")
    gbm_results.append((mae, test_preds, name))

# Also try XGBoost if available
try:
    import xgboost as xgb
    print(f"\n  XGBoost variants:")
    xgb_configs = [
        ('XGB d=3,n=30,lr=0.1',  dict(max_depth=3, n_estimators=30,  learning_rate=0.1, random_state=42, verbosity=0)),
        ('XGB d=4,n=30,lr=0.1',  dict(max_depth=4, n_estimators=30,  learning_rate=0.1, random_state=42, verbosity=0)),
        ('XGB d=3,n=50,lr=0.05', dict(max_depth=3, n_estimators=50,  learning_rate=0.05,random_state=42, verbosity=0)),
        ('XGB d=4,n=50,lr=0.05', dict(max_depth=4, n_estimators=50,  learning_rate=0.05,random_state=42, verbosity=0)),
        ('XGB d=5,n=30,lr=0.05', dict(max_depth=5, n_estimators=30,  learning_rate=0.05,random_state=42, verbosity=0)),
    ]
    for name, kw in xgb_configs:
        preds_looo = []
        for i in range(len(y_lv)):
            Xtr_ = np.delete(X_base_tr, i, 0); ytr_ = np.delete(y_lv, i)
            m = xgb.XGBRegressor(**kw); m.fit(Xtr_, ytr_)
            preds_looo.append(m.predict(X_base_tr[i:i+1])[0])
        m_f = xgb.XGBRegressor(**kw); m_f.fit(X_base_tr, y_lv)
        test_preds = m_f.predict(X_base_te)
        mae = np.mean(np.abs(y_lv - np.array(preds_looo)))
        print(f"  {name:30s}: LOOO={mae:.4f}  test_mean={test_preds.mean():.3f}  std={test_preds.std():.3f}")
        gbm_results.append((mae, test_preds, name))
except ImportError:
    print(f"\n  XGBoost not available")

# Find best single model
gbm_results.sort(key=lambda x: x[0])
best_single = gbm_results[0]
print(f"\n  Best single tree model: {best_single[2]}: LOOO={best_single[0]:.4f}")

# Best blend: GBM_best_cap(0.7) + Ridge_int(0.3)
sc_int = StandardScaler()
X_int_tr_s = sc_int.fit_transform(X_int_tr); X_int_te_s = sc_int.transform(X_int_te)
ridge_int = Ridge(alpha=100); ridge_int.fit(X_int_tr_s, y_lv)
ridge_int_test = ridge_int.predict(X_int_te_s)
ridge_int_looo = []
for i in range(len(y_lv)):
    sc_ = StandardScaler()
    X_tr_ = sc_.fit_transform(np.delete(X_int_tr, i, 0))
    X_te_ = sc_.transform(X_int_tr[i:i+1])
    m = Ridge(alpha=100); m.fit(X_tr_, np.delete(y_lv, i))
    ridge_int_looo.append(m.predict(X_te_)[0])
ridge_int_looo = np.array(ridge_int_looo)

print(f"\n  Best blend with best GBM + Ridge_int:")
for mae, gbm_tp, gname in gbm_results[:5]:
    # Fit the model fresh for test
    gkw = dict(gbm_configs[[c[0] for c in gbm_configs].index(gname)][1]
               if gname in [c[0] for c in gbm_configs] else gbm_configs[0][1])

# Directly use the precomputed test predictions and LOOO
gbm_best_tp = gbm_results[0][1]
gbm_best_name = gbm_results[0][2]
gbm_best_capped = np.minimum(gbm_best_tp, p95)

# Recompute LOOO for best GBM
best_gbm_kw = None
for name, kw in gbm_configs:
    if name == gbm_best_name:
        best_gbm_kw = kw
        break
if best_gbm_kw is None:
    best_gbm_kw = gbm_configs[0][1]  # fallback

gbm_best_looo = []
for i in range(len(y_lv)):
    Xtr_ = np.delete(X_base_tr, i, 0); ytr_ = np.delete(y_lv, i)
    m = GradientBoostingRegressor(**best_gbm_kw); m.fit(Xtr_, ytr_)
    gbm_best_looo.append(min(m.predict(X_base_tr[i:i+1])[0], p95))
gbm_best_looo = np.array(gbm_best_looo)

for wg in [0.5, 0.7, 0.8]:
    wr = 1 - wg
    ens_looo = wg * gbm_best_looo + wr * ridge_int_looo
    mae = np.mean(np.abs(y_lv - ens_looo))
    ens_test = wg * gbm_best_capped + wr * ridge_int_test
    print(f"  best_GBM_cap({wg:.1f})+Ridge_int({wr:.1f}): LOOO={mae:.4f}  "
          f"test_mean={ens_test.mean():.3f}  std={ens_test.std():.3f}")

print("\nDone.")
