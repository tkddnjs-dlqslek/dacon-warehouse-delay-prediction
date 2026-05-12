import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
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

# Full aggregation with extra features
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'), resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'), tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'), inflow=('order_inflow_15m','mean'),
    agv=('agv_task_success_rate','mean'), pkg=('packaging_material_cost','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
    inflow=('order_inflow_15m','mean'), agv=('agv_task_success_rate','mean'),
    pkg=('packaging_material_cost','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

# Base features
X4_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X4_te = te_lv_u[['oN_mean','pu','tw','conv']].values

# Extended features
X7_tr = tr_lv[['oof_mean','pu','tw','conv','inflow','agv','pkg']].values
X7_te = te_lv_u[['oN_mean','pu','tw','conv','inflow','agv','pkg']].values

# Base + pu*oof interaction (current best for Ridge component)
pu_oof_tr = X4_tr[:,0] * X4_tr[:,1]
pu_oof_te = X4_te[:,0] * X4_te[:,1]
X_int_tr = np.column_stack([X4_tr, pu_oof_tr])
X_int_te = np.column_stack([X4_te, pu_oof_te])

p95 = np.percentile(y_lv, 95)

def looo_ridge(X_tr, X_te, y, alpha):
    preds_tr, preds_te_list = [], []
    for i in range(len(y)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr, i, 0))
        Xte_s = sc_.transform(X_tr[i:i+1])
        lr = Ridge(alpha=alpha); lr.fit(Xtr_s, np.delete(y, i))
        preds_tr.append(lr.predict(Xte_s)[0])
    sc_f = StandardScaler()
    X_tr_s = sc_f.fit_transform(X_tr); X_te_s = sc_f.transform(X_te)
    lr_f = Ridge(alpha=alpha); lr_f.fit(X_tr_s, y)
    return np.mean(np.abs(y - np.array(preds_tr))), lr_f.predict(X_te_s), np.array(preds_tr)

def looo_gbm(X_tr, X_te, y, **kw):
    preds_tr = []
    for i in range(len(y)):
        m = GradientBoostingRegressor(**kw)
        m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
        preds_tr.append(min(m.predict(X_tr[i:i+1])[0], p95))
    m_f = GradientBoostingRegressor(**kw); m_f.fit(X_tr, y)
    return np.mean(np.abs(y - np.array(preds_tr))), np.minimum(m_f.predict(X_te), p95), np.array(preds_tr)

print("="*70)
print("Extended Feature Analysis for Layout-Level Models")
print("="*70)

# Current best reference
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
mae_gbm_base, gbm_tp, gbm_looo = looo_gbm(X4_tr, X4_te, y_lv, **gbm_kw)
mae_ridgeint, ridgeint_tp, ridgeint_looo = looo_ridge(X_int_tr, X_int_te, y_lv, alpha=100)
blend_looo = 0.7*gbm_looo + 0.3*ridgeint_looo
mae_current_best = np.mean(np.abs(y_lv - blend_looo))
print(f"\n  CURRENT BEST: GBM_base(0.7)+Ridge_int(0.3): LOOO={mae_current_best:.4f}")

# 1. GBM with extra features (7 features)
print(f"\n--- GBM with Extended Features ---")
for extra_name, Xtr_, Xte_ in [
    ('7feat(+inflow+agv+pkg)', X7_tr, X7_te),
    ('5feat(+inflow)', np.column_stack([X4_tr, tr_lv['inflow'].values]),
                       np.column_stack([X4_te, te_lv_u['inflow'].values])),
    ('5feat(+agv)',    np.column_stack([X4_tr, tr_lv['agv'].values]),
                       np.column_stack([X4_te, te_lv_u['agv'].values])),
    ('5feat(+inflow)+oof*pu_int',
     np.column_stack([X4_tr, tr_lv['inflow'].values, pu_oof_tr]),
     np.column_stack([X4_te, te_lv_u['inflow'].values, pu_oof_te])),
]:
    mae, tp, looo = looo_gbm(Xtr_, Xte_, y_lv, **gbm_kw)
    # blend with current ridge_int
    bl = np.mean(np.abs(y_lv - (0.7*looo + 0.3*ridgeint_looo)))
    print(f"  GBM {extra_name:35s}: LOOO={mae:.4f}  blend_w7_ridgeInt={bl:.4f}  test_mean={tp.mean():.3f}")

# 2. GBM with oof*pu as explicit feature
oof_pu_tr = X4_tr[:,0]*X4_tr[:,1]
oof_pu_te = X4_te[:,0]*X4_te[:,1]
X4int_tr = np.column_stack([X4_tr, oof_pu_tr])
X4int_te = np.column_stack([X4_te, oof_pu_te])
mae_gbmint, gbmint_tp, gbmint_looo = looo_gbm(X4int_tr, X4int_te, y_lv, **gbm_kw)
bl = np.mean(np.abs(y_lv - (0.7*gbmint_looo + 0.3*ridgeint_looo)))
print(f"\n  GBM with oof*pu interaction feat: LOOO={mae_gbmint:.4f}  blend_w7_ridgeInt={bl:.4f}")

# 3. GBM d=3,n=50 with extended features
gbm50_kw = dict(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=42)
print(f"\n--- GBM d=3,n=50 with Extended Features ---")
for extra_name, Xtr_, Xte_ in [
    ('7feat(+inflow+agv+pkg)', X7_tr, X7_te),
    ('5feat(+inflow)', np.column_stack([X4_tr, tr_lv['inflow'].values]),
                       np.column_stack([X4_te, te_lv_u['inflow'].values])),
]:
    mae, tp, looo = looo_gbm(Xtr_, Xte_, y_lv, **gbm50_kw)
    bl = np.mean(np.abs(y_lv - (0.7*looo + 0.3*ridgeint_looo)))
    print(f"  GBM50 {extra_name:35s}: LOOO={mae:.4f}  blend_w7_ridgeInt={bl:.4f}")

# 4. Ridge with extended features + interaction
print(f"\n--- Ridge with Extended Features ---")
pu_oof_tr7 = X7_tr[:,0]*X7_tr[:,1]
pu_oof_te7 = X7_te[:,0]*X7_te[:,1]
X7int_tr = np.column_stack([X7_tr, pu_oof_tr7])
X7int_te = np.column_stack([X7_te, pu_oof_te7])
for aname, Xtr_, Xte_, a in [
    ('7feat+int a=100', X7int_tr, X7int_te, 100),
    ('7feat+int a=200', X7int_tr, X7int_te, 200),
    ('7feat a=100',     X7_tr,    X7_te,    100),
    ('7feat a=200',     X7_tr,    X7_te,    200),
]:
    mae, tp, looo = looo_ridge(Xtr_, Xte_, y_lv, a)
    bl = np.mean(np.abs(y_lv - (0.7*gbm_looo + 0.3*looo)))
    print(f"  Ridge {aname:30s}: LOOO={mae:.4f}  gbm_base(0.7)+this(0.3)={bl:.4f}")

# 5. 3-way ensemble: GBM_base + GBM_d3n50 + Ridge_int
print(f"\n--- 3-way Ensemble ---")
mae_gbm50, gbm50_tp, gbm50_looo = looo_gbm(X4_tr, X4_te, y_lv, **gbm50_kw)
for w1, w2, w3 in [(0.4,0.3,0.3),(0.35,0.35,0.3),(0.5,0.2,0.3),(0.45,0.25,0.3)]:
    bl = w1*gbm_looo + w2*gbm50_looo + w3*ridgeint_looo
    mae = np.mean(np.abs(y_lv - bl))
    tp = w1*gbm_tp + w2*gbm50_tp + w3*ridgeint_tp
    print(f"  GBM30({w1})+GBM50({w2})+RidgeInt({w3}): LOOO={mae:.4f}  test_mean={tp.mean():.3f}")

print("\nDone.")
