import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
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

# Build layout-level features
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'),
    resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    barcode=('barcode_read_success_rate','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    barcode=('barcode_read_success_rate','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

feat_sets = {
    '3phys':         (['pu','tw','conv'], ['pu','tw','conv']),
    '5phys':         (['pu','tw','conv','pkg','agv'], ['pu','tw','conv','pkg','agv']),
    'oof+3phys':     (['oof_mean','pu','tw','conv'], ['oN_mean','pu','tw','conv']),
    'oof+5phys':     (['oof_mean','pu','tw','conv','pkg','agv'], ['oN_mean','pu','tw','conv','pkg','agv']),
    'oof+5phys+bar': (['oof_mean','pu','tw','conv','pkg','agv','barcode'],
                      ['oN_mean','pu','tw','conv','pkg','agv','barcode']),
    'oof_sq+3phys':  (None, None),  # special: add oof^2
}

print("="*70)
print("Non-linear Layout Corrections: LOOO MAE Comparison")
print("="*70)

import warnings
warnings.filterwarnings('ignore')

def looo_predict(model_cls, model_kw, X_tr, X_te, y, scale=False):
    preds = []
    for i in range(len(y)):
        Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y, i)
        if scale:
            sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_)
            Xte_s = sc_.transform(X_tr[i:i+1])
        else:
            Xtr_s = Xtr_; Xte_s = X_tr[i:i+1]
        m = model_cls(**model_kw); m.fit(Xtr_s, ytr_)
        preds.append(m.predict(Xte_s)[0])
    looo_mae = np.mean(np.abs(y - np.array(preds)))

    if scale:
        sc_f = StandardScaler(); X_tr_s = sc_f.fit_transform(X_tr); X_te_s = sc_f.transform(X_te)
    else:
        X_tr_s = X_tr; X_te_s = X_te
    m_f = model_cls(**model_kw); m_f.fit(X_tr_s, y)
    test_preds = m_f.predict(X_te_s)
    return looo_mae, test_preds

print(f"\n  Model comparison (oof+3phys features):")
feats_tr = tr_lv[['oof_mean','pu','tw','conv']].values
feats_te = te_lv_u[['oN_mean','pu','tw','conv']].values

models = [
    ('Ridge a100',    Ridge,                  {'alpha':100},   True),
    ('Ridge a50',     Ridge,                  {'alpha':50},    True),
    ('RF(n=100,d=3)', RandomForestRegressor,  {'n_estimators':100,'max_depth':3,'random_state':42}, False),
    ('RF(n=100,d=4)', RandomForestRegressor,  {'n_estimators':100,'max_depth':4,'random_state':42}, False),
    ('RF(n=100,d=2)', RandomForestRegressor,  {'n_estimators':100,'max_depth':2,'random_state':42}, False),
    ('GBM(d=2,n=50)', GradientBoostingRegressor, {'max_depth':2,'n_estimators':50,'learning_rate':0.1,'random_state':42}, False),
    ('GBM(d=3,n=30)', GradientBoostingRegressor, {'max_depth':3,'n_estimators':30,'learning_rate':0.1,'random_state':42}, False),
    ('DT(d=3)',       DecisionTreeRegressor,  {'max_depth':3,'random_state':42},  False),
]

best_mae = 999; best_preds = None; best_name = ''
for name, cls, kw, sc in models:
    mae, tp = looo_predict(cls, kw, feats_tr, feats_te, y_lv, scale=sc)
    print(f"  {name:25s}: LOOO_MAE={mae:.4f}  test_mean={tp.mean():.3f}  std={tp.std():.3f}")
    if mae < best_mae:
        best_mae = mae; best_preds = tp; best_name = name

# Also with 5-phys
print(f"\n  With oof+5phys features:")
feats5_tr = tr_lv[['oof_mean','pu','tw','conv','pkg','agv']].values
feats5_te = te_lv_u[['oN_mean','pu','tw','conv','pkg','agv']].values
for name, cls, kw, sc in [
    ('Ridge a100', Ridge, {'alpha':100}, True),
    ('RF(n=100,d=3)', RandomForestRegressor, {'n_estimators':100,'max_depth':3,'random_state':42}, False),
    ('GBM(d=2,n=50)', GradientBoostingRegressor, {'max_depth':2,'n_estimators':50,'learning_rate':0.1,'random_state':42}, False),
]:
    mae, tp = looo_predict(cls, kw, feats5_tr, feats5_te, y_lv, scale=sc)
    print(f"  {name:25s}: LOOO_MAE={mae:.4f}  test_mean={tp.mean():.3f}  std={tp.std():.3f}")
    if mae < best_mae:
        best_mae = mae; best_preds = tp; best_name = f"5phys:{name}"

# Interaction features: oof_mean * pack_util
print(f"\n  With interaction features (oof × pu, oof × tw):")
oof_tr = tr_lv['oof_mean'].values
pu_tr = tr_lv['pu'].values
feats_int_tr = np.column_stack([
    tr_lv[['oof_mean','pu','tw','conv']].values,
    oof_tr * pu_tr,
    oof_tr * tr_lv['tw'].values
])
oof_te = te_lv_u['oN_mean'].values
pu_te = te_lv_u['pu'].values
feats_int_te = np.column_stack([
    te_lv_u[['oN_mean','pu','tw','conv']].values,
    oof_te * pu_te,
    oof_te * te_lv_u['tw'].values
])
for alpha in [50, 100, 200]:
    mae, tp = looo_predict(Ridge, {'alpha':alpha}, feats_int_tr, feats_int_te, y_lv, scale=True)
    print(f"  Ridge+interaction a{alpha:5d}:    LOOO_MAE={mae:.4f}  test_mean={tp.mean():.3f}  std={tp.std():.3f}")
    if mae < best_mae:
        best_mae = mae; best_preds = tp; best_name = f"interaction a{alpha}"

# Save best non-linear model
print(f"\n  Best model: {best_name} (LOOO_MAE={best_mae:.4f})")
lid_to_best = dict(zip(te_lv_u['layout_id'], best_preds))
offset_best = 5.5 - best_preds.mean()
corr_best = np.zeros(len(test_raw))
for lid, c in lid_to_best.items():
    corr_best[(test_raw['layout_id'] == lid).values] = c + offset_best
ct_best = oracle_new_t.copy()
ct_best[unseen_mask] = oracle_new_t[unseen_mask] + corr_best[unseen_mask]
ct_best = np.clip(ct_best, 0, None)
du_best = ct_best[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Saved: D={du_best:+.4f}  unseen={ct_best[unseen_mask].mean():.3f}  "
      f"std={corr_best[unseen_mask].std():.3f}")
sub_best = sub_tmpl.copy(); sub_best['avg_delay_minutes_next_30m'] = ct_best
sub_best.to_csv(f'FINAL_NEW_oN_bestNonlinear_OOF8.3825.csv', index=False)

# Also save all models with Δ=5.5
print(f"\n  Saving RF(d=3) and GBM(d=2) corrections at Δ=5.5:")
for name, cls, kw, sc in [
    ('RF_d3', RandomForestRegressor, {'n_estimators':200,'max_depth':3,'random_state':42}, False),
    ('GBM_d2', GradientBoostingRegressor, {'max_depth':2,'n_estimators':100,'learning_rate':0.05,'random_state':42}, False),
]:
    mae, tp = looo_predict(cls, kw, feats_tr, feats_te, y_lv, scale=sc)
    offset_ = 5.5 - tp.mean()
    corr_ = np.zeros(len(test_raw))
    for lid, c in dict(zip(te_lv_u['layout_id'], tp)).items():
        corr_[(test_raw['layout_id'] == lid).values] = c + offset_
    ct_ = oracle_new_t.copy()
    ct_[unseen_mask] = oracle_new_t[unseen_mask] + corr_[unseen_mask]
    ct_ = np.clip(ct_, 0, None)
    du_ = ct_[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {name}: LOOO={mae:.4f}  D={du_:+.4f}  std={corr_[unseen_mask].std():.3f}")
    sub_ = sub_tmpl.copy(); sub_['avg_delay_minutes_next_30m'] = ct_
    sub_.to_csv(f'FINAL_NEW_oN_{name}_5p5_OOF8.3825.csv', index=False)

print("\nDone.")
