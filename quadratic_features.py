import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
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
    conv=('conveyor_speed_mps','mean'), inflow=('order_inflow_15m','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
    inflow=('order_inflow_15m','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

X_base_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_base_te = te_lv_u[['oN_mean','pu','tw','conv']].values
y = y_lv

import warnings; warnings.filterwarnings('ignore')

def looo_mae_scale(X_tr, X_te, y, alpha):
    preds = []
    for i in range(len(y)):
        Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y, i)
        sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_); Xte_s = sc_.transform(X_tr[i:i+1])
        lr = Ridge(alpha=alpha); lr.fit(Xtr_s, ytr_)
        preds.append(lr.predict(Xte_s)[0])
    looo_mae = np.mean(np.abs(y - np.array(preds)))
    sc_f = StandardScaler(); X_tr_s = sc_f.fit_transform(X_tr); X_te_s = sc_f.transform(X_te)
    lr_f = Ridge(alpha=alpha); lr_f.fit(X_tr_s, y)
    return looo_mae, lr_f.predict(X_te_s)

print("="*70)
print("Feature Engineering: Quadratic and Interaction Terms")
print("="*70)

# Baseline
mae_base, _ = looo_mae_scale(X_base_tr, X_base_te, y, alpha=100)
print(f"\n  Baseline (oof+phys3, a100): LOOO_MAE={mae_base:.4f}")

# Build various feature sets
feature_configs = []

# 1. Add pu^2
X_tr1 = np.column_stack([X_base_tr, X_base_tr[:,1]**2])
X_te1 = np.column_stack([X_base_te, X_base_te[:,1]**2])
for a in [50, 100, 200]:
    mae, preds = looo_mae_scale(X_tr1, X_te1, y, a)
    feature_configs.append(('oof+pu+pu^2+tw+conv', a, mae, preds))
    print(f"  oof+pu+pu^2+tw+conv a={a}: LOOO_MAE={mae:.4f}")

# 2. Add pu*oof interaction
pu_oof_tr = X_base_tr[:,0] * X_base_tr[:,1]
pu_oof_te = X_base_te[:,0] * X_base_te[:,1]
X_tr2 = np.column_stack([X_base_tr, pu_oof_tr])
X_te2 = np.column_stack([X_base_te, pu_oof_te])
for a in [50, 100, 200]:
    mae, preds = looo_mae_scale(X_tr2, X_te2, y, a)
    feature_configs.append(('oof+phys3+oof*pu', a, mae, preds))
    print(f"  oof+phys3+oof*pu a={a}: LOOO_MAE={mae:.4f}")

# 3. PolynomialFeatures degree=2 on (oof, pu) only
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_pu_oof_tr = tr_lv[['oof_mean','pu']].values
X_pu_oof_te = te_lv_u[['oN_mean','pu']].values
X_poly_tr = poly.fit_transform(X_pu_oof_tr)
X_poly_te = poly.transform(X_pu_oof_te)
# Combine with tw, conv
X_tr3 = np.column_stack([X_poly_tr, X_base_tr[:,2:]])
X_te3 = np.column_stack([X_poly_te, X_base_te[:,2:]])
for a in [100, 200, 500]:
    mae, preds = looo_mae_scale(X_tr3, X_te3, y, a)
    feature_configs.append(('poly2(oof,pu)+tw+conv', a, mae, preds))
    print(f"  poly2(oof,pu)+tw+conv a={a}: LOOO_MAE={mae:.4f}")

# 4. Full poly2 on all 4 features
poly4 = PolynomialFeatures(degree=2, include_bias=False)
X_poly4_tr = poly4.fit_transform(X_base_tr)
X_poly4_te = poly4.transform(X_base_te)
for a in [200, 500, 1000]:
    mae, preds = looo_mae_scale(X_poly4_tr, X_poly4_te, y, a)
    feature_configs.append(('poly2_all4', a, mae, preds))
    print(f"  poly2_all4 a={a}: LOOO_MAE={mae:.4f}")

# 5. oof+pu+tw+conv + pu*tw + pu*oof
pu_tw_tr = X_base_tr[:,1] * X_base_tr[:,2]
pu_tw_te = X_base_te[:,1] * X_base_te[:,2]
X_tr5 = np.column_stack([X_base_tr, pu_oof_tr, pu_tw_tr])
X_te5 = np.column_stack([X_base_te, pu_oof_te, pu_tw_te])
for a in [100, 200]:
    mae, preds = looo_mae_scale(X_tr5, X_te5, y, a)
    feature_configs.append(('oof+phys3+oof*pu+pu*tw', a, mae, preds))
    print(f"  oof+phys3+oof*pu+pu*tw a={a}: LOOO_MAE={mae:.4f}")

# Best config
best = min(feature_configs, key=lambda x: x[2])
print(f"\n  Best config: {best[0]} a={best[1]}: LOOO_MAE={best[2]:.4f}")

# Save best if better than GBM
gbm = GradientBoostingRegressor(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm.fit(X_base_tr, y)
gbm_test = gbm.predict(X_base_te)
p95 = np.percentile(y, 95)
gbm_capped = np.minimum(gbm_test, p95)
ridge_a100 = _  # from last looo_mae_scale
sc_r = StandardScaler(); ridge_a100 = Ridge(alpha=100)
sc_r.fit(X_base_tr); ridge_a100.fit(sc_r.transform(X_base_tr), y)
ridge_test = ridge_a100.predict(sc_r.transform(X_base_te))
blend_best = 0.7 * gbm_capped + 0.3 * ridge_test

if best[2] < 1.7778:
    print(f"  New best polynomial model beats GBM ensemble (1.7778)!")
    best_preds = best[3]
    offset = 5.5 - best_preds.mean()
    corr_arr = np.zeros(len(test_raw))
    for lid, c in zip(te_lv_u['layout_id'], best_preds + offset):
        corr_arr[(test_raw['layout_id'] == lid).values] = c
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  New best: D={du:+.4f}  std={corr_arr[unseen_mask].std():.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(f'FINAL_NEW_oN_polyBest_5p5_OOF8.3825.csv', index=False)
else:
    print(f"  No polynomial model beats GBM ensemble. GBM remains best.")

# Also check if blending best polynomial with GBM helps
print(f"\n  Blending polynomial best with GBM:")
_, poly_preds = looo_mae_scale(
    np.column_stack([X_base_tr, pu_oof_tr]),
    np.column_stack([X_base_te, pu_oof_te]),
    y, alpha=100)
for wp in [0.3, 0.5, 0.7]:
    wg = 1 - wp
    blend = wp * poly_preds + wg * gbm_capped
    blend_looo = []
    poly_looo = []
    for i in range(len(y)):
        Xtr_ = np.delete(np.column_stack([X_base_tr, pu_oof_tr]), i, 0)
        ytr_ = np.delete(y, i)
        sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_); Xte_s = sc_.transform(
            np.column_stack([X_base_tr[i:i+1], pu_oof_tr[i:i+1]]))
        lr = Ridge(alpha=100); lr.fit(Xtr_s, ytr_)
        p_pred = lr.predict(Xte_s)[0]

        Xgbtr_ = np.delete(X_base_tr, i, 0); ygtr_ = np.delete(y, i)
        gm = GradientBoostingRegressor(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
        gm.fit(Xgbtr_, ygtr_)
        g_pred = min(gm.predict(X_base_tr[i:i+1])[0], p95)
        blend_looo.append(wp * p_pred + wg * g_pred)
    mae_blend = np.mean(np.abs(y - np.array(blend_looo)))
    print(f"  poly_oof_pu({wp:.1f}) + GBM_cap({wg:.1f}): LOOO={mae_blend:.4f}")

print("\nDone.")
