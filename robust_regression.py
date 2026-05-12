import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge, HuberRegressor, QuantileRegressor
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

# Pre-compute GBM base LOOO
gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_looo = []
for i in range(len(y)):
    m = GradientBoostingRegressor(**gbm_kw)
    m.fit(np.delete(X_tr, i, 0), np.delete(y, i))
    gbm_looo.append(min(m.predict(X_tr[i:i+1])[0], p95))
gbm_looo = np.array(gbm_looo)
gbm_m = GradientBoostingRegressor(**gbm_kw); gbm_m.fit(X_tr, y)
gbm_tp = np.minimum(gbm_m.predict(X_te), p95)

# Pre-compute RidgeInt LOOO
ridgeint_looo = []
for i in range(len(y)):
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(X_int_tr, i, 0))
    Xte_s = sc_.transform(X_int_tr[i:i+1])
    m = Ridge(alpha=100); m.fit(Xtr_s, np.delete(y, i))
    ridgeint_looo.append(m.predict(Xte_s)[0])
ridgeint_looo = np.array(ridgeint_looo)
sc_ri = StandardScaler(); ri_f = Ridge(alpha=100)
ri_f.fit(sc_ri.fit_transform(X_int_tr), y)
ridgeint_tp = ri_f.predict(sc_ri.transform(X_int_te))

mae_ref = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*ridgeint_looo)))
print(f"Reference: GBM(0.7)+RidgeInt(0.3): LOOO={mae_ref:.4f}")

print("="*70)
print("Robust Regression / Quantile / Huber at Layout Level")
print("="*70)

# 1. Huber regression (robust to outlier layouts)
print(f"\n--- Huber Regression ---")
for epsilon in [1.1, 1.35, 1.5, 2.0]:
    looo = []
    for i in range(len(y)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_int_tr, i, 0))
        Xte_s = sc_.transform(X_int_tr[i:i+1])
        m = HuberRegressor(epsilon=epsilon, max_iter=500)
        m.fit(Xtr_s, np.delete(y, i))
        looo.append(m.predict(Xte_s)[0])
    looo = np.array(looo)
    mae = np.mean(np.abs(y - looo))
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    sc_f = StandardScaler(); m_f = HuberRegressor(epsilon=epsilon, max_iter=500)
    m_f.fit(sc_f.fit_transform(X_int_tr), y)
    tp = m_f.predict(sc_f.transform(X_int_te))
    print(f"  Huber eps={epsilon}: LOOO={mae:.4f}  gbm70+huber30={bl:.4f}  test_mean={tp.mean():.3f}")

# 2. Quantile regression at different quantiles
print(f"\n--- Quantile Regression (alpha=C for solver) ---")
for q in [0.45, 0.50, 0.55, 0.60, 0.65]:
    looo = []
    for i in range(len(y)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_int_tr, i, 0))
        Xte_s = sc_.transform(X_int_tr[i:i+1])
        m = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
        m.fit(Xtr_s, np.delete(y, i))
        looo.append(m.predict(Xte_s)[0])
    looo = np.array(looo)
    mae = np.mean(np.abs(y - looo))
    bl = np.mean(np.abs(y - (0.7*gbm_looo + 0.3*looo)))
    sc_f = StandardScaler(); m_f = QuantileRegressor(quantile=q, alpha=0.1, solver='highs')
    m_f.fit(sc_f.fit_transform(X_int_tr), y)
    tp = m_f.predict(sc_f.transform(X_int_te))
    print(f"  Quantile q={q}: LOOO={mae:.4f}  gbm70+qr30={bl:.4f}  test_mean={tp.mean():.3f}")

# 3. Fine-grained GBM weight search
print(f"\n--- Fine-grained weight grid (GBM + RidgeInt) ---")
best_mae, best_w = 99, 0
for wg in np.arange(0.50, 0.85, 0.05):
    wr = 1 - wg
    bl = wg*gbm_looo + wr*ridgeint_looo
    mae = np.mean(np.abs(y - bl))
    tp = wg*gbm_tp + wr*ridgeint_tp
    mark = " ←" if mae < best_mae else ""
    print(f"  GBM({wg:.2f})+RidgeInt({wr:.2f}): LOOO={mae:.4f}  test_mean={tp.mean():.3f}{mark}")
    if mae < best_mae: best_mae = mae; best_w = wg

# 4. GBM with different cap strategies
print(f"\n--- Alternative Capping Strategies ---")
for cap in [9.0, 10.0, 11.0, 12.59, 14.0, None]:
    looo_c = np.array([min(p, cap) if cap else p for p in gbm_looo])
    bl = np.mean(np.abs(y - (0.7*looo_c + 0.3*ridgeint_looo)))
    tp_c = np.minimum(gbm_tp, cap) if cap else gbm_tp
    tp_bl = 0.7*tp_c + 0.3*ridgeint_tp
    print(f"  GBM_cap({str(cap):5s})(0.7)+RidgeInt(0.3): LOOO={bl:.4f}  test_mean={tp_bl.mean():.3f}  std={tp_bl.std():.3f}")

# 5. GBM with sample weights (upweight high-residual layouts)
print(f"\n--- GBM with Sample Weights (upweight extreme layouts) ---")
for pw in [1, 2, 3]:
    sw = (y**pw) / (y**pw).mean()  # weight proportional to residual
    looo_sw = []
    for i in range(len(y)):
        sw_ = np.delete(sw, i); sw_ = sw_/sw_.mean()
        m = GradientBoostingRegressor(**gbm_kw)
        m.fit(np.delete(X_tr, i, 0), np.delete(y, i), sample_weight=sw_)
        looo_sw.append(min(m.predict(X_tr[i:i+1])[0], p95))
    looo_sw = np.array(looo_sw)
    mae = np.mean(np.abs(y - looo_sw))
    bl = np.mean(np.abs(y - (0.7*looo_sw + 0.3*ridgeint_looo)))
    m_f = GradientBoostingRegressor(**gbm_kw)
    m_f.fit(X_tr, y, sample_weight=sw/sw.mean())
    tp = np.minimum(m_f.predict(X_te), p95)
    tp_bl = 0.7*tp + 0.3*ridgeint_tp
    print(f"  GBM_sw_pw={pw}: LOOO={mae:.4f}  gbm70+ri30={bl:.4f}  test_mean={tp_bl.mean():.3f}")

print("\nDone.")
