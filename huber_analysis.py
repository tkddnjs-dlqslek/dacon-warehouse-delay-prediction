import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge, HuberRegressor
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

def looo_gbm(X_tr_, X_te_, y_, cap=None, **kw):
    looo = []
    for i in range(len(y_)):
        m = GradientBoostingRegressor(**kw)
        m.fit(np.delete(X_tr_, i, 0), np.delete(y_, i))
        p = m.predict(X_tr_[i:i+1])[0]
        looo.append(min(p, cap) if cap else p)
    m_f = GradientBoostingRegressor(**kw); m_f.fit(X_tr_, y_)
    tp = m_f.predict(X_te_)
    if cap: tp = np.minimum(tp, cap)
    return np.array(looo), tp

def looo_huber(X_tr_, X_te_, y_, eps, alpha=0.0001):
    looo = []
    for i in range(len(y_)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr_, i, 0))
        Xte_s = sc_.transform(X_tr_[i:i+1])
        m = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=1000)
        m.fit(Xtr_s, np.delete(y_, i))
        looo.append(m.predict(Xte_s)[0])
    sc_f = StandardScaler()
    m_f = HuberRegressor(epsilon=eps, alpha=alpha, max_iter=1000)
    m_f.fit(sc_f.fit_transform(X_tr_), y_)
    return np.array(looo), m_f.predict(sc_f.transform(X_te_))

gbm_kw = dict(max_depth=3, n_estimators=30, learning_rate=0.1, random_state=42)
gbm_looo, gbm_tp = looo_gbm(X_tr, X_te, y, cap=p95, **gbm_kw)

print("="*70)
print("Deep Huber Analysis: Feature Sets × Epsilon × Weights")
print("="*70)

# 1. Huber epsilon sweep on different feature sets
print(f"\n--- Epsilon sweep on base and interaction features ---")
best_overall = (99, None, None, None, None)
for feat_name, Xtr_, Xte_ in [
    ('base4',     X_tr,     X_te),
    ('int+pu*oof', X_int_tr, X_int_te),
]:
    print(f"\n  Features: {feat_name}")
    for eps in [1.35, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0]:
        for alpha in [0.0001, 0.01, 0.1]:
            h_looo, h_tp = looo_huber(Xtr_, Xte_, y, eps, alpha)
            for wg in [0.60, 0.65, 0.70, 0.75, 0.80]:
                bl = wg*gbm_looo + (1-wg)*h_looo
                mae = np.mean(np.abs(y - bl))
                tp_bl = wg*gbm_tp + (1-wg)*h_tp
                if mae < best_overall[0]:
                    best_overall = (mae, feat_name, eps, alpha, wg, h_tp)
            # Print only at wg=0.70
            bl = 0.70*gbm_looo + 0.30*h_looo
            mae = np.mean(np.abs(y - bl))
            print(f"    eps={eps:5.2f} a={alpha}: LOOO_huber={np.mean(np.abs(y-h_looo)):.4f}  "
                  f"gbm70+h30={mae:.4f}  h_test_mean={h_tp.mean():.3f}")

print(f"\n  BEST OVERALL:")
if best_overall[1]:
    mae, fname, eps, alpha, wg, h_tp = best_overall
    wr = 1-wg
    print(f"  {fname} eps={eps} a={alpha} GBM({wg:.2f})+Huber({wr:.2f}): LOOO={mae:.4f}  test_mean={(wg*gbm_tp+wr*h_tp).mean():.3f}")

# 2. Confirm best from initial run: GBM(0.7)+Huber(eps=2.0, a=0.0001, base4)
h_looo_2, h_tp_2 = looo_huber(X_tr, X_te, y, eps=2.0, alpha=0.0001)
bl_best = 0.7*gbm_looo + 0.3*h_looo_2
mae_best = np.mean(np.abs(y - bl_best))
print(f"\n  Confirmed: GBM(0.7)+Huber2.0(0.3): LOOO={mae_best:.4f}  test_mean={(0.7*gbm_tp+0.3*h_tp_2).mean():.3f}")

# 3. 3-way blend: GBM + Huber + RidgeInt
from sklearn.linear_model import Ridge
print(f"\n--- 3-way: GBM + Huber_best + RidgeInt ---")
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

# Best epsilon Huber LOOO
for wg, wh, wr in [(0.6,0.2,0.2),(0.65,0.15,0.2),(0.65,0.2,0.15),(0.7,0.2,0.1),(0.7,0.15,0.15)]:
    bl = wg*gbm_looo + wh*h_looo_2 + wr*ridgeint_looo
    mae = np.mean(np.abs(y - bl))
    tp = wg*gbm_tp + wh*h_tp_2 + wr*ridgeint_tp
    print(f"  GBM({wg})+Huber2({wh})+RidgeInt({wr}): LOOO={mae:.4f}  test_mean={tp.mean():.3f}")

print("\nDone.")
