import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
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
    inflow=('order_inflow_15m','mean'),
).reset_index()

y = tr_lv['resid_mean'].values
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
    inflow=('order_inflow_15m','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy().reset_index(drop=True)

# Base arrays
oof_tr = tr_lv['oof_mean'].values
pu_tr  = tr_lv['pu'].values
tw_tr  = tr_lv['tw'].values
conv_tr= tr_lv['conv'].values
oof_te = te_lv_u['oN_mean'].values
pu_te  = te_lv_u['pu'].values
tw_te  = te_lv_u['tw'].values
conv_te= te_lv_u['conv'].values

def looo_gp_hub(X_tr_, X_te_, y_, wh=0.1, nr=3):
    kern = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
    looo_gp = []
    for i in range(len(y_)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr_, i, 0))
        Xte_s = sc_.transform(X_tr_[i:i+1])
        gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=nr)
        gp.fit(Xtr_s, np.delete(y_, i))
        looo_gp.append(gp.predict(Xte_s)[0])
    looo_gp = np.array(looo_gp)

    # Huber LOOO
    looo_h = []
    for i in range(len(y_)):
        sc_ = StandardScaler()
        Xtr_s = sc_.fit_transform(np.delete(X_tr_, i, 0))
        Xte_s = sc_.transform(X_tr_[i:i+1])
        m = HuberRegressor(epsilon=2.0, alpha=0.0001, max_iter=1000)
        m.fit(Xtr_s, np.delete(y_, i))
        looo_h.append(m.predict(Xte_s)[0])
    looo_h = np.array(looo_h)

    blend_looo = (1-wh)*looo_gp + wh*looo_h
    mae = np.mean(np.abs(y_ - blend_looo))

    # Final fit
    sc_f = StandardScaler()
    gp_f = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=nr)
    gp_f.fit(sc_f.fit_transform(X_tr_), y_)
    gp_tp = gp_f.predict(sc_f.transform(X_te_))

    sc_h2 = StandardScaler()
    h_f = HuberRegressor(epsilon=2.0, alpha=0.0001, max_iter=1000)
    h_f.fit(sc_h2.fit_transform(X_tr_), y_)
    hub_tp = h_f.predict(sc_h2.transform(X_te_))

    blend_tp = (1-wh)*gp_tp + wh*hub_tp
    return mae, blend_tp, np.mean(np.abs(y_ - looo_gp))

print("="*70)
print("GP+Hub(0.1) Feature Set Sweep")
print("="*70)

# Build feature sets
feature_sets = []

# F1: baseline (oof, pu, tw, conv, pu×oof) — current best
pu_oof_tr = oof_tr * pu_tr; pu_oof_te = oof_te * pu_te
F1_tr = np.column_stack([oof_tr, pu_tr, tw_tr, conv_tr, pu_oof_tr])
F1_te = np.column_stack([oof_te, pu_te, tw_te, conv_te, pu_oof_te])
feature_sets.append(('F1_base5[oof,pu,tw,conv,pu*oof]', F1_tr, F1_te))

# F2: drop conv (ARD showed conv useless)
F2_tr = np.column_stack([oof_tr, pu_tr, tw_tr, pu_oof_tr])
F2_te = np.column_stack([oof_te, pu_te, tw_te, pu_oof_te])
feature_sets.append(('F2_noconv[oof,pu,tw,pu*oof]', F2_tr, F2_te))

# F3: pu² added
pu2_tr = pu_tr**2; pu2_te = pu_te**2
F3_tr = np.column_stack([oof_tr, pu_tr, tw_tr, pu_oof_tr, pu2_tr])
F3_te = np.column_stack([oof_te, pu_te, tw_te, pu_oof_te, pu2_te])
feature_sets.append(('F3_pu2[oof,pu,tw,pu*oof,pu2]', F3_tr, F3_te))

# F4: minimal (oof, pu, pu×oof) — just the 3 key features
F4_tr = np.column_stack([oof_tr, pu_tr, pu_oof_tr])
F4_te = np.column_stack([oof_te, pu_te, pu_oof_te])
feature_sets.append(('F4_min3[oof,pu,pu*oof]', F4_tr, F4_te))

# F5: oof, pu, tw, pu×oof, pu×tw
putw_tr = pu_tr * tw_tr; putw_te = pu_te * tw_te
F5_tr = np.column_stack([oof_tr, pu_tr, tw_tr, pu_oof_tr, putw_tr])
F5_te = np.column_stack([oof_te, pu_te, tw_te, pu_oof_te, putw_te])
feature_sets.append(('F5_putw[oof,pu,tw,pu*oof,pu*tw]', F5_tr, F5_te))

# F6: all interactions (oof, pu, tw, pu*oof, pu2, pu*tw)
F6_tr = np.column_stack([oof_tr, pu_tr, tw_tr, pu_oof_tr, pu2_tr, putw_tr])
F6_te = np.column_stack([oof_te, pu_te, tw_te, pu_oof_te, pu2_te, putw_te])
feature_sets.append(('F6_full[oof,pu,tw,pu*oof,pu2,pu*tw]', F6_tr, F6_te))

# F7: add inflow_mean (order_inflow_15m avg per layout)
infl_tr = tr_lv['inflow'].values; infl_te = te_lv_u['inflow'].values
F7_tr = np.column_stack([oof_tr, pu_tr, tw_tr, pu_oof_tr, infl_tr])
F7_te = np.column_stack([oof_te, pu_te, tw_te, pu_oof_te, infl_te])
feature_sets.append(('F7_inflow[oof,pu,tw,pu*oof,inflow]', F7_tr, F7_te))

# F8: oof, pu, pu*oof, inflow (minimal + inflow)
F8_tr = np.column_stack([oof_tr, pu_tr, pu_oof_tr, infl_tr])
F8_te = np.column_stack([oof_te, pu_te, pu_oof_te, infl_te])
feature_sets.append(('F8_min+infl[oof,pu,pu*oof,inflow]', F8_tr, F8_te))

best_mae, best_name, best_tp = 99, '', None
for fname, Xtr, Xte in feature_sets:
    mae, tp, gp_only_mae = looo_gp_hub(Xtr, Xte, y, wh=0.1)
    print(f"  {fname:40s}: LOOO={mae:.4f}  (GP_only={gp_only_mae:.4f})  test_mean={tp.mean():.3f}  max={tp.max():.2f}")
    if mae < best_mae:
        best_mae, best_name, best_tp = mae, fname, tp

print(f"\n  Best: {best_name}, LOOO={best_mae:.4f}")

# Fine-tune Huber weight for best feature set
print(f"\n--- Huber weight fine-tune for best feature set ---")
best_fs_idx = [f[0] for f in feature_sets].index(best_name)
Xtr_b, Xte_b = feature_sets[best_fs_idx][1], feature_sets[best_fs_idx][2]

# Precompute LOOO arrays once
kern = ConstantKernel(1.0)*Matern(length_scale=1.0, nu=2.5) + WhiteKernel(0.1)
gp_looo = []
for i in range(len(y)):
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(Xtr_b, i, 0))
    Xte_s = sc_.transform(Xtr_b[i:i+1])
    gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=3)
    gp.fit(Xtr_s, np.delete(y, i))
    gp_looo.append(gp.predict(Xte_s)[0])
gp_looo = np.array(gp_looo)

hub_looo = []
for i in range(len(y)):
    sc_ = StandardScaler()
    Xtr_s = sc_.fit_transform(np.delete(Xtr_b, i, 0))
    Xte_s = sc_.transform(Xtr_b[i:i+1])
    m = HuberRegressor(epsilon=2.0, alpha=0.0001, max_iter=1000)
    m.fit(Xtr_s, np.delete(y, i))
    hub_looo.append(m.predict(Xte_s)[0])
hub_looo = np.array(hub_looo)

print(f"  GP standalone LOOO: {np.mean(np.abs(y-gp_looo)):.4f}")
for wh in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    bl = (1-wh)*gp_looo + wh*hub_looo
    mae = np.mean(np.abs(y - bl))
    print(f"  GP({1-wh:.2f})+Hub({wh:.2f}): LOOO={mae:.4f}")

print("\nDone.")
