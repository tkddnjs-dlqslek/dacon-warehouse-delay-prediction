import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
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

phys_feats = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps']

# Training layout-level features
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'),
    resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

# Unseen test layout-level features
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

import warnings
warnings.filterwarnings('ignore')

def looo_and_predict(X_tr, X_te, y, alpha, lid_te):
    preds_tr = []
    for i in range(len(y)):
        Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y, i)
        sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_)
        Xte_s = sc_.transform(X_tr[i:i+1])
        lr = Ridge(alpha=alpha); lr.fit(Xtr_s, ytr_)
        preds_tr.append(lr.predict(Xte_s)[0])
    looo_mae = np.mean(np.abs(y - np.array(preds_tr)))

    # Full model for test
    sc_f = StandardScaler()
    Xtr_s = sc_f.fit_transform(X_tr)
    Xte_s = sc_f.transform(X_te)
    lr_f = Ridge(alpha=alpha); lr_f.fit(Xtr_s, y)
    test_preds = lr_f.predict(Xte_s)
    return looo_mae, test_preds, dict(zip(lid_te, test_preds)), lr_f.coef_, lr_f.intercept_

def save_corr(lid_to_raw, target, label, fname):
    offset = target - np.mean(list(lid_to_raw.values()))
    corr_arr = np.zeros(len(test_raw))
    for lid, c in lid_to_raw.items():
        corr_arr[(test_raw['layout_id'] == lid).values] = c + offset
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std_u = corr_arr[unseen_mask].std()
    print(f"  {label:50s}: D={du:+.4f}  unseen={ct[unseen_mask].mean():.3f}  std={std_u:.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return corr_arr[unseen_mask]

print("="*70)
print("Optimal Combined Corrections (best alpha per feature set)")
print("="*70)

configs = [
    # (name, X_tr cols, X_te cols, alpha, targets, file_label)
    ('oof+phys3 a100',
     tr_lv[['oof_mean','pu','tw','conv']].values,
     te_lv_u[['oN_mean','pu','tw','conv']].values,
     100, [5.0, 5.5, 5.77], 'oofPhys3_a100'),
    ('oof+phys5 a100',
     tr_lv[['oof_mean','pu','tw','conv','pkg','agv']].values,
     te_lv_u[['oN_mean','pu','tw','conv','pkg','agv']].values,
     100, [5.0, 5.5, 5.77], 'oofPhys5_a100'),
    ('oof+phys3 a50',
     tr_lv[['oof_mean','pu','tw','conv']].values,
     te_lv_u[['oN_mean','pu','tw','conv']].values,
     50, [5.5], 'oofPhys3_a50'),
    ('oof+phys3+inflow a100',
     tr_lv[['oof_mean','inflow_mean','pu','tw','conv']].values,
     te_lv_u[['oN_mean','inflow_mean','pu','tw','conv']].values,
     100, [5.5], 'oofInfPhys3_a100'),
]

print(f"\n  LOOO MAE and model details:")
best_corrs = {}
for cfg in configs:
    name, X_tr, X_te, alpha, targets, flabel = cfg
    mae, test_preds, lid_map, coefs, intercept = looo_and_predict(
        X_tr, X_te, y_lv, alpha, te_lv_u['layout_id'].values)
    n_feats = X_tr.shape[1]
    feat_names = ['oof_mean','pu','tw','conv','pkg','agv','inflow'][:n_feats]
    print(f"\n  {name}: LOOO_MAE={mae:.4f}")
    print(f"    coefs: " + "  ".join(f"{n}={c:+.3f}" for n,c in zip(feat_names, coefs)))
    print(f"    intercept={intercept:+.4f}")
    print(f"    test raw: mean={np.mean(list(lid_map.values())):.3f}  "
          f"std={np.std(list(lid_map.values())):.3f}  "
          f"min={min(lid_map.values()):.3f}  max={max(lid_map.values()):.3f}")
    for t in targets:
        c_arr = save_corr(lid_map, t, f'{name} → {t}',
                          f'FINAL_NEW_oN_{flabel}_{str(t).replace(".","p")}_OOF8.3825.csv')
        if t == 5.5:
            best_corrs[name] = c_arr

# ============================================================
# Blend combined models with flat and with each other
# ============================================================
print(f"\n{'='*70}")
print("Blended Combinations")
print(f"{'='*70}")

_, tp_a100, lid_a100, _, _ = looo_and_predict(
    tr_lv[['oof_mean','pu','tw','conv']].values,
    te_lv_u[['oN_mean','pu','tw','conv']].values,
    y_lv, 100, te_lv_u['layout_id'].values)

_, tp_phys3, lid_phys3, _, _ = looo_and_predict(
    tr_lv[['pu','tw','conv']].values,
    te_lv_u[['pu','tw','conv']].values,
    y_lv, 100, te_lv_u['layout_id'].values)

offset_a100 = 5.5 - np.mean(tp_a100)
offset_phys3 = 5.5 - np.mean(tp_phys3)

def make_corr_arr(lid_map, offset):
    arr = np.zeros(len(test_raw))
    for lid, c in lid_map.items():
        arr[(test_raw['layout_id'] == lid).values] = c + offset
    return arr

arr_a100  = make_corr_arr(lid_a100, offset_a100)
arr_phys3 = make_corr_arr(lid_phys3, offset_phys3)

# Load iso_lvL75 correction
iso_df = pd.read_csv('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv')
iso_df = iso_df.set_index('ID').reindex(id_order).reset_index()
iso_t = iso_df['avg_delay_minutes_next_30m'].values
iso_corr = iso_t - oracle_new_t  # correction array (unseen already scaled)

def save_blend_arr(corr_arr, label, fname):
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std_u = corr_arr[unseen_mask].std()
    print(f"  {label:55s}: D={du:+.4f}  unseen={ct[unseen_mask].mean():.3f}  std={std_u:.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)

print(f"\n  Combined model blends (all target≈5.5):")
# iso + combined_a100
for wi in [0.3, 0.4, 0.5, 0.6, 0.7]:
    wc = 1 - wi
    ens = wi * iso_corr + wc * arr_a100
    # Scale to exactly 5.5
    ens_scaled = ens * (5.5 / ens[unseen_mask].mean())
    save_blend_arr(ens_scaled, f'iso({wi:.1f}) + combined_a100({wc:.1f}) [scaled5.5]',
                   f'FINAL_NEW_oN_isoComb_w{int(wi*10)}_OOF8.3825.csv')

# Flat + combined_a100
print(f"\n  Flat + combined_a100 blend:")
flat_arr = np.zeros(len(test_raw))
flat_arr[unseen_mask] = 5.5
for wc in [0.3, 0.5, 0.7, 0.8, 0.9]:
    wf = 1 - wc
    ens = wc * arr_a100 + wf * flat_arr
    ens_scaled = ens * (5.5 / ens[unseen_mask].mean())
    save_blend_arr(ens_scaled, f'combined_a100({wc:.1f}) + flat({wf:.1f}) [scaled5.5]',
                   f'FINAL_NEW_oN_combFlat_w{int(wc*10)}_OOF8.3825.csv')

# 3-way: iso + combined + flat
print(f"\n  3-way iso + combined + flat:")
for (wi, wc, wf) in [(0.33,0.33,0.34),(0.25,0.50,0.25),(0.40,0.40,0.20),(0.50,0.25,0.25)]:
    ens = wi * iso_corr + wc * arr_a100 + wf * flat_arr
    ens_scaled = ens * (5.5 / ens[unseen_mask].mean())
    save_blend_arr(ens_scaled, f'iso({wi:.2f})+comb({wc:.2f})+flat({wf:.2f}) [5.5]',
                   f'FINAL_NEW_oN_3way_{int(wi*100):02d}{int(wc*100):02d}_OOF8.3825.csv')

# Show variance of corrections across layouts
print(f"\n  Per-layout correction variance comparison:")
from scipy.stats import pearsonr
print(f"  {'Method':40s}  std  min   max  r(iso,this)")
methods = {
    'flat 5.5': flat_arr,
    'phys3_a100 (scaled)': arr_phys3,
    'combined_a100 (scaled)': arr_a100,
    'iso_lvL75': iso_corr,
}
for mn, arr in methods.items():
    u_corr = arr[unseen_mask]
    r, _ = pearsonr(iso_corr[unseen_mask], u_corr) if mn != 'iso_lvL75' else (1.0, 0)
    print(f"  {mn:40s}: std={u_corr.std():.3f}  [{u_corr.min():.2f}, {u_corr.max():.2f}]  r={r:.3f}")

print("\nDone.")
