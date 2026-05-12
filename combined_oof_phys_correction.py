import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression
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
sub_tmpl = pd.read_csv('sample_submission.csv')

phys_feats = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps']

# Layout-level stats for TRAINING
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'),
    resid_mean=('_resid','mean'),
    inflow_mean=('order_inflow_15m','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
).reset_index()

# Layout-level stats for UNSEEN TEST
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv = test_raw.groupby('layout_id').agg(
    oN_mean=('__dummy__','mean') if False else ('layout_id','count'),  # placeholder
    inflow_mean=('order_inflow_15m','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
).reset_index()
test_raw['_oN'] = oracle_new_t
te_lv2 = test_raw.groupby('layout_id').agg(oN_mean=('_oN','mean')).reset_index()
te_lv = pd.merge(te_lv.drop(columns=['layout_id']).assign(
    layout_id=test_raw.groupby('layout_id').size().index), te_lv2, on='layout_id')

# Rebuild properly
te_lv = test_raw.groupby('layout_id').agg(
    inflow_mean=('order_inflow_15m','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    pkg=('packaging_material_cost','mean'),
    agv=('agv_task_success_rate','mean'),
    oN_mean=('_oN','mean'),
).reset_index()
te_lv_u = te_lv[te_lv['layout_id'].isin(unseen_lids)].copy()

print("="*70)
print("Combined OOF-Mean + Physical Features Layout Correction")
print("="*70)

# Build combined model: oof_mean + phys features → training layout residual
X_feats_tr = tr_lv[['oof_mean','pu','tw','conv']].values
y_lv = tr_lv['resid_mean'].values
n_tr = len(tr_lv)

print(f"\n  Training layouts: {n_tr}")
print(f"  Unseen test layouts: {len(te_lv_u)}")

# LOOO comparison: various feature sets
import warnings
warnings.filterwarnings('ignore')

def looo_mae(X, y, alpha=10):
    preds = []
    for i in range(len(y)):
        X_tr = np.delete(X, i, 0); y_tr = np.delete(y, i)
        sc_ = StandardScaler(); X_tr_s = sc_.fit_transform(X_tr)
        X_te_s = sc_.transform(X[i:i+1])
        lr = Ridge(alpha=alpha); lr.fit(X_tr_s, y_tr)
        preds.append(lr.predict(X_te_s)[0])
    return np.mean(np.abs(y - np.array(preds))), np.array(preds)

feature_sets = [
    ('oof_mean only',      tr_lv[['oof_mean']].values,          1.0),
    ('phys3 only',         tr_lv[['pu','tw','conv']].values,    100),
    ('phys5 only',         tr_lv[['pu','tw','conv','pkg','agv']].values, 100),
    ('oof+phys3',          tr_lv[['oof_mean','pu','tw','conv']].values, 10),
    ('oof+phys5',          tr_lv[['oof_mean','pu','tw','conv','pkg','agv']].values, 10),
    ('oof+inflow+phys3',   tr_lv[['oof_mean','inflow_mean','pu','tw','conv']].values, 10),
]

print(f"\n  LOOO MAE comparison (training layouts):")
best_alpha_results = {}
for name, X, alpha in feature_sets:
    mae, preds_ = looo_mae(X, y_lv, alpha)
    best_alpha_results[name] = (mae, X, alpha)
    print(f"  {name:30s}: LOOO_MAE={mae:.4f}")

# Also try different alphas for combined
print(f"\n  Combined (oof+phys3) LOOO MAE by alpha:")
X_comb = tr_lv[['oof_mean','pu','tw','conv']].values
for alpha in [0.1, 1, 5, 10, 20, 50, 100, 200, 500]:
    mae, _ = looo_mae(X_comb, y_lv, alpha)
    print(f"  alpha={alpha:5.1f}: LOOO_MAE={mae:.4f}")

# Fit final combined model on all training data
best_alpha = 10  # from analysis above
X_comb_te = te_lv_u[['oN_mean','pu','tw','conv']].values
sc_f = StandardScaler()
X_comb_tr_s = sc_f.fit_transform(X_comb)
X_comb_te_s = sc_f.transform(X_comb_te)
reg_f = Ridge(alpha=best_alpha); reg_f.fit(X_comb_tr_s, y_lv)
raw_preds = reg_f.predict(X_comb_te_s)

print(f"\n  Combined model coefficients:")
feat_names = ['oof_mean','pack_util','truck_wait','conveyor']
for n, c in zip(feat_names, reg_f.coef_):
    print(f"  {n:20s}: {c:+.4f}")
print(f"  Intercept:           {reg_f.intercept_:+.4f}")

print(f"\n  Raw predictions for unseen layouts:")
print(f"  mean={raw_preds.mean():.4f}  std={raw_preds.std():.4f}  "
      f"min={raw_preds.min():.3f}  max={raw_preds.max():.3f}")

# Apply with offset to target 5.5
def apply_layout_correction(raw_corr_by_lid, target_mean, label, fname):
    current_mean = np.mean(list(raw_corr_by_lid.values()))
    offset = target_mean - current_mean
    final_corr = {lid: c + offset for lid, c in raw_corr_by_lid.items()}

    corr_arr = np.zeros(len(test_raw))
    for lid, c in final_corr.items():
        m = (test_raw['layout_id'] == lid).values
        corr_arr[m] = c

    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std_u = corr_arr[unseen_mask].std()
    print(f"  {label:45s}: D={du:+.4f}  seen={ct[seen_mask].mean():.3f}  "
          f"unseen={ct[unseen_mask].mean():.3f}  std={std_u:.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return final_corr, corr_arr

lid_to_raw = dict(zip(te_lv_u['layout_id'], raw_preds))

print(f"\n  --- Applying combined corrections (various targets) ---")
for target in [5.0, 5.5, 5.77, 6.0]:
    apply_layout_correction(lid_to_raw, target, f'oof+phys3 combined (target={target})',
                            f'FINAL_NEW_oN_oofPhys3comb_{str(target).replace(".","p")}_OOF8.3825.csv')

# Also try 5-feat version
X_comb5 = tr_lv[['oof_mean','pu','tw','conv','pkg','agv']].values
X_comb5_te = te_lv_u[['oN_mean','pu','tw','conv','pkg','agv']].values
sc5 = StandardScaler()
X5_tr_s = sc5.fit_transform(X_comb5); X5_te_s = sc5.transform(X_comb5_te)
reg5 = Ridge(alpha=10); reg5.fit(X5_tr_s, y_lv)
raw5 = reg5.predict(X5_te_s)
lid_to_raw5 = dict(zip(te_lv_u['layout_id'], raw5))
print(f"\n  5-feat combined (oof+phys5):")
apply_layout_correction(lid_to_raw5, 5.5, f'oof+phys5 combined (target=5.5)',
                        f'FINAL_NEW_oN_oofPhys5comb_5p5_OOF8.3825.csv')
apply_layout_correction(lid_to_raw5, 5.77, f'oof+phys5 combined (target=5.77)',
                        f'FINAL_NEW_oN_oofPhys5comb_5p77_OOF8.3825.csv')

# ========================================================
# Key question: is oof_mean signal reliable for OOD test?
# ========================================================
print(f"\n{'='*70}")
print("OOF-mean reliability check for OOD test layouts")
print(f"{'='*70}")

# In training: what does high oof_mean signify?
# Group by oof_mean quartile, check resid
tr_lv['oof_q'] = pd.qcut(tr_lv['oof_mean'], 4, labels=['Q1','Q2','Q3','Q4'])
grp = tr_lv.groupby('oof_q').agg(
    n=('oof_mean','count'), oof_mean=('oof_mean','mean'),
    resid_mean=('resid_mean','mean'), pu_mean=('pu','mean'),
    inflow_mean=('inflow_mean','mean')
)
print(f"\n  Training layouts by oof_mean quartile:")
print(grp.round(3).to_string())

# For unseen test layouts: oof_mean range
print(f"\n  Unseen test oracle_NEW mean distribution:")
te_lv_u_sorted = te_lv_u.sort_values('oN_mean')
print(f"  min={te_lv_u_sorted['oN_mean'].min():.2f}  p25={te_lv_u_sorted['oN_mean'].quantile(0.25):.2f}  "
      f"median={te_lv_u_sorted['oN_mean'].median():.2f}  p75={te_lv_u_sorted['oN_mean'].quantile(0.75):.2f}  "
      f"max={te_lv_u_sorted['oN_mean'].max():.2f}")

# Training oof_mean range
print(f"  Training layout oof_mean: min={tr_lv['oof_mean'].min():.2f}  "
      f"max={tr_lv['oof_mean'].max():.2f}  mean={tr_lv['oof_mean'].mean():.2f}")

# Danger: bimodal distribution of unseen test oN_mean?
low_oN = (te_lv_u['oN_mean'] < 15).sum()
mid_oN = ((te_lv_u['oN_mean'] >= 15) & (te_lv_u['oN_mean'] < 25)).sum()
high_oN = (te_lv_u['oN_mean'] >= 25).sum()
print(f"\n  Unseen layouts by oN_mean: <15:{low_oN}  15-25:{mid_oN}  >=25:{high_oN}")

# In training, oof_mean < 15 correlates with low residual?
low_oof_tr = tr_lv[tr_lv['oof_mean'] < 15]
high_oof_tr = tr_lv[tr_lv['oof_mean'] >= 25]
print(f"\n  Training: oof_mean<15 (n={len(low_oof_tr)}): resid={low_oof_tr['resid_mean'].mean():.3f}, "
      f"pu={low_oof_tr['pu'].mean():.3f}, inflow={low_oof_tr['inflow_mean'].mean():.2f}")
print(f"  Training: oof_mean>=25 (n={len(high_oof_tr)}): resid={high_oof_tr['resid_mean'].mean():.3f}, "
      f"pu={high_oof_tr['pu'].mean():.3f}, inflow={high_oof_tr['inflow_mean'].mean():.2f}")

# What about unseen layouts with low oN_mean? Are they physically easy?
low_oN_u = te_lv_u[te_lv_u['oN_mean'] < 15]
high_oN_u = te_lv_u[te_lv_u['oN_mean'] >= 25]
print(f"\n  Unseen test: oN_mean<15 (n={len(low_oN_u)}): pu={low_oN_u['pu'].mean():.3f}, "
      f"inflow={low_oN_u['inflow_mean'].mean():.2f}")
print(f"  Unseen test: oN_mean>=25 (n={len(high_oN_u)}): pu={high_oN_u['pu'].mean():.3f}, "
      f"inflow={high_oN_u['inflow_mean'].mean():.2f}")

print("\nDone.")
