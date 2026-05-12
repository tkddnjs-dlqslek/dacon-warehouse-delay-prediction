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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

inflow_col = 'order_inflow_15m'
inflow_tr = train_raw[inflow_col].fillna(0).values
inflow_te = test_raw[inflow_col].fillna(0).values
hi_mask = inflow_tr >= 132

# Timeslot positions
ts_pos_tr = train_raw.sort_values(['layout_id','scenario_id']).groupby(
    ['layout_id','scenario_id']).cumcount().values + 1
train_raw['ts_pos'] = pd.Series(ts_pos_tr, index=train_raw.sort_values(
    ['layout_id','scenario_id']).index).sort_index().values

ts_pos_te = test_raw.sort_values(['layout_id','scenario_id']).groupby(
    ['layout_id','scenario_id']).cumcount().values + 1
test_raw['ts_pos'] = pd.Series(ts_pos_te, index=test_raw.sort_values(
    ['layout_id','scenario_id']).index).sort_index().values
ts_arr_te = test_raw['ts_pos'].values
ts_arr_tr = train_raw['ts_pos'].values

# Physical feature model (3 features, alpha=100)
train_raw['_resid'] = residuals_train
layout_grp = train_raw.groupby('layout_id')
layout_resid_mean = layout_grp['_resid'].mean()
phys_feats = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps']
lids_tr = layout_resid_mean.index
y_resid_lv = layout_resid_mean.values
tr_lv_feats = layout_grp[phys_feats].mean()
te_lv_feats = test_raw.groupby('layout_id')[phys_feats].mean()
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
X_tr_lv = tr_lv_feats.loc[lids_tr].values
X_u_lv  = te_lv_feats.loc[unseen_lids].values
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr_lv)
X_u_s  = sc.transform(X_u_lv)
reg = Ridge(alpha=100)
reg.fit(X_tr_s, y_resid_lv)
phys_pred = reg.predict(X_u_s)
phys_mean = phys_pred.mean()
lid_to_phys = {lid: phys_pred[i] for i, lid in enumerate(unseen_lids)}

# Timeslot corrections from hi-inflow training
ts_resids_hi = {}
for ts in range(1, 26):
    m = (ts_arr_tr == ts) & hi_mask
    ts_resids_hi[ts] = residuals_train[m].mean() if m.sum() > 0 else 5.5
ts_hi_mean = np.mean(list(ts_resids_hi.values()))
ts_deviations = {ts: ts_resids_hi[ts] - ts_hi_mean for ts in range(1, 26)}

print("="*60)
print("Combined Layout-Physics + Timeslot Correction")
print("="*60)
print(f"\n  Phys model mean: {phys_mean:.4f}  ts-hi mean: {ts_hi_mean:.4f}")
print(f"  ts-hi std (deviations): {np.std(list(ts_deviations.values())):.4f}")

def save_combo(target_mean, label, fname):
    # combined = phys_pred[layout] + ts_deviation[ts] + offset_to_hit_target
    row_corr_raw = np.zeros(len(test_raw))
    for lid in unseen_lids:
        m = (test_raw['layout_id'] == lid).values
        row_corr_raw[m] = lid_to_phys[lid]
    for ts in range(1, 26):
        m = (ts_arr_te == ts)
        row_corr_raw[m] += ts_deviations[ts]
    # Offset to hit target mean for unseen rows
    current_mean = row_corr_raw[unseen_mask].mean()
    offset = target_mean - current_mean
    row_corr = row_corr_raw + offset

    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + row_corr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {label}: D={du:+.4f}  seen={ct[seen_mask].mean():.3f}  "
          f"unseen={ct[unseen_mask].mean():.3f}  std(corr)={row_corr[unseen_mask].std():.4f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return row_corr[unseen_mask]

for target in [5.0, 5.5, 5.77, 6.0]:
    corr = save_combo(target, f"phys+ts (target={target})",
                      f"FINAL_NEW_oN_physTs_{str(target).replace('.','p')}_OOF8.3825.csv")

# Show detailed per-row stats
corr5p5 = save_combo(5.5, "phys+ts (5.5) [verify]", "/dev/null")
print(f"\n  Row correction stats for phys+ts (target=5.5):")
print(f"  mean={corr5p5.mean():.4f}  std={corr5p5.std():.4f}")
print(f"  p5={np.percentile(corr5p5,5):.3f}  p25={np.percentile(corr5p5,25):.3f}  "
      f"p50={np.percentile(corr5p5,50):.3f}  p75={np.percentile(corr5p5,75):.3f}  "
      f"p95={np.percentile(corr5p5,95):.3f}")
print(f"  min={corr5p5.min():.3f}  max={corr5p5.max():.3f}")

# Compare: flat +5.5 vs ts-scaled vs phys+ts
print(f"\n  Comparison: flat vs ts-scaled vs phys+ts (all Δ=5.5)")
flat_corr = np.full(unseen_mask.sum(), 5.5)

ts_scaled = np.array([ts_resids_hi[ts] * (5.5/ts_hi_mean)
                       for ts in ts_arr_te[unseen_mask]])
print(f"  flat: std=0.000")
print(f"  ts-scaled: std={ts_scaled.std():.4f}  min={ts_scaled.min():.3f}  max={ts_scaled.max():.3f}")
print(f"  phys+ts: std={corr5p5.std():.4f}  min={corr5p5.min():.3f}  max={corr5p5.max():.3f}")

print("\nDone.")
