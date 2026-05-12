import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
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

# Infer timeslot positions
train_raw_ts = train_raw.copy()
train_raw_ts['ts_pos'] = (train_raw_ts.sort_values(['layout_id','scenario_id'])
                           .groupby(['layout_id','scenario_id']).cumcount().values + 1)
train_raw_ts = train_raw_ts.sort_values('_row_id').reset_index(drop=True)

test_raw_ts = test_raw.copy()
test_raw_ts['ts_pos'] = (test_raw_ts.sort_values(['layout_id','scenario_id'])
                          .groupby(['layout_id','scenario_id']).cumcount().values + 1)
test_raw_ts = test_raw_ts.sort_values('_row_id').reset_index(drop=True)

ts_pos_tr = train_raw_ts['ts_pos'].values
ts_pos_te = test_raw_ts['ts_pos'].values

inflow_col = 'order_inflow_15m'
hi_inflow_mask = train_raw['order_inflow_15m'].fillna(0).values >= 132

print("="*60)
print("Timeslot-Stratified Correction for Unseen Test")
print("="*60)

# Per-ts residual: high-inflow training rows
ts_resids_hi = {}
ts_resids_all = {}
for ts in range(1, 26):
    m_hi = (ts_pos_tr == ts) & hi_inflow_mask
    m_all = ts_pos_tr == ts
    ts_resids_hi[ts] = residuals_train[m_hi].mean() if m_hi.sum() > 0 else 3.21
    ts_resids_all[ts] = residuals_train[m_all].mean()

print(f"\n  Per-ts residuals (hi-inflow vs all training):")
print(f"  {'ts':>4}  {'hi_inflow':>10}  {'all':>8}  {'ratio':>6}")
for ts in range(1, 26):
    r = ts_resids_hi[ts] / ts_resids_all[ts] if ts_resids_all[ts] > 0 else 1.0
    print(f"  {ts:4d}:  {ts_resids_hi[ts]:+10.3f}  {ts_resids_all[ts]:+8.3f}  {r:6.3f}")

# Mean correction if applied to balanced unseen
mean_hi_ts = np.mean(list(ts_resids_hi.values()))
mean_all_ts = np.mean(list(ts_resids_all.values()))
print(f"\n  Mean hi-inflow ts correction: {mean_hi_ts:.4f}")
print(f"  Mean all-training ts correction: {mean_all_ts:.4f}")

# Scale hi-inflow corrections to hit consensus mean of 5.5
scale_to_5p5 = 5.5 / mean_hi_ts
ts_resids_hi_scaled = {ts: ts_resids_hi[ts] * scale_to_5p5 for ts in range(1, 26)}
print(f"\n  Scaled hi-inflow ts corrections (target mean=5.5, scale={scale_to_5p5:.4f}):")
for ts in range(1, 26):
    print(f"  ts={ts:2d}: raw={ts_resids_hi[ts]:+.3f}  scaled={ts_resids_hi_scaled[ts]:+.3f}")

def apply_ts_correction(ts_corr_dict, name, fname):
    ts_corr_arr = np.array([ts_corr_dict[ts] for ts in ts_pos_te])
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + ts_corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {name}: D={du:+.4f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    return du

print(f"\n--- Applying timeslot-stratified corrections ---")
apply_ts_correction(ts_resids_hi, "hi-inflow raw ts",
                    "FINAL_NEW_oN_tsHiInflow_OOF8.3825.csv")
apply_ts_correction(ts_resids_hi_scaled, "hi-inflow scaled(5.5) ts",
                    "FINAL_NEW_oN_tsHiScaled5p5_OOF8.3825.csv")
apply_ts_correction(ts_resids_all, "all-training ts",
                    "FINAL_NEW_oN_tsAll_OOF8.3825.csv")

# Scale all-training to 5.5
scale_all_5p5 = 5.5 / mean_all_ts
ts_resids_all_scaled = {ts: ts_resids_all[ts] * scale_all_5p5 for ts in range(1, 26)}
apply_ts_correction(ts_resids_all_scaled, "all-training scaled(5.5) ts",
                    "FINAL_NEW_oN_tsAllScaled5p5_OOF8.3825.csv")

# Important check: timeslot distribution for unseen test
ts_te_u = ts_pos_te[unseen_mask]
print(f"\n  Timeslot distribution for unseen test:")
for ts in range(1, 26):
    n = (ts_te_u == ts).sum()
    print(f"  ts={ts:2d}: n={n:5d}  hi_corr={ts_resids_hi[ts]:+.3f}  scaled={ts_resids_hi_scaled[ts]:+.3f}")

print("\nDone.")
