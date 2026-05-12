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

# Timeslot positions
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

inflow_tr = train_raw['order_inflow_15m'].fillna(0).values
inflow_te = test_raw['order_inflow_15m'].fillna(0).values

print("="*60)
print("2D Correction: Timeslot × Inflow Bucket")
print("="*60)

# Inflow bins
inflow_bins = [(-np.inf, 35), (35, 78), (78, 132), (132, 200), (200, np.inf)]
inflow_labels = ['<35', '35-78', '78-132', '132-200', '>200']

def get_inflow_bucket(v):
    for i, (lo, hi) in enumerate(inflow_bins):
        if v >= lo and v < hi: return i
    return len(inflow_bins)-1

# Compute 2D residual table: ts_pos x inflow_bucket
ts_groups = [range(1,6), range(6,11), range(11,16), range(16,21), range(21,26)]
ts_labels = ['ts1-5', 'ts6-10', 'ts11-15', 'ts16-20', 'ts21-25']

print(f"\n  Training 2D residual table (ts_group x inflow_bucket):")
print(f"  {'ts_group':12s}", end='')
for il in inflow_labels: print(f"  {il:10s}", end='')
print()

resid_2d = {}
for tg_idx, ts_group in enumerate(ts_groups):
    ts_m = np.isin(ts_pos_tr, list(ts_group))
    print(f"  {ts_labels[tg_idx]:12s}", end='')
    for ib_idx, (lo, hi) in enumerate(inflow_bins):
        inf_m = (inflow_tr >= lo) & (inflow_tr < hi)
        m = ts_m & inf_m
        if m.sum() >= 200:
            r = residuals_train[m].mean()
            resid_2d[(tg_idx, ib_idx)] = r
            print(f"  {r:+10.3f}", end='')
        else:
            resid_2d[(tg_idx, ib_idx)] = None
            print(f"  {'N/A':>10}", end='')
    print()

# For unseen test rows, apply 2D ts-inflow correction
# ts_pos=1-5 (tg=0), inflow in [132,200) (ib=3) or (200+) (ib=4)
print(f"\n  Unseen test rows: ts × inflow distribution:")
for tg_idx, ts_group in enumerate(ts_groups):
    ts_m = np.isin(ts_pos_te[unseen_mask], list(ts_group))
    for ib_idx, (lo, hi) in enumerate(inflow_bins):
        inf_m = (inflow_te[unseen_mask] >= lo) & (inflow_te[unseen_mask] < hi)
        m = ts_m & inf_m
        if m.sum() > 0:
            print(f"  {ts_labels[tg_idx]}×{inflow_labels[ib_idx]}: n={m.sum():5d}")

# Apply 2D correction to unseen test
def get_2d_correction_ts(ts_val, inflow_val):
    tg_idx = (ts_val - 1) // 5  # ts 1-5 → 0, 6-10 → 1, etc.
    for ib_idx, (lo, hi) in enumerate(inflow_bins):
        if inflow_val >= lo and inflow_val < hi:
            r = resid_2d.get((tg_idx, ib_idx))
            if r is not None:
                return r
            # fallback: use ts-only (hi-inflow)
            hi_inflow_resids = {0: 3.47, 1: 5.49, 2: 5.54, 3: 6.21, 4: 7.58}
            return hi_inflow_resids.get(tg_idx, 5.5)
    return 5.5

ts_arr_te = ts_pos_te
inf_arr_te = inflow_te
corr_2d = np.array([get_2d_correction_ts(ts_arr_te[i], inf_arr_te[i])
                    for i in range(len(ts_arr_te))])
print(f"\n  2D ts×inflow correction stats for unseen:")
print(f"  mean={corr_2d[unseen_mask].mean():.4f}  std={corr_2d[unseen_mask].std():.4f}")

ct_2d = oracle_new_t.copy()
ct_2d[unseen_mask] = oracle_new_t[unseen_mask] + corr_2d[unseen_mask]
ct_2d = np.clip(ct_2d, 0, None)
du = ct_2d[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  2D ts×inflow: D={du:+.4f}  seen={ct_2d[seen_mask].mean():.3f}  unseen={ct_2d[unseen_mask].mean():.3f}")
fname = 'FINAL_NEW_oN_tsInflow2D_OOF8.3825.csv'
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_2d
sub.to_csv(fname, index=False)
print(f"  Saved: {fname}")

# Scale to hit 5.5
scale_2d = 5.5 / corr_2d[unseen_mask].mean()
corr_2d_scaled = corr_2d * scale_2d
ct_2d_sc = oracle_new_t.copy()
ct_2d_sc[unseen_mask] = oracle_new_t[unseen_mask] + corr_2d_scaled[unseen_mask]
ct_2d_sc = np.clip(ct_2d_sc, 0, None)
du2 = ct_2d_sc[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  2D ts×inflow scaled(5.5): D={du2:+.4f}  seen={ct_2d_sc[seen_mask].mean():.3f}  unseen={ct_2d_sc[unseen_mask].mean():.3f}")
fname2 = 'FINAL_NEW_oN_tsInflow2D_s5p5_OOF8.3825.csv'
sub2 = sub_tmpl.copy(); sub2['avg_delay_minutes_next_30m'] = ct_2d_sc
sub2.to_csv(fname2, index=False)
print(f"  Saved: {fname2}")

# Also: per-row ts correction (25 levels) x inflow (5 levels) → full 125-cell table
print(f"\n--- Full 25×5 2D table for hi-inflow (132-200) and (>200) rows ---")
print(f"  (these are the inflow buckets for 58% of unseen test rows)")
for ib_idx, ib_name in [(3,'132-200'), (4,'>200')]:
    lo, hi = inflow_bins[ib_idx]
    inf_m = (inflow_tr >= lo) & (inflow_tr < hi)
    print(f"  Inflow {ib_name}: {'ts':>4}  {'n':>7}  {'resid':>8}")
    for ts in range(1, 26):
        ts_m = ts_pos_tr == ts
        m = ts_m & inf_m
        if m.sum() >= 100:
            r = residuals_train[m].mean()
            print(f"  ts={ts:2d}: n={m.sum():7d}  resid={r:+8.3f}")

print("\nDone.")
