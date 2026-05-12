import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
groups = train_raw['layout_id'].values

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

print("=== 1. Calibration Curve Analysis ===")
print("oracle_OOF prediction bins → median/mean y_true")
bins = [0,2,4,6,8,10,15,20,25,30,40,50,100,200]
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (fw4_oo >= lo) & (fw4_oo < hi)
    if mask.sum() > 100:
        yt = y_true[mask]; pr = fw4_oo[mask]
        print(f"  pred=[{lo:3d},{hi:3d}): n={mask.sum():6d}  pred_mean={pr.mean():6.2f}  y_median={np.median(yt):6.2f}  y_mean={yt.mean():6.2f}  bias(mean-pred)={yt.mean()-pr.mean():+.2f}")

print("\n=== 2. GroupKFold Quantile Calibration Validation ===")
print("Isotonic regression (q=0.5 = median) on oracle_OOF → y_true")
gkf = GroupKFold(n_splits=5)
maes_base, maes_cal = [], []
for fold, (tr_idx, val_idx) in enumerate(gkf.split(fw4_oo, y_true, groups)):
    # Fit isotonic on train fold
    iso = IsotonicRegression(out_of_bounds='clip')
    # Note: IsotonicRegression default uses MSE. For MAE we need median isotonic.
    # We'll approximate with quantile binning instead.
    pred_tr = fw4_oo[tr_idx]; y_tr = y_true[tr_idx]
    pred_val = fw4_oo[val_idx]; y_val = y_true[val_idx]

    # Build median calibration via binning + isotonic smoothing
    # Sort by prediction and compute rolling median
    sort_idx = np.argsort(pred_tr)
    pred_sorted = pred_tr[sort_idx]; y_sorted = y_tr[sort_idx]

    # Compute median in quantile bins (100 bins)
    n_bins = 100
    bin_edges = np.quantile(pred_sorted, np.linspace(0, 1, n_bins+1))
    bin_meds_pred, bin_meds_ytrue = [], []
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b+1]
        if b == n_bins-1: m = (pred_sorted >= lo)
        else: m = (pred_sorted >= lo) & (pred_sorted < hi)
        if m.sum() > 0:
            bin_meds_pred.append(np.median(pred_sorted[m]))
            bin_meds_ytrue.append(np.median(y_sorted[m]))

    bin_meds_pred = np.array(bin_meds_pred)
    bin_meds_ytrue = np.array(bin_meds_ytrue)

    # Isotonic fit on bin medians (ensures monotone calibration)
    iso.fit(bin_meds_pred, bin_meds_ytrue)

    # Apply calibration
    pred_cal = np.clip(iso.predict(pred_val), 0, None)

    mae_base = np.mean(np.abs(y_val - pred_val))
    mae_cal  = np.mean(np.abs(y_val - pred_cal))
    maes_base.append(mae_base); maes_cal.append(mae_cal)
    print(f"  Fold {fold+1}: base={mae_base:.4f}  calibrated={mae_cal:.4f}  delta={mae_cal-mae_base:+.4f}")

print(f"  Mean: base={np.mean(maes_base):.4f}  calibrated={np.mean(maes_cal):.4f}  delta={np.mean(maes_cal)-np.mean(maes_base):+.4f}")

print("\n=== 3. Calibration curve shape analysis ===")
sort_idx = np.argsort(fw4_oo)
pred_s = fw4_oo[sort_idx]; y_s = y_true[sort_idx]
n_bins = 100
bin_edges = np.quantile(pred_s, np.linspace(0, 1, n_bins+1))
print("pred_bin_median → y_true_median (the calibration function):")
prev_y = 0
for b in range(0, n_bins, 10):
    lo, hi = bin_edges[b], bin_edges[b+1]
    if b == n_bins-1: m = (pred_s >= lo)
    else: m = (pred_s >= lo) & (pred_s < hi)
    if m.sum() > 0:
        pm = np.median(pred_s[m]); ym = np.median(y_s[m])
        print(f"  pred_median={pm:6.2f} → y_median={ym:6.2f}  ratio={ym/max(pm,0.01):.3f}  delta={ym-pm:+.2f}")
        prev_y = ym

print("\n=== 4. Seen vs Unseen-like layouts calibration comparison ===")
# Use inflow as proxy for unseen-like
tr_lv = train_raw.groupby('layout_id').agg(inflow=('order_inflow_15m','mean')).reset_index()
hi_inflow_layouts = tr_lv[tr_lv['inflow'] > tr_lv['inflow'].quantile(0.75)]['layout_id'].values
hi_mask = train_raw['layout_id'].isin(hi_inflow_layouts).values
lo_mask = ~hi_mask

for label, mask in [('low_inflow (seen-like)', lo_mask), ('high_inflow (unseen-like)', hi_mask)]:
    pr = fw4_oo[mask]; yt = y_true[mask]
    mae = np.mean(np.abs(yt-pr))
    med = np.median(yt-pr)
    mean_r = (yt-pr).mean()
    print(f"  {label}: n={mask.sum()}  MAE={mae:.4f}  median_resid={med:+.4f}  mean_resid={mean_r:+.4f}")

print("\nDone.")
