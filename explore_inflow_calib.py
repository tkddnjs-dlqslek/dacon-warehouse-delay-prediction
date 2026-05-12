import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask     = ~unseen_mask

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

mega33_oof  = d33['meta_avg_oof'][id2]; mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]; mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy');       xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy');    lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t  = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t   = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2];   slhm_t = np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]
savg_o = np.load('results/cascade/spec_avg_oof.npy')[id2];            savg_t = np.load('results/cascade/spec_avg_test.npy')[te_id2]

mae_fn = lambda p: float(np.mean(np.abs(np.clip(p, 0, None) - y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx  = wm*mega   + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

bb_o, bb_t = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)
fw4_o = np.clip(0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
fw4_t = np.clip(0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)

n1, w1, n2, w2 = 2000, 0.15, 5500, 0.08
sfw = np.sort(fw4_o); sft = np.sort(fw4_t)
m1_o = fw4_o >= sfw[-n1]; m2_o = fw4_o >= sfw[-n2]
dual_o = fw4_o.copy()
dual_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
dual_o[m2_o] = (1-w2)*dual_o[m2_o] + w2*slhm_o[m2_o]
dual_mae = mae_fn(dual_o)

m1_t = fw4_t >= sft[-n1]; m2_t = fw4_t >= sft[-n2]
dual_t = fw4_t.copy()
dual_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
dual_t[m2_t] = (1-w2)*dual_t[m2_t] + w2*slhm_t[m2_t]
dual_t = np.clip(dual_t, 0, None)
print(f"dual_gate: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

m1_s_t = (fw4_t >= sft[-n1]) & seen_mask
m2_s_t = (fw4_t >= sft[-n2]) & seen_mask

print("\n" + "="*70)
print("Part 1: Inflow-delay relationship in training data")
print("="*70)

inflow_col = 'order_inflow_15m'
train_raw['inflow'] = train_raw[inflow_col]
train_raw['pred_dual'] = dual_o

# Compute scenario-level stats
scen_stats = train_raw.groupby(['layout_id','scenario_id']).agg(
    y_mean=('avg_delay_minutes_next_30m','mean'),
    inflow_mean=(inflow_col,'mean'),
    pred_mean=('pred_dual','mean')
).reset_index()
scen_stats['residual'] = scen_stats['y_mean'] - scen_stats['pred_mean']

print(f"Scenario inflow range: {scen_stats.inflow_mean.min():.1f} - {scen_stats.inflow_mean.max():.1f}")
print(f"Test unseen inflow range: {test_raw[test_raw['layout_id'].isin(set(test_raw['layout_id'].unique())-train_layouts)][inflow_col].agg(['min','max','mean'])}")

# Inflow quantile analysis
for q in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]:
    thresh = scen_stats.inflow_mean.quantile(q)
    mask_q = scen_stats.inflow_mean >= thresh
    print(f"  Inflow >= p{int(q*100):3d} ({thresh:.1f}): n={mask_q.sum():5d}  y_mean={scen_stats.loc[mask_q,'y_mean'].mean():.2f}  pred_mean={scen_stats.loc[mask_q,'pred_mean'].mean():.2f}  resid={scen_stats.loc[mask_q,'residual'].mean():.3f}")

# Check top-inflow scenarios
print(f"\nTop 20 highest-inflow training scenarios:")
top_inf = scen_stats.nlargest(20, 'inflow_mean')
print(top_inf[['layout_id','scenario_id','inflow_mean','y_mean','pred_mean','residual']].to_string(index=False))

# Fit a simple linear model: y ~ inflow
from numpy.polynomial import polynomial as P
xs = scen_stats.inflow_mean.values
ys = scen_stats.y_mean.values
coeffs = np.polyfit(xs, ys, 1)
print(f"\nLinear fit: y_mean = {coeffs[0]:.4f} * inflow + {coeffs[1]:.4f}")
# Predict for unseen inflow 172
for inf_val in [94, 130, 158, 172, 190]:
    pred_y = coeffs[0]*inf_val + coeffs[1]
    print(f"  inflow={inf_val}: predicted_delay={pred_y:.2f}")

# Fit log-linear
log_xs = np.log(xs)
log_coeffs = np.polyfit(log_xs, ys, 1)
print(f"\nLog-linear fit: y_mean = {log_coeffs[0]:.4f} * log(inflow) + {log_coeffs[1]:.4f}")
for inf_val in [94, 130, 158, 172, 190]:
    pred_y = log_coeffs[0]*np.log(inf_val) + log_coeffs[1]
    print(f"  inflow={inf_val}: predicted_delay={pred_y:.2f}")

print("\n" + "="*70)
print("Part 2: Model underprediction at high inflow in training")
print("(OOF residual analysis by inflow quantile)")
print("="*70)

# Check residuals by inflow percentile
inflow_arr = train_raw[inflow_col].values
preds_arr = dual_o
resid_arr = y_true - preds_arr

for q in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
    thresh = np.percentile(inflow_arr, q*100)
    mask_q = inflow_arr >= thresh
    mean_resid = resid_arr[mask_q].mean()
    mean_y = y_true[mask_q].mean()
    mean_pred = preds_arr[mask_q].mean()
    mae_q = np.mean(np.abs(resid_arr[mask_q]))
    print(f"  inflow >= p{int(q*100):3d} ({thresh:.1f}): n={mask_q.sum():6d}  y={mean_y:.2f}  pred={mean_pred:.2f}  resid={mean_resid:.3f}  MAE={mae_q:.3f}")

# Unseen test layout stats
unseen_inflow = test_raw.loc[unseen_mask, inflow_col].values
print(f"\nUnseen test inflow: mean={unseen_inflow.mean():.1f}  std={unseen_inflow.std():.1f}")
print(f"Training inflow max: {inflow_arr.max():.1f}")
print(f"Extrapolation factor (unseen/train_max): {unseen_inflow.mean()/inflow_arr.max():.2f}")

print("\n" + "="*70)
print("Part 3: Scenario-level residual analysis for calibration")
print("="*70)

# How well does the model track at the scenario level for high-inflow scenarios?
# For each training scenario, compute predicted_mean and actual_mean
train_raw['dual_pred'] = dual_o
layout_stats = train_raw.groupby('layout_id').agg(
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('dual_pred','mean'),
    inflow_mean=(inflow_col,'mean'),
    n=('ID','count')
).reset_index()
layout_stats['ratio'] = layout_stats['y_mean'] / layout_stats['pred_mean']
print("Layout-level: y/pred ratio by inflow percentile:")
layout_stats_sorted = layout_stats.sort_values('inflow_mean')
for q in [10, 25, 50, 75, 90, 100]:
    n = int(len(layout_stats_sorted) * q / 100)
    if n == 0: n = 1
    if n > len(layout_stats_sorted): n = len(layout_stats_sorted)
    subset = layout_stats_sorted.head(n)
    print(f"  Bottom-{q:3d}pct (inflow<={subset.inflow_mean.max():.1f}): ratio={subset.ratio.mean():.3f}")

subset_high = layout_stats_sorted.tail(int(len(layout_stats_sorted)*0.1))
print(f"\nTop-10% inflow layouts ({subset_high.inflow_mean.min():.1f}+): ratio={subset_high.ratio.mean():.3f}  y_mean={subset_high.y_mean.mean():.2f}  pred_mean={subset_high.pred_mean.mean():.2f}")

# Estimate what ratio we'd expect for unseen inflow 172
# Fit ratio vs inflow
ratio_xs = layout_stats.inflow_mean.values
ratio_ys = layout_stats.ratio.values
r_coeffs = np.polyfit(ratio_xs, ratio_ys, 1)
print(f"\nRatio vs inflow linear fit: ratio = {r_coeffs[0]:.6f} * inflow + {r_coeffs[1]:.4f}")
for inf_val in [94, 120, 140, 158, 172]:
    pred_ratio = r_coeffs[0]*inf_val + r_coeffs[1]
    print(f"  inflow={inf_val}: predicted_ratio={pred_ratio:.3f}")

print("\n" + "="*70)
print("Part 4: Apply inflow-based calibration to unseen test rows")
print("="*70)

# Use fitted ratio to calibrate unseen predictions
# Current dual_t[unseen_mask].mean() = 23.479
# If ratio at inflow=172 is pred_ratio, apply calibration
pred_ratio_172 = r_coeffs[0]*172 + r_coeffs[1]
print(f"Predicted ratio at inflow=172: {pred_ratio_172:.4f}")

# Simple multiplicative calibration for unseen test rows
# Scale factor: pred_ratio_172 / overall_seen_ratio
overall_ratio = layout_stats['ratio'].mean()
print(f"Overall mean ratio: {overall_ratio:.4f}")
scale_factor = pred_ratio_172 / overall_ratio
print(f"Scale factor for unseen: {scale_factor:.4f}")

# Apply calibration to unseen test rows
t_calib = dual_t.copy()
t_calib[unseen_mask] = np.clip(dual_t[unseen_mask] * scale_factor, 0, None)
print(f"Calibrated: test={t_calib.mean():.3f}  seen={t_calib[seen_mask].mean():.3f}  unseen={t_calib[unseen_mask].mean():.3f}")

# Grid search calibration scales
print("\nGrid search calibration scale for unseen:")
for sf in [0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50]:
    t = dual_t.copy()
    t[unseen_mask] = np.clip(dual_t[unseen_mask] * sf, 0, None)
    print(f"  scale={sf:.2f}: test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}")

print("\n" + "="*70)
print("Part 5: Save calibrated unseen submissions")
print("="*70)

sub_template = pd.read_csv('sample_submission.csv')

# Use the linear ratio extrapolation
for sf in [pred_ratio_172, 1.10, 1.20]:
    t = dual_t.copy()
    t[unseen_mask] = np.clip(dual_t[unseen_mask] * sf, 0, None)
    sub = sub_template.copy()
    sub['avg_delay_minutes_next_30m'] = t
    fname = f"submission_inflowCalib_sf{sf:.3f}_OOF{dual_mae:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  unseen={t[unseen_mask].mean():.3f}  test={t.mean():.3f}")
