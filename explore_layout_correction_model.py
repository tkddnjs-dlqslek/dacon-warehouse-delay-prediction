import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

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
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

# Load oracle predictions
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
oracle5_o = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)
oracle5_t = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)

# Reconstruct oracle_NEW OOF
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
cb_test_mega=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o_raw=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_t_raw=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2_*r2_test+w3_*r3_test
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o_raw+0.10*xgbc_o+0.08*mono_o,0,None)
bb_tt=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem2*rem_t,0,None)
bb_tt=np.clip((1-w_cb)*bb_tt+w_cb*cb_test_mega,0,None)
fw4_tt=np.clip(0.74*bb_tt+0.08*slh_t_raw+0.10*xgbc_t+0.08*mono_t,0,None)

# Load oracle_NEW test
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

print("="*70)
print("PER-LAYOUT analysis: oracle_NEW OOF vs y_true")
print("="*70)

train_layouts_list = sorted(train_raw['layout_id'].unique())
layout_data = []
for lay in train_layouts_list:
    mask = (train_raw['layout_id'] == lay).values
    y_lay = y_true[mask]
    p_lay = fw4_oo[mask]
    o5_lay = oracle5_o[mask]
    y_mean = y_lay.mean()
    p_mean = p_lay.mean()
    o5_mean = o5_lay.mean()
    resid_oN = p_mean - y_mean  # oracle_NEW: negative = underpredicting
    resid_o5 = o5_mean - y_mean
    n = mask.sum()
    # Also get inflow
    if 'order_inflow_15m' in train_raw.columns:
        inflow_mean = train_raw.loc[mask, 'order_inflow_15m'].mean()
    else:
        inflow_mean = np.nan
    layout_data.append({
        'layout_id': lay, 'n': n, 'y_mean': y_mean, 'p_mean': p_mean,
        'o5_mean': o5_mean, 'resid_oN': resid_oN, 'resid_o5': resid_o5,
        'inflow_mean': inflow_mean
    })

df_lay = pd.DataFrame(layout_data)
print(f"\nLayout-level statistics:")
print(f"  n_layouts: {len(df_lay)}")
print(f"  p_mean range: [{df_lay.p_mean.min():.2f}, {df_lay.p_mean.max():.2f}]")
print(f"  y_mean range: [{df_lay.y_mean.min():.2f}, {df_lay.y_mean.max():.2f}]")
print(f"  resid_oN: mean={df_lay.resid_oN.mean():.3f}  std={df_lay.resid_oN.std():.3f}")

# Correlation between prediction level and residual
r_pred_resid, p_val = pearsonr(df_lay.p_mean, df_lay.resid_oN)
r_inflow_resid, _ = pearsonr(df_lay.inflow_mean.dropna(), df_lay.resid_oN[df_lay.inflow_mean.notna()])
r_y_resid, _ = pearsonr(df_lay.y_mean, df_lay.resid_oN)

print(f"\n  r(p_mean, resid_oN) = {r_pred_resid:.4f}  (prediction level predicts residual?)")
print(f"  r(y_mean, resid_oN) = {r_y_resid:.4f}  (true level predicts residual?)")
print(f"  r(inflow_mean, resid_oN) = {r_inflow_resid:.4f}  (inflow predicts residual?)")

# Simple linear model: resid = a * p_mean + b
print(f"\n  Linear fit: resid = a*p_mean + b (on training layouts)")
A = np.column_stack([df_lay.p_mean, np.ones(len(df_lay))])
b_vec = df_lay.resid_oN.values
coeff, _, _, _ = np.linalg.lstsq(A, b_vec, rcond=None)
a, b = coeff
print(f"  a={a:.4f}  b={b:.4f}")
pred_resid_linear = a * df_lay.p_mean + b
r2_linear = 1 - np.var(df_lay.resid_oN - pred_resid_linear) / np.var(df_lay.resid_oN)
print(f"  R² = {r2_linear:.4f}")

# Quadratic fit
A2 = np.column_stack([df_lay.p_mean**2, df_lay.p_mean, np.ones(len(df_lay))])
coeff2, _, _, _ = np.linalg.lstsq(A2, b_vec, rcond=None)
a2, a1, a0 = coeff2
print(f"\n  Quadratic fit: resid = {a2:.5f}*p² + {a1:.4f}*p + {a0:.4f}")
pred_resid_quad = a2 * df_lay.p_mean**2 + a1 * df_lay.p_mean + a0
r2_quad = 1 - np.var(df_lay.resid_oN - pred_resid_quad) / np.var(df_lay.resid_oN)
print(f"  R² = {r2_quad:.4f}")

print()
print("="*70)
print("Apply layout-level linear correction to unseen test")
print("="*70)

# For each unseen test layout, predict correction from oracle_NEW test mean prediction
unseen_layouts_test = sorted(test_raw[unseen_mask]['layout_id'].unique())
layout_test_data = []
for lay in unseen_layouts_test:
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    p_lay_mean = oracle_new_t[mask].mean()
    predicted_correction = -a * p_lay_mean - b  # negate since resid = pred-y, correction = -resid
    layout_test_data.append({
        'layout_id': lay, 'n': mask.sum(), 'p_mean': p_lay_mean,
        'pred_correction_linear': predicted_correction,
        'pred_correction_quad': -(a2*p_lay_mean**2 + a1*p_lay_mean + a0)
    })

df_test_lay = pd.DataFrame(layout_test_data)
print(f"\nPredicted corrections for unseen test layouts (top-10 by p_mean):")
print(f"  {'layout':15s} {'n':>5} {'p_mean':>8} {'linear_corr':>12} {'quad_corr':>10}")
for _, row in df_test_lay.sort_values('p_mean', ascending=False).head(10).iterrows():
    print(f"  {row.layout_id:15s} {row.n:5.0f} {row.p_mean:8.3f} {row.pred_correction_linear:+12.3f} {row.pred_correction_quad:+10.3f}")

# Apply linear correction to oracle_NEW test
ct_linear = oracle_new_t.copy()
for _, row in df_test_lay.iterrows():
    lay = row['layout_id']
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    correction = np.clip(row['pred_correction_linear'], -5, 20)  # cap correction
    ct_linear[mask] += correction
ct_linear = np.clip(ct_linear, 0, None)
print(f"\nLinear correction applied: seen={ct_linear[seen_mask].mean():.3f}  unseen={ct_linear[unseen_mask].mean():.3f}")

# Apply at partial fractions
sub_tmpl = pd.read_csv('sample_submission.csv')
for frac in [0.1, 0.2, 0.3, 0.5]:
    ct = oracle_new_t.copy()
    for _, row in df_test_lay.iterrows():
        lay = row['layout_id']
        mask = (test_raw['layout_id'] == lay).values & unseen_mask
        correction = np.clip(frac * row['pred_correction_linear'], -5, 15)
        ct[mask] += correction
    ct = np.clip(ct, 0, None)
    print(f"  linear_{frac:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
    if frac in [0.2, 0.3]:
        fname = f"FINAL_NEW_oN_linCorr{int(frac*10)}_OOF8.3825.csv"
        sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
        sub.to_csv(fname, index=False)
        print(f"    Saved: {fname}")

# Full linear correction with cap
ct = oracle_new_t.copy()
for _, row in df_test_lay.iterrows():
    lay = row['layout_id']
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    correction = np.clip(row['pred_correction_linear'], -3, 15)
    ct[mask] += correction
ct = np.clip(ct, 0, None)
fname = f"FINAL_NEW_oN_linCorrFull_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
sub.to_csv(fname, index=False)
print(f"\nSaved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print()
print("="*70)
print("SUMMARY: What's the best unseen mean to target?")
print("="*70)
print(f"\noracle_NEW training bias: mean = {df_lay.resid_oN.mean():.3f} per layout")
print(f"  → correction needed: +{-df_lay.resid_oN.mean():.3f} minutes on average")
print(f"\nFor unseen test layouts (n={len(unseen_layouts_test)}):")
print(f"  oracle_NEW mean prediction: {df_test_lay.p_mean.mean():.3f}")
print(f"  Linear correction (full): +{df_test_lay.pred_correction_linear.mean():.3f}")
print(f"  → Corrected unseen mean: {df_test_lay.p_mean.mean() + df_test_lay.pred_correction_linear.mean():.3f}")
print(f"  Quad correction (full): +{df_test_lay.pred_correction_quad.mean():.3f}")
print(f"  → Quad corrected unseen mean: {df_test_lay.p_mean.mean() + df_test_lay.pred_correction_quad.mean():.3f}")

print(f"\nCompare with oracle_NEW unseen = {oracle_new_t[unseen_mask].mean():.3f}")
print(f"  Note: layout-level mean != row-level mean (due to within-layout distribution)")
print(f"  oracle_NEW unseen (row mean) = {oracle_new_t[unseen_mask].mean():.3f}")
print(f"  oracle_NEW unseen (layout mean of layout means) = {df_test_lay.p_mean.mean():.3f}")

print("\nDone.")
