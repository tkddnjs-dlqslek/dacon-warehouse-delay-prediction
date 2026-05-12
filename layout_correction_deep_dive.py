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

# Training layout stats
tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'),
    resid_mean=('_resid','mean'),
    ytrue_mean=('avg_delay_minutes_next_30m','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

# Unseen test layout stats
unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'),
    pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
    inflow_mean=('order_inflow_15m','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

# Fit combined model (oof+phys3, alpha=100)
X_tr = tr_lv[['oof_mean','pu','tw','conv']].values
X_te = te_lv_u[['oN_mean','pu','tw','conv']].values
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)
reg = Ridge(alpha=100); reg.fit(X_tr_s, y_lv)
combined_raw = reg.predict(X_te_s)
combined_offset = 5.5 - combined_raw.mean()
combined_corr = combined_raw + combined_offset
lid_to_combined = dict(zip(te_lv_u['layout_id'], combined_corr))

# Load iso corrections
iso_df = pd.read_csv('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv')
iso_t = iso_df.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
iso_by_lid = {lid: iso_t[test_raw['layout_id']==lid].mean() - oracle_new_t[test_raw['layout_id']==lid].mean()
              for lid in unseen_lids}

# Load phys3 corrections
phys3_df = pd.read_csv('FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv')
phys3_t = phys3_df.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
phys3_by_lid = {lid: phys3_t[test_raw['layout_id']==lid].mean() - oracle_new_t[test_raw['layout_id']==lid].mean()
                for lid in unseen_lids}

print("="*70)
print("Per-Layout Correction Deep Dive: All 50 Unseen Layouts")
print("="*70)

# Build full comparison table
te_lv_u['iso'] = te_lv_u['layout_id'].map(iso_by_lid)
te_lv_u['phys3'] = te_lv_u['layout_id'].map(phys3_by_lid)
te_lv_u['combined'] = te_lv_u['layout_id'].map(lid_to_combined)
te_lv_u['diff_ip'] = te_lv_u['iso'] - te_lv_u['phys3']
te_lv_u['diff_ic'] = te_lv_u['iso'] - te_lv_u['combined']

# Sort by combined correction
df_sorted = te_lv_u.sort_values('combined')
print(f"\n  All 50 unseen layouts sorted by combined correction:")
print(f"  {'layout_id':10s}  {'oN_mean':7s}  {'pu':5s}  {'tw':5s}  {'conv':5s}  {'inflow':7s}  "
      f"{'iso':6s}  {'phys3':6s}  {'comb':6s}  {'d_ic':6s}")
for _, row in df_sorted.iterrows():
    print(f"  {row['layout_id']:10s}  {row['oN_mean']:7.2f}  {row['pu']:.3f}  {row['tw']:.3f}  "
          f"{row['conv']:.3f}  {row['inflow_mean']:7.2f}  "
          f"{row['iso']:6.2f}  {row['phys3']:6.2f}  {row['combined']:6.2f}  {row['diff_ic']:6.2f}")

# In training: find layouts with similar characteristics to the disagreement cases
print(f"\n{'='*70}")
print("Training layouts: Low oof_mean but High physical features (iso-underestimates)")
print(f"{'='*70}")
# Quadrant: oof_mean < median AND (pu > median OR tw > median)
oof_med = tr_lv['oof_mean'].median()
pu_med = tr_lv['pu'].median()
tr_lv['low_oof'] = tr_lv['oof_mean'] < oof_med
tr_lv['high_pu'] = tr_lv['pu'] > pu_med

q_lowoof_highpu = tr_lv[(tr_lv['oof_mean'] < oof_med) & (tr_lv['pu'] > pu_med)]
q_highoof_lowpu = tr_lv[(tr_lv['oof_mean'] >= oof_med) & (tr_lv['pu'] <= pu_med)]
q_highoof_highpu = tr_lv[(tr_lv['oof_mean'] >= oof_med) & (tr_lv['pu'] > pu_med)]
q_lowoof_lowpu = tr_lv[(tr_lv['oof_mean'] < oof_med) & (tr_lv['pu'] <= pu_med)]

print(f"\n  Quadrant analysis (training layouts, n=250):")
print(f"  {'quadrant':25s}  n    oof_mean  pu_mean  resid_mean  iso_vs_phys")
for quad_name, quad in [
    ('low_oof + high_pu', q_lowoof_highpu),
    ('high_oof + low_pu', q_highoof_lowpu),
    ('high_oof + high_pu', q_highoof_highpu),
    ('low_oof + low_pu', q_lowoof_lowpu)]:
    n = len(quad)
    oof_m = quad['oof_mean'].mean()
    pu_m = quad['pu'].mean()
    resid_m = quad['resid_mean'].mean()
    print(f"  {quad_name:25s}: n={n:3d}  oof={oof_m:.2f}  pu={pu_m:.3f}  resid={resid_m:.3f}")

# For this quadrant, check LOOO predictions
print(f"\n  LOOO errors per quadrant:")
preds_looo = []
for i in range(len(tr_lv)):
    Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y_lv, i)
    sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_)
    Xte_s = sc_.transform(X_tr[i:i+1])
    lr = Ridge(alpha=100); lr.fit(Xtr_s, ytr_)
    preds_looo.append(lr.predict(Xte_s)[0])
tr_lv['looo_pred'] = preds_looo
tr_lv['looo_err'] = tr_lv['resid_mean'] - tr_lv['looo_pred']

for quad_name, quad_idx in [
    ('low_oof + high_pu', (tr_lv['oof_mean'] < oof_med) & (tr_lv['pu'] > pu_med)),
    ('high_oof + low_pu', (tr_lv['oof_mean'] >= oof_med) & (tr_lv['pu'] <= pu_med)),
    ('high_oof + high_pu', (tr_lv['oof_mean'] >= oof_med) & (tr_lv['pu'] > pu_med)),
    ('low_oof + low_pu', (tr_lv['oof_mean'] < oof_med) & (tr_lv['pu'] <= pu_med))]:
    q = tr_lv[quad_idx]
    mae = q['looo_err'].abs().mean()
    bias = q['looo_err'].mean()
    print(f"  {quad_name:25s}: n={len(q):3d}  LOOO_MAE={mae:.3f}  bias={bias:+.3f}")

# WH_193, WH_214 type analysis: find training analogues
print(f"\n  Training analogues of WH_193 (iso+1.2, phys+7.9): low oN, high pu, high tw")
# Low oof_mean (<12), high pack_util (>0.50), high truck_wait (>17)
analogue_193 = tr_lv[(tr_lv['oof_mean'] < 12) & (tr_lv['pu'] > 0.50)]
if len(analogue_193) > 0:
    print(f"  Found {len(analogue_193)} layouts:")
    print(analogue_193[['layout_id','oof_mean','pu','tw','conv','inflow_mean','resid_mean']].round(3).to_string())
else:
    print(f"  No exact analogues. Closest:")
    tr_lv['dist_193'] = (tr_lv['oof_mean']/tr_lv['oof_mean'].max())**2 + (tr_lv['pu']- 0.65)**2
    print(tr_lv.nsmallest(5, 'dist_193')[['layout_id','oof_mean','pu','tw','inflow_mean','resid_mean']].round(3).to_string())

print(f"\n  Training analogues of WH_234 (iso+12.4, phys+1.2): high oN, low pu, low tw")
# High oof_mean (>25), low pack_util (<0.10)
analogue_234 = tr_lv[(tr_lv['oof_mean'] > 20) & (tr_lv['pu'] < 0.15)]
if len(analogue_234) > 0:
    print(f"  Found {len(analogue_234)} layouts:")
    print(analogue_234[['layout_id','oof_mean','pu','tw','conv','inflow_mean','resid_mean']].round(3).to_string())
else:
    print(f"  No exact analogues. Closest:")
    tr_lv['dist_234'] = (tr_lv['pu'] - 0.042)**2 * 50 + (tr_lv['oof_mean'] - 30)**2 / 100
    print(tr_lv.nsmallest(5, 'dist_234')[['layout_id','oof_mean','pu','tw','inflow_mean','resid_mean']].round(3).to_string())

# Final summary of which method to trust
print(f"\n{'='*70}")
print("Summary: correction distribution comparison")
print(f"{'='*70}")
print(f"\n  Per correction method, distribution for 50 unseen layouts:")
from scipy.stats import pearsonr
from scipy.stats import spearmanr
r_iso_comb, _ = spearmanr(te_lv_u['iso'], te_lv_u['combined'])
r_phys_comb, _ = spearmanr(te_lv_u['phys3'], te_lv_u['combined'])
r_iso_phys, _ = spearmanr(te_lv_u['iso'], te_lv_u['phys3'])
print(f"  Spearman r(iso, combined) = {r_iso_comb:.4f}")
print(f"  Spearman r(phys, combined) = {r_phys_comb:.4f}")
print(f"  Spearman r(iso, phys) = {r_iso_phys:.4f}")

print(f"\n  Correction ranges for unseen test:")
print(f"  iso:      [{te_lv_u['iso'].min():.2f}, {te_lv_u['iso'].max():.2f}]  std={te_lv_u['iso'].std():.3f}")
print(f"  phys3:    [{te_lv_u['phys3'].min():.2f}, {te_lv_u['phys3'].max():.2f}]  std={te_lv_u['phys3'].std():.3f}")
print(f"  combined: [{te_lv_u['combined'].min():.2f}, {te_lv_u['combined'].max():.2f}]  std={te_lv_u['combined'].std():.3f}")

print("\nDone.")
