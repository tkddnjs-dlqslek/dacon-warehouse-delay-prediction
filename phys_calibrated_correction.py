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
print(f"Physical feature model raw mean: {phys_mean:+.4f}")

print("="*60)
print("Calibrated Physical Feature Corrections")
print("="*60)

def apply_corr(corrections, name, fname):
    lid_to_c = {lid: corrections[i] for i, lid in enumerate(unseen_lids)}
    ct = oracle_new_t.copy()
    for lid in unseen_lids:
        m = (test_raw['layout_id'] == lid).values
        ct[m] = oracle_new_t[m] + lid_to_c[lid]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"  {name}: D={du:+.4f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)

# 1. Additive offset: phys_pred + (5.5 - phys_mean)
offset = 5.5 - phys_mean
phys_offset = phys_pred + offset
print(f"\n  Additive offset: raw + {offset:.4f}")
apply_corr(phys_offset, "physFeat+offset(5.5)", "FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv")

# 2. Scale to mean 5.5: phys_pred * (5.5/phys_mean)
scale = 5.5 / phys_mean
phys_scaled = phys_pred * scale
print(f"\n  Scale factor: {scale:.4f}")
apply_corr(phys_scaled, "physFeat*scale(5.5)", "FINAL_NEW_oN_physScale5p5_OOF8.3825.csv")

# 3. Blend: 0.5*phys_pred + 0.5*5.5 (shrink toward flat)
phys_blend50 = 0.5*phys_pred + 0.5*5.5
apply_corr(phys_blend50, "0.5*phys+0.5*5.5", "FINAL_NEW_oN_physBlend50_OOF8.3825.csv")

# 4. Blend: 0.3*phys_pred + 0.7*5.5
phys_blend30 = 0.3*phys_pred + 0.7*5.5
apply_corr(phys_blend30, "0.3*phys+0.7*5.5", "FINAL_NEW_oN_physBlend30_OOF8.3825.csv")

# Show per-layout comparison: raw vs offset vs scaled
print(f"\n  Per-layout correction comparison:")
print(f"  {'layout':12s}  {'oracle_t':8s}  {'phys_raw':8s}  {'offset':8s}  {'scaled':8s}  {'blend50':8s}")
for i in np.argsort(phys_pred):
    lid = unseen_lids[i]
    oN = oracle_new_t[(test_raw['layout_id']==lid).values].mean()
    print(f"  {lid:12s}  {oN:8.2f}  {phys_pred[i]:+8.3f}  {phys_offset[i]:+8.3f}  "
          f"{phys_scaled[i]:+8.3f}  {phys_blend50[i]:+8.3f}")

# ================================================================
# Final summary: all correction methods ranked by Δ
# ================================================================
print(f"\n{'='*70}")
print("FULL comparison table (all methods)")
print(f"{'='*70}")
all_files = [
    'submission_oracle_NEW_OOF8.3825.csv',
    'FINAL_NEW_oN_udelta5p5_OOF8.3825.csv',
    'FINAL_NEW_oN_udelta5p77_OOF8.3825.csv',
    'FINAL_NEW_oN_ridgeCorr_OOF8.3825.csv',
    'FINAL_NEW_oN_linearBias_OOF8.3825.csv',
    'FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv',
    'FINAL_NEW_oN_lvInflow_OOF8.3825.csv',
    'FINAL_NEW_oN_inflowQuantile_OOF8.3825.csv',
    'FINAL_NEW_oN_inflow2D_OOF8.3825.csv',
    'FINAL_NEW_oN_knn10_OOF8.3825.csv',
    'FINAL_NEW_oN_physFeat_OOF8.3825.csv',
    'FINAL_NEW_oN_physOffset5p5_OOF8.3825.csv',
    'FINAL_NEW_oN_physScale5p5_OOF8.3825.csv',
    'FINAL_NEW_oN_physBlend50_OOF8.3825.csv',
    'FINAL_NEW_oN_physBlend30_OOF8.3825.csv',
]
print(f"  {'file':50s}  {'seen':>8}  {'unseen':>8}  {'delta':>9}")
for fname in all_files:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        du = p[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
        print(f"  {fname[:50]:50s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {du:+9.3f}")
    except:
        print(f"  {fname[:50]:50s}  MISSING")

print("\nDone.")
