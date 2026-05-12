import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
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
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
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

# Load oracle_NEW test predictions
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot')]

train_raw['_resid'] = residuals_train
layout_grp = train_raw.groupby('layout_id')
layout_stats = layout_grp[feat_cols].mean().fillna(lambda x: x.median())
layout_stats = layout_stats.apply(lambda col: col.fillna(col.median()))
layout_resid = layout_grp['_resid'].mean()

test_layout_stats = test_raw.groupby('layout_id')[feat_cols].mean()
test_layout_stats = test_layout_stats.apply(lambda col: col.fillna(col.median()))

unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()
seen_test_layouts   = test_raw[seen_mask]['layout_id'].unique()

# ============================================================
# OOD check: top-5 features for unseen test vs training range
# ============================================================
top5 = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps',
        'packaging_material_cost', 'agv_task_success_rate']
print("="*70)
print("OOD Check: Top-5 residual-correlated features")
print("="*70)
print(f"\n  {'feature':35s}  {'train_min':>9}  {'train_max':>9}  {'train_mean':>10}  {'unseen_mean':>11}  {'seen_mean':>9}  {'extrapolating?'}")
print("-"*115)
for f in top5:
    if f not in feat_cols:
        continue
    tr_vals = layout_stats[f].values
    u_vals  = test_layout_stats.loc[unseen_test_layouts, f].values if f in test_layout_stats.columns else np.array([])
    s_vals  = test_layout_stats.loc[seen_test_layouts, f].values if f in test_layout_stats.columns else np.array([])
    tr_min, tr_max, tr_mean = tr_vals.min(), tr_vals.max(), tr_vals.mean()
    u_mean = u_vals.mean() if len(u_vals) > 0 else np.nan
    s_mean = s_vals.mean() if len(s_vals) > 0 else np.nan
    ood = "YES !" if (not np.isnan(u_mean) and (u_mean > tr_max or u_mean < tr_min)) else "no"
    print(f"  {f:35s}  {tr_min:9.3f}  {tr_max:9.3f}  {tr_mean:10.3f}  {u_mean:11.3f}  {s_mean:9.3f}  {ood}")

# ============================================================
# R² with each feature independently — inflow vs pack_util
# ============================================================
print("\n" + "="*70)
print("Independent feature regressions: R² and extrapolation")
print("="*70)
from sklearn.linear_model import LinearRegression

for f in top5[:5]:
    if f not in feat_cols:
        continue
    x = layout_stats[f].values.reshape(-1,1)
    y = layout_resid.values
    lr = LinearRegression().fit(x, y)
    r2 = lr.score(x, y)
    u_val = test_layout_stats.loc[unseen_test_layouts, f].mean()
    s_val = test_layout_stats.loc[seen_test_layouts, f].mean()
    pred_u = lr.predict([[u_val]])[0]
    pred_s = lr.predict([[s_val]])[0]
    print(f"\n  {f}: coef={lr.coef_[0]:.4f}  intercept={lr.intercept_:.4f}  R²={r2:.4f}")
    print(f"    Predicted residual: unseen_mean({u_val:.3f}) → {pred_u:+.4f}  seen_mean({s_val:.3f}) → {pred_s:+.4f}")

# ============================================================
# 2-feature ridge: pack_utilization + outbound_truck_wait_min
# ============================================================
print("\n" + "="*70)
print("2-feature Ridge: pack_utilization + outbound_truck_wait_min")
print("="*70)
f2 = ['pack_utilization', 'outbound_truck_wait_min']
X2 = layout_stats[f2].values
sc2 = StandardScaler()
X2s = sc2.fit_transform(X2)
r2f = Ridge(alpha=1.0).fit(X2s, layout_resid.values)
pred_train2 = r2f.predict(X2s)
print(f"  Train R² = {r2f.score(X2s, layout_resid.values):.4f}")
u_X2 = sc2.transform(test_layout_stats.loc[unseen_test_layouts, f2].values)
s_X2 = sc2.transform(test_layout_stats.loc[seen_test_layouts, f2].values)
pred_u2 = r2f.predict(u_X2)
pred_s2 = r2f.predict(s_X2)
print(f"  Unseen predicted residual: mean={pred_u2.mean():.4f}  std={pred_u2.std():.4f}")
print(f"  Seen predicted residual:   mean={pred_s2.mean():.4f}  std={pred_s2.std():.4f}")

# Apply 2-feature correction to unseen test rows
ct_2feat = oracle_new_t.copy()
layout_to_2f = dict(zip(unseen_test_layouts, pred_u2))
for lid, corr in layout_to_2f.items():
    row_mask = (test_raw['layout_id'] == lid).values
    ct_2feat[row_mask] = oracle_new_t[row_mask] + corr
ct_2feat = np.clip(ct_2feat, 0, None)
du_2f = ct_2feat[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"\n  2-feature correction: seen={ct_2feat[seen_mask].mean():.3f}  unseen={ct_2feat[unseen_mask].mean():.3f}  Δ={du_2f:+.4f}")
fname2f = "FINAL_NEW_oN_2featCorr_OOF8.3825.csv"
sub2f = sub_tmpl.copy(); sub2f['avg_delay_minutes_next_30m'] = ct_2feat
sub2f.to_csv(fname2f, index=False)
print(f"  Saved: {fname2f}")

# ============================================================
# Scatter: pack_utilization vs residual for training layouts
# Show unseen test layout range
# ============================================================
print("\n" + "="*70)
print("pack_utilization: training layout distribution vs unseen test")
print("="*70)
pu_tr = layout_stats['pack_utilization'].values
pu_u  = test_layout_stats.loc[unseen_test_layouts, 'pack_utilization'].values
pu_s  = test_layout_stats.loc[seen_test_layouts, 'pack_utilization'].values
resid = layout_resid.values

# Bin by pack_utilization
bins_pu = np.percentile(pu_tr, [0, 20, 40, 60, 80, 100])
print(f"\n  Training layout pack_utilization: min={pu_tr.min():.4f}  max={pu_tr.max():.4f}  mean={pu_tr.mean():.4f}")
print(f"  Unseen test:                     min={pu_u.min():.4f}  max={pu_u.max():.4f}  mean={pu_u.mean():.4f}")
print(f"  Seen test:                       min={pu_s.min():.4f}  max={pu_s.max():.4f}  mean={pu_s.mean():.4f}")
print(f"\n  Training quintile summary (pack_utilization vs resid):")
print(f"  {'bin':18s}  {'n_layouts':>9}  {'mean_resid':>10}  {'% unseen test in this bin'}")
prev_lo = bins_pu[0]
for i, hi in enumerate(bins_pu[1:]):
    mask = (pu_tr >= prev_lo) & (pu_tr <= hi) if i == len(bins_pu)-2 else (pu_tr >= prev_lo) & (pu_tr < hi)
    if mask.sum() > 0:
        mr = resid[mask].mean()
        n_u_in = ((pu_u >= prev_lo) & (pu_u <= hi)).sum() if i == len(bins_pu)-2 else ((pu_u >= prev_lo) & (pu_u < hi)).sum()
        pct_u = 100*n_u_in/len(pu_u) if len(pu_u) > 0 else 0
        print(f"  [{prev_lo:.4f}, {hi:.4f}): n={mask.sum():9d}  resid={mr:10.4f}  ({pct_u:.0f}% of unseen test)")
    prev_lo = hi

# Check: unseen test pack_utilization is OOD?
print(f"\n  Max training pack_util: {pu_tr.max():.4f}")
print(f"  Unseen test exceeding training max: {(pu_u > pu_tr.max()).sum()} / {len(pu_u)} layouts")
print(f"  Fraction of unseen test rows with pack_util > training max:")
pu_test_all = test_raw['pack_utilization'].fillna(test_raw['pack_utilization'].median()).values
pct_ood = 100 * (pu_test_all[unseen_mask] > pu_tr.max()).mean()
print(f"  {pct_ood:.1f}%")

# ============================================================
# Conservative correction: cap the extrapolation at training max
# ============================================================
print("\n" + "="*70)
print("Capped extrapolation: only use features within training range")
print("Clip unseen layout feature values to training max before predicting")
print("="*70)
top5_cap = ['pack_utilization', 'outbound_truck_wait_min', 'conveyor_speed_mps',
            'packaging_material_cost', 'agv_task_success_rate']
top5_avail = [f for f in top5_cap if f in feat_cols and f in test_layout_stats.columns]

X_tr5 = layout_stats[top5_avail].values
sc5 = StandardScaler()
X_tr5s = sc5.fit_transform(X_tr5)
ridge5 = Ridge(alpha=5.0).fit(X_tr5s, layout_resid.values)
print(f"  5-feature Ridge R² = {ridge5.score(X_tr5s, layout_resid.values):.4f}")

# Clip unseen to training range
X_u5 = test_layout_stats.loc[unseen_test_layouts, top5_avail].values.copy()
for j, f in enumerate(top5_avail):
    tr_min, tr_max = layout_stats[f].min(), layout_stats[f].max()
    X_u5[:, j] = np.clip(X_u5[:, j], tr_min, tr_max)
X_u5s = sc5.transform(X_u5)
pred_u5_capped = ridge5.predict(X_u5s)
print(f"  Capped 5-feature correction (unseen): mean={pred_u5_capped.mean():.4f}  std={pred_u5_capped.std():.4f}")

ct_5cap = oracle_new_t.copy()
layout_to_5cap = dict(zip(unseen_test_layouts, pred_u5_capped))
for lid, corr in layout_to_5cap.items():
    row_mask = (test_raw['layout_id'] == lid).values
    ct_5cap[row_mask] = oracle_new_t[row_mask] + corr
ct_5cap = np.clip(ct_5cap, 0, None)
du_5cap = ct_5cap[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
print(f"  Capped 5-feat correction: seen={ct_5cap[seen_mask].mean():.3f}  unseen={ct_5cap[unseen_mask].mean():.3f}  Δ={du_5cap:+.4f}")
fname5cap = "FINAL_NEW_oN_5featCapped_OOF8.3825.csv"
sub5c = sub_tmpl.copy(); sub5c['avg_delay_minutes_next_30m'] = ct_5cap
sub5c.to_csv(fname5cap, index=False)
print(f"  Saved: {fname5cap}")

print("\nDone.")
