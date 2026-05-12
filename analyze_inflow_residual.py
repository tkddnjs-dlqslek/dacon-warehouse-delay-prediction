import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

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

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t   = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t   = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

fx_orig_oof = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fx_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
oracle_oof = np.clip(0.64*fx_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fx_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

fx_rw_oof = wm_best*((0.75*mega33_oof+0.25*mega34_oof)) + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
fx_rw_test = wm_best*((0.75*mega33_test+0.25*mega34_test)) + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test
wf=0.72; w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
best_base_oof = np.clip((1-0.12)*(wf*fx_rw_oof+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o) + 0.12*cb_oof, 0, None)
best_base_test = np.clip((1-0.12)*(wf*fx_rw_test+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t) + 0.12*cb_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
best_base_v = mae(best_base_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'best_base (m34_rw_wf072+cb12): OOF={best_base_v:.5f}  test_mean={best_base_test.mean():.3f}')

# Row-level inflow residual analysis
train_raw['oracle_pred'] = oracle_oof
train_raw['residual'] = y_true - np.clip(oracle_oof, 0, None)
train_raw['inflow'] = train_raw['order_inflow_15m']
train_raw['pack'] = train_raw['pack_utilization']

print('\n=== Row-level: oracle residual by inflow quantile ===')
q25, q50, q75, q90, q95 = train_raw['inflow'].quantile([0.25, 0.5, 0.75, 0.90, 0.95])
print(f'Training inflow quantiles: p25={q25:.1f}  p50={q50:.1f}  p75={q75:.1f}  p90={q90:.1f}  p95={q95:.1f}')
inflow_max = train_raw['inflow'].max()
print(f'Training inflow max (per-row): {inflow_max:.1f}')

for label, mask in [
    ('all', train_raw['inflow'] >= 0),
    ('inflow<p25', train_raw['inflow'] < q25),
    ('p25-p50', (train_raw['inflow'] >= q25) & (train_raw['inflow'] < q50)),
    ('p50-p75', (train_raw['inflow'] >= q50) & (train_raw['inflow'] < q75)),
    ('p75-p90', (train_raw['inflow'] >= q75) & (train_raw['inflow'] < q90)),
    ('p90-p95', (train_raw['inflow'] >= q90) & (train_raw['inflow'] < q95)),
    ('inflow>p95', train_raw['inflow'] >= q95),
]:
    sub = train_raw[mask]
    resid = sub['residual'].mean()
    y_m = sub['avg_delay_minutes_next_30m'].mean()
    pred_m = sub['oracle_pred'].mean()
    print(f'  {label:12s}: n={len(sub):6d}  y_mean={y_m:7.2f}  pred={pred_m:7.2f}  resid={resid:+7.2f}  corr_pack={np.corrcoef(sub["pack"],sub["residual"])[0,1]:.3f}')

# Fit a correction model on inflow + pack
print('\n=== Inflow-based row-level correction model (CV) ===')
# Features: inflow, pack, inflow^2, pack^2, inflow*pack
X_train = np.column_stack([
    train_raw['inflow'].values,
    train_raw['pack'].values,
    train_raw['inflow'].values**2,
    train_raw['pack'].values**2,
    train_raw['inflow'].values * train_raw['pack'].values
])
y_resid = y_true - np.clip(oracle_oof, 0, None)

# Normalize
X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-8
X_norm = (X_train - X_mean) / X_std

# CV to estimate residual model performance
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
resid_preds = np.zeros(len(y_true))
for fi, (train_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    reg = Ridge(alpha=1.0)
    reg.fit(X_norm[train_idx], y_resid[train_idx])
    resid_preds[val_idx] = reg.predict(X_norm[val_idx])

corr_resid = np.corrcoef(y_resid, resid_preds)[0,1]
print(f'Residual model: corr(y_resid, pred_resid)={corr_resid:.4f}')
corrected_oof = np.clip(oracle_oof + resid_preds, 0, None)
print(f'Corrected OOF: {mae(corrected_oof):.5f}  delta={mae(corrected_oof)-base_oof:+.6f}')

# Apply to test at small weight
from sklearn.linear_model import Ridge
reg_full = Ridge(alpha=1.0)
reg_full.fit(X_norm, y_resid)
X_test = np.column_stack([
    test_raw['order_inflow_15m'].values,
    test_raw['pack_utilization'].values,
    test_raw['order_inflow_15m'].values**2,
    test_raw['pack_utilization'].values**2,
    test_raw['order_inflow_15m'].values * test_raw['pack_utilization'].values
])
X_test_norm = (X_test - X_mean) / X_std
resid_test = reg_full.predict(X_test_norm)
print(f'Test residual predictions: mean={resid_test.mean():.3f}  std={resid_test.std():.3f}  max={resid_test.max():.3f}')
print(f'Unseen layout correction: {resid_test[test_raw["layout_id"].isin(set(test_raw["layout_id"]) - set(train_raw["layout_id"]))].mean():.3f}')

# Apply correction at small weight
for alpha in [0.1, 0.2, 0.3, 0.5, 1.0]:
    corr = np.clip(oracle_test + alpha*resid_test, 0, None)
    print(f'  alpha={alpha:.1f}: test_mean={corr.mean():.3f}  delta={corr.mean()-oracle_test.mean():+.3f}')

# Simple inflow-threshold correction on test
print('\n=== Simple inflow threshold correction on test ===')
# Compute mean residual from TRAINING for inflow>X
for thresh in [100, 110, 120, 128]:
    mask = train_raw['inflow'] > thresh
    mean_resid = y_resid[mask].mean()
    # Apply to test rows with inflow > thresh
    test_mask = test_raw['order_inflow_15m'] > thresh
    n_affected = test_mask.sum()
    corrected = oracle_test.copy()
    corrected[test_mask] = np.clip(oracle_test[test_mask] + mean_resid, 0, None)
    print(f'  thresh>{thresh}: train_resid={mean_resid:+.3f}  n_test={n_affected}({n_affected/len(test_raw)*100:.1f}%)  new_test_mean={corrected.mean():.3f}')

# Same for unseen layouts only
print('\n=== Unseen-layout specific analysis ===')
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
seen_mask = test_raw['layout_id'].isin(train_layouts)

print(f'Unseen test rows: {unseen_mask.sum()}  Seen test rows: {seen_mask.sum()}')
print(f'Oracle: unseen_mean={oracle_test[unseen_mask].mean():.3f}  seen_mean={oracle_test[seen_mask].mean():.3f}')
print(f'Residual correction: unseen_mean={resid_test[unseen_mask].mean():.3f}  seen_mean={resid_test[seen_mask].mean():.3f}')

# Key result: inflow distribution comparison train vs unseen test
train_inflow_max = train_raw['inflow'].max()
print(f'\nTraining inflow max per-row: {train_inflow_max:.1f}')
test_high_inflow_mask = test_raw['order_inflow_15m'] > train_inflow_max
n_hi = test_high_inflow_mask.sum()
print(f'Test rows with inflow > training max: {n_hi} ({n_hi/len(test_raw)*100:.1f}%)')
if n_hi > 0:
    hi_oracle = oracle_test[test_high_inflow_mask].mean()
    hi_inflow = test_raw.loc[test_high_inflow_mask,'order_inflow_15m'].mean()
    print(f'  These rows: oracle_mean={hi_oracle:.3f}  inflow_mean={hi_inflow:.1f}')

# Try residual correction only on unseen layouts
for alpha in [0.3, 0.5, 1.0, 2.0]:
    corr_unseen = oracle_test.copy()
    corr_unseen[unseen_mask] = np.clip(oracle_test[unseen_mask] + alpha*resid_test[unseen_mask], 0, None)
    print(f'  unseen-only alpha={alpha:.1f}: test_mean={corr_unseen.mean():.3f}  delta={corr_unseen.mean()-oracle_test.mean():+.3f}  unseen_mean={corr_unseen[unseen_mask].mean():.3f}')

print('\nDone.')
