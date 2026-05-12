import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
import lightgbm as lgb
from scipy.stats import pearsonr

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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

inflow_col = 'order_inflow_15m'
feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot')]

train_feat = train_raw[feat_cols].copy()
test_feat  = test_raw[feat_cols].copy()
for c in feat_cols:
    med = train_feat[c].median()
    train_feat[c] = train_feat[c].fillna(med)
    test_feat[c] = test_feat[c].fillna(med)

X_tr = train_feat.values.astype(np.float32)
X_te = test_feat.values.astype(np.float32)
inflow_tr = train_feat[inflow_col].values
inflow_te = test_feat[inflow_col].values

print("="*70)
print("High-Inflow Only LGB: train on top-X% inflow rows")
print("="*70)

params = {
    'objective': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 64,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 5.0,
    'reg_lambda': 5.0,
    'verbose': -1,
    'n_jobs': -1,
}

print(f"\n  Training inflow percentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"    p{p}: {np.percentile(inflow_tr, p):.1f}")

# Test different thresholds
print(f"\n  {'threshold':12s}  {'n_rows':>8}  {'OOF_MAE':>9}  {'unseen_mean':>12}  {'corr_oN':>8}")

for thr in [75, 100, 125, 150]:
    hi_mask = inflow_tr >= thr
    n = hi_mask.sum()
    if n < 1000:
        print(f"  inflow>={thr:3d}: only {n} rows — skip")
        continue

    X_hi = X_tr[hi_mask]
    y_hi = y_true[hi_mask]

    # Simple 5-fold on the high-inflow subset (no GroupKFold)
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_hi = np.zeros(n)
    for tr_i, va_i in kf.split(X_hi):
        ds_tr = lgb.Dataset(X_hi[tr_i], y_hi[tr_i], free_raw_data=False)
        ds_va = lgb.Dataset(X_hi[va_i], y_hi[va_i], free_raw_data=False)
        m = lgb.train(params, ds_tr, num_boost_round=200, valid_sets=[ds_va],
                      callbacks=[lgb.log_evaluation(-1)])
        oof_hi[va_i] = m.predict(X_hi[va_i])
    oof_hi = np.clip(oof_hi, 0, None)
    mae_hi = np.mean(np.abs(oof_hi - y_hi))

    # Train final model on all high-inflow data, predict test
    ds_all_hi = lgb.Dataset(X_hi, y_hi, free_raw_data=False)
    mdl_hi = lgb.train(params, ds_all_hi, num_boost_round=200, callbacks=[lgb.log_evaluation(-1)])
    te_hi = np.clip(mdl_hi.predict(X_te), 0, None)

    r_oN = np.corrcoef(te_hi[unseen_mask], oracle_new_t[unseen_mask])[0,1]
    print(f"  inflow>={thr:3d}: n={n:8d}  OOF_MAE={mae_hi:9.4f}  unseen={te_hi[unseen_mask].mean():12.3f}  r={r_oN:.4f}")

    # Blend: can this model help correct oracle_NEW for unseen rows with high inflow?
    # unseen test rows with inflow > thr
    hi_unseen = unseen_mask & (inflow_te >= thr)
    if hi_unseen.sum() > 0:
        mean_hi_oN = oracle_new_t[hi_unseen].mean()
        mean_hi_lgb = te_hi[hi_unseen].mean()
        print(f"    unseen rows with inflow>={thr}: n={hi_unseen.sum()}  oN={mean_hi_oN:.3f}  hi_lgb={mean_hi_lgb:.3f}")

print("\n" + "="*70)
print("Analysis: What training scenarios are most similar to unseen test?")
print("="*70)
# Check training rows with inflow > 150 (closest to test unseen inflow=167)
hi_150 = inflow_tr >= 150
print(f"\n  Training rows with inflow >= 150: {hi_150.sum()} ({100*hi_150.mean():.1f}%)")
print(f"  y_true mean for those rows: {y_true[hi_150].mean():.4f}")

# Compare vs test unseen (all)
print(f"  oracle_NEW test unseen mean: {oracle_new_t[unseen_mask].mean():.4f}")

# Distribution of y_true for high-inflow training rows
print(f"\n  y_true distribution for inflow>=150 training rows:")
for p in [10, 25, 50, 75, 90]:
    print(f"    p{p}: {np.percentile(y_true[hi_150], p):.3f}")
print(f"    mean: {y_true[hi_150].mean():.3f}")

# Compare oracle_NEW OOF for those high-inflow rows
# Need to rebuild fw4_oo
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

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

print(f"\n  oracle_NEW OOF for inflow>=150 training rows: {fw4_oo[hi_150].mean():.4f}")
print(f"  residual for inflow>=150: {(y_true[hi_150] - fw4_oo[hi_150]).mean():+.4f}")
print(f"  (vs overall residual: {(y_true - fw4_oo).mean():+.4f})")

# Per-quantile of inflow for training
print(f"\n  Residual by inflow quantile (training data):")
for q_lo, q_hi in [(0,25), (25,50), (50,75), (75,90), (90,100)]:
    thr_lo = np.percentile(inflow_tr, q_lo) if q_lo > 0 else -np.inf
    thr_hi = np.percentile(inflow_tr, q_hi)
    m = (inflow_tr >= thr_lo) & (inflow_tr < thr_hi)
    resid = (y_true[m] - fw4_oo[m]).mean()
    print(f"    Q[{q_lo:2d}-{q_hi:2d}] inflow=[{thr_lo:.0f},{thr_hi:.0f}): n={m.sum():7d}  y_mean={y_true[m].mean():.3f}  oof={fw4_oo[m].mean():.3f}  resid={resid:+.3f}")

print("\nDone.")
