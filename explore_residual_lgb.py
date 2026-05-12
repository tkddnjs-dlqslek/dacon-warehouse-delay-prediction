import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
import lightgbm as lgb
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
mega33_oof=d33['meta_avg_oof'][id2]; mega33_test=d33['meta_avg_test'][te_id2]
mega34_oof=d34['meta_avg_oof'][id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof=np.clip(d33['meta_oofs']['cb'][id2],0,None); cb_test=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]; rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

# Build base predictions (dual gate)
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
fx=wm*mega+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
bb_o=np.clip(wf*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
bb_t=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t,0,None)
bb_o=np.clip((1-w_cb)*bb_o+w_cb*cb_oof,0,None)
bb_t=np.clip((1-w_cb)*bb_t+w_cb*cb_test,0,None)
fw4_o=np.clip(0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
fw4_t=np.clip(0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t,0,None)
n1,w1,n2,w2=2000,0.15,5500,0.08
sfw=np.sort(fw4_o); sft=np.sort(fw4_t)
dual_o=fw4_o.copy()
dual_o[fw4_o>=sfw[-n1]]=(1-w1)*fw4_o[fw4_o>=sfw[-n1]]+w1*rh_o[fw4_o>=sfw[-n1]]
dual_o[fw4_o>=sfw[-n2]]=(1-w2)*dual_o[fw4_o>=sfw[-n2]]+w2*slhm_o[fw4_o>=sfw[-n2]]
dual_o=np.clip(dual_o,0,None)
dual_t=fw4_t.copy()
dual_t[fw4_t>=sft[-n1]]=(1-w1)*fw4_t[fw4_t>=sft[-n1]]+w1*rh_t[fw4_t>=sft[-n1]]
dual_t[fw4_t>=sft[-n2]]=(1-w2)*dual_t[fw4_t>=sft[-n2]]+w2*slhm_t[fw4_t>=sft[-n2]]
dual_t=np.clip(dual_t,0,None)
dual_mae=mae_fn(dual_o)
print(f"Base dual: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 1: Feature selection for residual model")
print("="*70)

# Features: all numeric, exclude target/ID/layout/scenario (group vars)
# Include dual_o prediction as meta-feature
exclude = {'ID','layout_id','scenario_id','avg_delay_minutes_next_30m'}
feat_cols = [c for c in train_raw.columns if c not in exclude and c != '_row_id']
print(f"Features: {len(feat_cols)} columns")
print(f"Sample: {feat_cols[:10]}")

X_train = train_raw[feat_cols].values.astype(np.float32)
X_test  = test_raw[feat_cols].values.astype(np.float32)

# Add base prediction as a feature
X_train = np.hstack([X_train, dual_o.reshape(-1,1)])
X_test  = np.hstack([X_test,  dual_t.reshape(-1,1)])

residual = y_true - dual_o

print(f"Residual stats: mean={residual.mean():.3f}  std={residual.std():.3f}")
print(f"  Positive residuals (underpredicted): {(residual>0).mean():.1%}")
print(f"  Negative residuals (overpredicted):  {(residual<0).mean():.1%}")

# ============================================================
print("\n" + "="*70)
print("Part 2: GroupKFold residual LightGBM (layout_id groups)")
print("="*70)

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

lgb_params = {
    'objective': 'mae',
    'n_estimators': 500,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_lambda': 1.0,
    'n_jobs': -1,
    'verbose': -1,
    'random_state': 42,
}

resid_oof = np.zeros(len(y_true))
resid_test_folds = []

for fi, (tr_idx, va_idx) in enumerate(gkf.split(X_train, residual, groups)):
    X_tr, X_va = X_train[tr_idx], X_train[va_idx]
    y_tr, y_va = residual[tr_idx], residual[va_idx]

    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr, y_tr,
              eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

    resid_oof[va_idx] = model.predict(X_va)
    resid_test_folds.append(model.predict(X_test))

    fold_oof_mae = float(np.mean(np.abs(np.clip(dual_o[va_idx]+resid_oof[va_idx],0,None)-y_true[va_idx])))
    fold_base_mae = float(np.mean(np.abs(dual_o[va_idx]-y_true[va_idx])))
    print(f"  Fold {fi}: base={fold_base_mae:.5f}  calib={fold_oof_mae:.5f}  delta={fold_oof_mae-fold_base_mae:+.5f}  best_iter={model.best_iteration_}")

resid_test = np.mean(resid_test_folds, axis=0)

calib_oof = np.clip(dual_o + resid_oof, 0, None)
calib_test = np.clip(dual_t + resid_test, 0, None)
calib_oof_mae = mae_fn(calib_oof)
print(f"\nCalibrated OOF MAE: {calib_oof_mae:.5f}  (base: {dual_mae:.5f}  delta: {calib_oof_mae-dual_mae:+.5f})")
print(f"Calibrated test: {calib_test.mean():.3f}  seen={calib_test[seen_mask].mean():.3f}  unseen={calib_test[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Alpha scaling of residual correction")
print("="*70)

print(f"\n{'alpha':>6}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
best_alpha, best_mae = 0.0, dual_mae
for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.00]:
    co = np.clip(dual_o + alpha*resid_oof, 0, None)
    ct = np.clip(dual_t + alpha*resid_test, 0, None)
    m = mae_fn(co)
    flag = " <-- BEST" if m < best_mae else ""
    print(f"  {alpha:>5.2f}  {m:>9.5f}  {ct.mean():>8.3f}  {ct[seen_mask].mean():>8.3f}  {ct[unseen_mask].mean():>8.3f}{flag}")
    if m < best_mae:
        best_mae = m
        best_alpha = alpha

print(f"\nBest alpha={best_alpha:.2f}  OOF={best_mae:.5f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Residual model feature importance (top 20)")
print("="*70)

feat_names = feat_cols + ['dual_pred']
imp = model.feature_importances_
top_idx = np.argsort(imp)[::-1][:20]
for i in top_idx:
    print(f"  {feat_names[i]:40s}  {imp[i]:6.0f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Unseen rows — residual model coverage")
print("="*70)

# Check what resid_test predicts for unseen rows
print(f"resid_test stats (all):    mean={resid_test.mean():.3f}  std={resid_test.std():.3f}")
print(f"resid_test (seen):         mean={resid_test[seen_mask].mean():.3f}")
print(f"resid_test (unseen):       mean={resid_test[unseen_mask].mean():.3f}")

# Compare with simple inflow bins
inflow_col='order_inflow_15m'
train_inflow=train_raw[inflow_col].values
test_inflow=test_raw[inflow_col].values
bins=[0,10,20,30,40,50,60,70,80,100,120,150,200,250,300,500,1000]
bin_residuals={}
for i in range(len(bins)-1):
    lo,hi=bins[i],bins[i+1]
    mask=(train_inflow>=lo)&(train_inflow<hi)
    if mask.sum()==0: continue
    bin_residuals[(lo,hi)]=float(np.nanmean(residual[mask]))
def inflow_to_residual(arr):
    r=np.zeros(len(arr))
    for (lo,hi),mr in bin_residuals.items(): r[(arr>=lo)&(arr<hi)]=mr
    return r
test_binresid=inflow_to_residual(test_inflow)
print(f"\nBin-based residual (unseen): mean={test_binresid[unseen_mask].mean():.3f}")
print(f"LGB residual (unseen):       mean={resid_test[unseen_mask].mean():.3f}")
print(f"  LGB is {'MORE' if abs(resid_test[unseen_mask].mean()) > abs(test_binresid[unseen_mask].mean()) else 'LESS'} aggressive on unseen")

# ============================================================
print("\n" + "="*70)
print("Part 6: Combine LGB residual with triple gate")
print("="*70)

# rh triple gate base
sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)
print(f"rh_triple OOF: {trip_mae:.5f}  test={rh_trip_t.mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")

for alpha in [0.05, 0.10, 0.20, 0.30]:
    # LGB residual on rh_triple
    co = np.clip(rh_trip_o + alpha*resid_oof, 0, None)
    ct = np.clip(rh_trip_t + alpha*resid_test, 0, None)
    m = mae_fn(co)
    print(f"  triple + LGB_resid*{alpha:.2f}:  OOF={m:.5f}  test={ct.mean():.3f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# Bin calib as comparison
for alpha in [0.10, 0.20]:
    ct = np.clip(rh_trip_t + alpha*test_binresid, 0, None)
    co_bin = np.clip(rh_trip_o + alpha*inflow_to_residual(train_inflow), 0, None)
    m_bin = mae_fn(co_bin)
    print(f"  triple + BIN_resid*{alpha:.2f}:   OOF={m_bin:.5f}  test={ct.mean():.3f}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 7: Save if LGB calibration improves OOF")
print("="*70)

best_alpha_trip, best_trip_mae = 0.0, trip_mae
for alpha in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    co = np.clip(rh_trip_o + alpha*resid_oof, 0, None)
    m = mae_fn(co)
    if m < best_trip_mae:
        best_trip_mae = m
        best_alpha_trip = alpha

if best_alpha_trip > 0:
    best_ct = np.clip(rh_trip_t + best_alpha_trip*resid_test, 0, None)
    sub_tmpl=pd.read_csv('sample_submission.csv')
    sub=sub_tmpl.copy(); sub['avg_delay_minutes_next_30m']=best_ct
    fname=f"FINAL_triple_lgbResid_a{best_alpha_trip:.2f}_OOF{best_trip_mae:.5f}.csv"
    sub.to_csv(fname,index=False)
    print(f"Saved: {fname}  test={best_ct.mean():.3f}  seen={best_ct[seen_mask].mean():.3f}  unseen={best_ct[unseen_mask].mean():.3f}")
else:
    print("LGB residual did NOT improve OOF on triple gate — skip saving")

# Also test on dual base
if best_alpha > 0 and best_mae < dual_mae:
    best_ct2 = np.clip(dual_t + best_alpha*resid_test, 0, None)
    sub_tmpl=pd.read_csv('sample_submission.csv')
    sub=sub_tmpl.copy(); sub['avg_delay_minutes_next_30m']=best_ct2
    fname2=f"FINAL_dual_lgbResid_a{best_alpha:.2f}_OOF{best_mae:.5f}.csv"
    sub.to_csv(fname2,index=False)
    print(f"Saved: {fname2}  test={best_ct2.mean():.3f}  seen={best_ct2[seen_mask].mean():.3f}  unseen={best_ct2[unseen_mask].mean():.3f}")
