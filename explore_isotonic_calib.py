import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from scipy.stats import pearsonr
from sklearn.isotonic import IsotonicRegression

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

# Reconstruct oracle_NEW OOF and test
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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

print("="*70)
print("Isotonic regression calibration of oracle_NEW OOF → test")
print("="*70)

# Fit isotonic regression on OOF predictions
p_oof = fw4_oo  # oracle_NEW OOF
y     = y_true

# Sort by prediction for isotonic fit
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(p_oof, y)

# Evaluate on training (will be perfectly calibrated in-sample)
y_iso_train = iso.predict(p_oof)
print(f"Isotonic on train OOF: MAE={mae_fn(y_iso_train):.5f}  (in-sample, optimistic)")
print(f"  train: mean_pred={y_iso_train.mean():.3f}  mean_y={y.mean():.3f}")

# Apply to test
p_test = fw4_tt  # use fw4_tt which is the oracle_NEW base test prediction
y_iso_test = iso.predict(p_test)
y_iso_test = np.clip(y_iso_test, 0, None)
print(f"\nIsotonic applied to test (fw4_tt):")
print(f"  seen={y_iso_test[seen_mask].mean():.3f}  unseen={y_iso_test[unseen_mask].mean():.3f}")
print(f"  r(oracle_NEW, iso_test) = {pearsonr(y_iso_test, oracle_new_t)[0]:.4f}")

# Apply to oracle_new_t directly
y_iso_oracle = iso.predict(oracle_new_t)
y_iso_oracle = np.clip(y_iso_oracle, 0, None)
print(f"\nIsotonic applied to oracle_NEW test:")
print(f"  seen={y_iso_oracle[seen_mask].mean():.3f}  unseen={y_iso_oracle[unseen_mask].mean():.3f}")

print()
print("="*70)
print("Partial isotonic (blend with original: frac * iso + (1-frac) * original)")
print("="*70)
for frac in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
    blended = frac * y_iso_oracle + (1-frac) * oracle_new_t
    blended = np.clip(blended, 0, None)
    print(f"  frac={frac:.2f}: seen={blended[seen_mask].mean():.3f}  unseen={blended[unseen_mask].mean():.3f}  r(oN)={pearsonr(blended, oracle_new_t)[0]:.4f}")

print()
print("="*70)
print("Isotonic calibration function: prediction → calibrated_prediction")
print("="*70)
# Show the isotonic function at key points
test_points = [3, 5, 7, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70]
for tp in test_points:
    cal = float(iso.predict([tp])[0])
    print(f"  f({tp:3d}) = {cal:.3f}  (correction = {cal-tp:+.3f})")

print()
print("="*70)
print("Save isotonic candidates")
print("="*70)
sub_tmpl = pd.read_csv('sample_submission.csv')

for frac in [0.10, 0.20, 0.30]:
    blended = np.clip(frac * y_iso_oracle + (1-frac) * oracle_new_t, 0, None)
    fname = f"FINAL_NEW_oN_iso{int(frac*100)}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = blended
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  seen={blended[seen_mask].mean():.3f}  unseen={blended[unseen_mask].mean():.3f}")

# Full isotonic on oracle_NEW test
fname = "FINAL_NEW_oN_isoFull_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = y_iso_oracle
sub.to_csv(fname, index=False)
print(f"Saved: {fname}  seen={y_iso_oracle[seen_mask].mean():.3f}  unseen={y_iso_oracle[unseen_mask].mean():.3f}")

# Cross-layout isotonic check: train only on high-inflow layouts
print()
print("="*70)
print("Isotonic by layout group: seen vs unseen train calibration")
print("="*70)

# For training layouts: compute per-layout means and fit isotonic
train_layouts_list = sorted(train_raw['layout_id'].unique())
lay_pred = []
lay_true = []
for lay in train_layouts_list:
    mask = (train_raw['layout_id'] == lay).values
    lay_pred.append(fw4_oo[mask].mean())
    lay_true.append(y_true[mask].mean())

lay_pred = np.array(lay_pred)
lay_true = np.array(lay_true)

# Fit isotonic at layout level
iso_lay = IsotonicRegression(out_of_bounds='clip')
iso_lay.fit(lay_pred, lay_true)

# Check on training layouts
lay_cal = iso_lay.predict(lay_pred)
mae_lay = np.mean(np.abs(lay_cal - lay_true))
print(f"Layout-level isotonic: MAE={mae_lay:.4f} (in-sample)")

# Show calibration function
print(f"\nLayout isotonic function:")
for tp in [8, 10, 12, 15, 18, 20, 22, 25, 28, 30]:
    cal = float(iso_lay.predict([tp])[0])
    print(f"  f({tp:3d}) = {cal:.3f}  (correction = {cal-tp:+.3f})")

# Apply to test unseen layouts
unseen_layouts_test = sorted(test_raw[unseen_mask]['layout_id'].unique())
ct_lay_iso = oracle_new_t.copy()
for lay in unseen_layouts_test:
    mask = (test_raw['layout_id'] == lay).values & unseen_mask
    p_lay_mean = oracle_new_t[mask].mean()
    cal_mean = float(iso_lay.predict([p_lay_mean])[0])
    correction = cal_mean - p_lay_mean
    ct_lay_iso[mask] = oracle_new_t[mask] + correction
ct_lay_iso = np.clip(ct_lay_iso, 0, None)
print(f"\nLayout-isotonic correction applied: seen={ct_lay_iso[seen_mask].mean():.3f}  unseen={ct_lay_iso[unseen_mask].mean():.3f}")

# Partial application
for frac in [0.1, 0.2, 0.3, 0.5]:
    ct = np.clip(frac * ct_lay_iso + (1-frac) * oracle_new_t, 0, None)
    print(f"  lay_iso_frac={frac:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")
    if frac == 0.2:
        fname = "FINAL_NEW_oN_layIso20_OOF8.3825.csv"
        sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
        sub.to_csv(fname, index=False)
        print(f"    Saved: {fname}")

# Full layout isotonic
fname = "FINAL_NEW_oN_layIsoFull_OOF8.3825.csv"
sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct_lay_iso
sub.to_csv(fname, index=False)
print(f"\nSaved: {fname}  seen={ct_lay_iso[seen_mask].mean():.3f}  unseen={ct_lay_iso[unseen_mask].mean():.3f}")

print("\nDone.")
