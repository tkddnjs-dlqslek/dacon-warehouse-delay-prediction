import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

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

xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
cb_test_mega=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]; rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]
spec_cb_w30_o=np.load('results/cascade/spec_cb_w30_oof.npy')[id2]
spec_cb_w30_t=np.load('results/cascade/spec_cb_w30_test.npy')[te_id2]
spec_cb_raw_o=np.load('results/cascade/spec_cb_raw_oof.npy')[id2]
spec_cb_raw_t=np.load('results/cascade/spec_cb_raw_test.npy')[te_id2]

# CB standalone (from results root)
cb_oof_s = np.load('results/cb_oof.npy')
cb_test_s = np.load('results/cb_test.npy')
print(f"cb_standalone shape: oof={cb_oof_s.shape}  test={cb_test_s.shape}")
if len(cb_oof_s) == len(y_true): cb_oof_sc = np.clip(cb_oof_s, 0, None)
elif len(cb_oof_s) == len(train_ls): cb_oof_sc = np.clip(cb_oof_s[id2], 0, None)
else: cb_oof_sc = None
if len(cb_test_s) == len(test_raw): cb_test_sc = np.clip(cb_test_s, 0, None)
elif len(cb_test_s) == len(test_ls): cb_test_sc = np.clip(cb_test_s[te_id2], 0, None)
else: cb_test_sc = None

if cb_oof_sc is not None: print(f"cb_standalone: OOF={mae_fn(cb_oof_sc):.5f}")
if cb_test_sc is not None: print(f"  test={cb_test_sc.mean():.3f}  seen={cb_test_sc[seen_mask].mean():.3f}  unseen={cb_test_sc[unseen_mask].mean():.3f}")

# XGB standalone
xgb_oof_s = np.load('results/xgb_oof.npy')
xgb_test_s = np.load('results/xgb_test.npy')
print(f"xgb_standalone shape: oof={xgb_oof_s.shape}  test={xgb_test_s.shape}")
if len(xgb_oof_s) == len(y_true): xgb_oof_sc = np.clip(xgb_oof_s, 0, None)
elif len(xgb_oof_s) == len(train_ls): xgb_oof_sc = np.clip(xgb_oof_s[id2], 0, None)
else: xgb_oof_sc = None
if len(xgb_test_s) == len(test_raw): xgb_test_sc = np.clip(xgb_test_s, 0, None)
elif len(xgb_test_s) == len(test_ls): xgb_test_sc = np.clip(xgb_test_s[te_id2], 0, None)
else: xgb_test_sc = None
if xgb_oof_sc is not None: print(f"xgb_standalone: OOF={mae_fn(xgb_oof_sc):.5f}")
if xgb_test_sc is not None: print(f"  test={xgb_test_sc.mean():.3f}  seen={xgb_test_sc[seen_mask].mean():.3f}  unseen={xgb_test_sc[unseen_mask].mean():.3f}")

# oracle_NEW reference
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_t_eq = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)
oracle_o_eq = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2_*r2_test+w3_*r3_test
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
bb_tt=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem2*rem_t,0,None)
bb_tt=np.clip((1-w_cb)*bb_tt+w_cb*cb_test_mega,0,None)
fw4_tt=np.clip(0.74*bb_tt+0.08*slh_t+0.10*xgbc_t+0.08*mono_t,0,None)
print(f"\nfw4_t (oracle_NEW base): OOF={mae_fn(fw4_oo):.5f}  seen={fw4_tt[seen_mask].mean():.3f}  unseen={fw4_tt[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Cascade CatBoost models")
print("="*70)
spec_cb_w30_c = np.clip(spec_cb_w30_t, 0, None)
spec_cb_raw_c = np.clip(spec_cb_raw_t, 0, None)
print(f"spec_cb_w30: OOF={mae_fn(np.clip(spec_cb_w30_o,0,None)):.5f}  seen={spec_cb_w30_c[seen_mask].mean():.3f}  unseen={spec_cb_w30_c[unseen_mask].mean():.3f}")
print(f"spec_cb_raw: OOF={mae_fn(np.clip(spec_cb_raw_o,0,None)):.5f}  seen={spec_cb_raw_c[seen_mask].mean():.3f}  unseen={spec_cb_raw_c[unseen_mask].mean():.3f}")
print(f"r(spec_cb_w30, oracle_NEW) = {__import__('scipy').stats.pearsonr(spec_cb_w30_c, oracle_new_t)[0]:.4f}")

# Can spec_cb_w30 replace slh/xgbc/mono in fw4_t?
# fw4_tt currently: 0.74*bb_tt + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
# Try: 0.74*bb_tt + 0.08*spec_cb_w30_t + 0.10*xgbc_t + 0.08*mono_t
fw4_spec_cb = np.clip(0.74*bb_tt + 0.08*spec_cb_w30_c + 0.10*xgbc_t + 0.08*mono_t, 0, None)
fw4_spec_cb_oof = np.clip(0.74*bb_oo + 0.08*np.clip(spec_cb_w30_o,0,None) + 0.10*xgbc_o + 0.08*mono_o, 0, None)
print(f"\nfw4 with spec_cb_w30 replacing slh: OOF={mae_fn(fw4_spec_cb_oof):.5f}  seen={fw4_spec_cb[seen_mask].mean():.3f}  unseen={fw4_spec_cb[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("FINAL COMPREHENSIVE SUMMARY — All submission candidates")
print("="*70)

# Load all FINAL_NEW and key submission CSVs
import glob
final_files = sorted(glob.glob('FINAL_NEW_*.csv')) + sorted(glob.glob('submission_oracle_NEW*.csv'))
final_files = [f for f in final_files if 'OOF_oracle' in f or 'OOF8.37032' in f or 'OOF8.38' in f]

all_candidates = {}
for fname in final_files:
    try:
        df = pd.read_csv(fname).set_index('ID').reindex(id_order).reset_index()
        p = df['avg_delay_minutes_next_30m'].values
        all_candidates[fname] = p
    except: pass

# Add oracle_NEW explicitly
all_candidates['[oracle_NEW]'] = oracle_new_t
# Add oracle_5way
all_candidates['[oracle_5way]'] = oracle_t_eq
# Add fw4_tt
all_candidates['[fw4_nogate]'] = fw4_tt
# Add spec_cb_w30 variant
all_candidates['[fw4_spec_cbw30]'] = fw4_spec_cb

print(f"\n{'Filename':65s}  {'seen':>8}  {'unseen':>8}  {'test':>8}")
print("-"*100)
# Sort by seen predictions
items = sorted(all_candidates.items(), key=lambda x: x[1][seen_mask].mean())
for name, p in items:
    print(f"  {name[:63]:63s}  {p[seen_mask].mean():8.3f}  {p[unseen_mask].mean():8.3f}  {p.mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("PRIORITY SUBMISSION LIST (recommended by analysis)")
print("="*70)
print("""
Analysis summary:
  - oracle_NEW (seen=17.046, unseen=22.716, OOF=8.38247) = BEST LB=9.7527
  - OOF inversion: better OOF (8.37) → worse LB; oracle_NEW is at optimum
  - Unseen mean boost (asym_u150 etc): KK variants at 22.8+ didn't improve
  - pure oracle_5way (OOF=8.41): potentially better LB via inversion? Untested.
  - rem_only (OOF=8.47): most aggressively detuned, highest risk

PRIORITY ORDER FOR SUBMISSION:
  1. [oracle_5way] seen=16.789, unseen=22.358 — pure oracle blend, OOF=8.41
     (slightly worse OOF than oracle_NEW → might improve LB)
  2. [oN70o5w30] seen=16.969, unseen=22.609 — moderate oracle_5way mix
  3. [oracle5w90_rank10] seen=16.832, unseen=22.420 — oracle_5way + 10% rank
  4. [rem_xgb_lv2_3way] seen=16.773, unseen=22.303 — rem-heavy oracle mix
  5. [rem_only] seen=16.606, unseen=22.051 — most detuned, highest risk/reward

  NOTE: The unseen calibration files (asym_u150, fixedLL_u100 etc.) are
  unlikely to help based on KK variant evidence. Skip unless oracle_5way
  variants don't improve LB.
""")

sub_tmpl = pd.read_csv('sample_submission.csv')

# Final save of any not yet saved
for label, ct in [
    ('spec_cbw30_variant', fw4_spec_cb),
]:
    fname = f"FINAL_NEW_{label}_OOF_special.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

print("\nDone.")
