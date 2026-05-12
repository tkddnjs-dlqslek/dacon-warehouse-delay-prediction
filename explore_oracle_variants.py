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
cb_oof=np.clip(d33['meta_oofs']['cb'][id2],0,None); cb_test=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]; rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

sub_tmpl = pd.read_csv('sample_submission.csv')

# ============================================================
print("="*70)
print("Part 1: Oracle model OOF performance review")
print("="*70)

print(f"\n{'Model':12s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
oracle_models = [
    ('xgb',   xgb_o,  xgb_t),
    ('lv2',   lv2_o,  lv2_t),
    ('rem',   rem_o,  rem_t),
    ('xgbc',  xgbc_o, xgbc_t),
    ('mono',  mono_o, mono_t),
]
for name, oo, ot in oracle_models:
    print(f"  {name:12s}  {mae_fn(np.clip(oo,0,None)):9.5f}  {np.clip(ot,0,None).mean():8.3f}  "
          f"{np.clip(ot,0,None)[seen_mask].mean():8.3f}  {np.clip(ot,0,None)[unseen_mask].mean():8.3f}")

print(f"\n  oracle_NEW     8.38247  19.314    17.046    22.716  [BEST LB=9.7527]")

# ============================================================
print("\n" + "="*70)
print("Part 2: Oracle variants with more 'rem' weight")
print("="*70)
# rem has seen=16.606, unseen=22.051 — both lowest
# rem+xgb: combine low-seen with low-unseen

# Build base oracle blend varying rem weight
print(f"\n{'Config':35s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")
print("-"*65)

candidate_blends = []
# Rem-focused blends (without mega)
for w_rem, w_xgb, w_lv2, w_xgbc, w_mono in [
    (0.5, 0.25, 0.25, 0.0, 0.0),
    (0.4, 0.3, 0.3, 0.0, 0.0),
    (0.3, 0.3, 0.3, 0.1, 0.0),
    (0.25, 0.25, 0.25, 0.25, 0.0),
    (0.2, 0.2, 0.2, 0.2, 0.2),  # equal
    (0.0, 0.5, 0.5, 0.0, 0.0),  # xgb+lv2
    (0.0, 1.0, 0.0, 0.0, 0.0),  # xgb only
    (0.0, 0.0, 1.0, 0.0, 0.0),  # lv2 only
    (1.0, 0.0, 0.0, 0.0, 0.0),  # rem only
    (0.0, 0.0, 0.0, 1.0, 0.0),  # xgbc only
]:
    oo = w_rem*rem_o + w_xgb*xgb_o + w_lv2*lv2_o + w_xgbc*xgbc_o + w_mono*mono_o
    ot = w_rem*rem_t + w_xgb*xgb_t + w_lv2*lv2_t + w_xgbc*xgbc_t + w_mono*mono_t
    oo_c = np.clip(oo, 0, None); ot_c = np.clip(ot, 0, None)
    name = f"rem{w_rem:.1f}_x{w_xgb:.1f}_l{w_lv2:.1f}_xb{w_xgbc:.1f}_mn{w_mono:.1f}"
    print(f"  {name:35s}  {mae_fn(oo_c):9.5f}  {ot_c[seen_mask].mean():8.3f}  {ot_c[unseen_mask].mean():8.3f}")
    candidate_blends.append((name, oo_c, ot_c))

# ============================================================
print("\n" + "="*70)
print("Part 3: mega+oracle blends with varying mega/oracle ratio")
print("="*70)
print(f"(oracle_NEW: wf=0.72 mega + 0.28 oracle)")
print()
print(f"{'Config':35s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")
print("-"*65)

# Pure oracle (equal weights of all 5)
oracle_t_eq = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)
oracle_o_eq = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)
print(f"  oracle_5way:                        {mae_fn(oracle_o_eq):9.5f}  {oracle_t_eq[seen_mask].mean():8.3f}  {oracle_t_eq[unseen_mask].mean():8.3f}")

# Varying mega/oracle ratio
for w_mega in [0.5, 0.6, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85, 0.90]:
    w_oracle = 1.0 - w_mega
    # Keep oracle model proportions the same as current
    wxgb = 0.12*w_oracle/0.36; wlv2 = 0.16*w_oracle/0.36; wrem2 = 0.08*w_oracle/0.36
    bb_oo = np.clip(w_mega*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
    bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof, 0, None)
    fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)

    bb_tt = np.clip(w_mega*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem2*rem_t, 0, None)
    bb_tt = np.clip((1-w_cb)*bb_tt + w_cb*cb_test, 0, None)
    fw4_tt = np.clip(0.74*bb_tt + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)

    print(f"  mega{w_mega:.2f}+oracle{w_oracle:.2f} (nogate):     {mae_fn(fw4_oo):9.5f}  {fw4_tt[seen_mask].mean():8.3f}  {fw4_tt[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: mega+oracle with rem upweighted")
print("="*70)
# Instead of xgb:lv2:rem = 12:16:8, try rem-upweighted versions
print()
print(f"{'Config':40s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")
print("-"*65)

for wxgb2, wlv2_2, wrem2, label in [
    (0.0933, 0.1244, 0.0622, 'current'),         # 12:16:8
    (0.08,   0.12,   0.08,   'xgb0.08_lv2_0.12_rem0.08'),  # equal raise rem
    (0.06,   0.10,   0.12,   'xgb0.06_lv2_0.10_rem0.12'),  # rem dominant
    (0.05,   0.08,   0.15,   'xgb0.05_lv2_0.08_rem0.15'),  # rem very high
    (0.10,   0.10,   0.10,   'all_equal_010'),
    (0.12,   0.12,   0.04,   'reduce_rem'),
]:
    # Normalize to keep total oracle contribution = 0.28
    total = wxgb2 + wlv2_2 + wrem2
    wxgb2_n = wxgb2/total*0.28; wlv2_n = wlv2_2/total*0.28; wrem_n = wrem2/total*0.28

    bb_oo = np.clip(0.72*fx_o + wxgb2_n*xgb_o + wlv2_n*lv2_o + wrem_n*rem_o, 0, None)
    bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof, 0, None)
    fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)

    bb_tt = np.clip(0.72*fxt + wxgb2_n*xgb_t + wlv2_n*lv2_t + wrem_n*rem_t, 0, None)
    bb_tt = np.clip((1-w_cb)*bb_tt + w_cb*cb_test, 0, None)
    fw4_tt = np.clip(0.74*bb_tt + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)

    print(f"  {label:40s}  {mae_fn(fw4_oo):9.5f}  {fw4_tt[seen_mask].mean():8.3f}  {fw4_tt[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Full grid - find blend between oracle_NEW and pure oracle")
print("="*70)
# oracle_NEW (fw4_t) vs oracle_5way: what blend minimizes unseen predictions near 22.3-22.5?
print()
print(f"oracle_NEW: seen=17.046, unseen=22.716")
print(f"oracle_5way: seen={oracle_t_eq[seen_mask].mean():.3f}, unseen={oracle_t_eq[unseen_mask].mean():.3f}")
print()
print(f"{'w_oracle5way':12s}  {'seen':>8}  {'unseen':>8}")
for w5 in np.arange(0, 1.1, 0.1):
    ct = np.clip((1-w5)*oracle_new_t + w5*oracle_t_eq, 0, None)
    print(f"  {w5:.2f}          {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 6: Save promising new oracle variants")
print("="*70)

# Key submissions to try based on analysis
to_save_configs = []

# oracle_5way alone
to_save_configs.append(('oracle5way', oracle_t_eq))

# oracle_C blend (from file, if available, compare to oracle 5way)
try:
    oracle_c_df = pd.read_csv('submission_oracle_C_blend.csv')
    oracle_c_df = oracle_c_df.set_index('ID').reindex(id_order).reset_index()
    oracle_c_t  = oracle_c_df['avg_delay_minutes_next_30m'].values
    to_save_configs.append(('oracle_C_as_base', oracle_c_t))
except: pass

# Rem-only
to_save_configs.append(('rem_only',  np.clip(rem_t, 0, None)))

# Oracle_5way + oracle_NEW blend
for w5 in [0.1, 0.2, 0.3, 0.5]:
    ct = np.clip((1-w5)*oracle_new_t + w5*oracle_t_eq, 0, None)
    to_save_configs.append((f'oN{1-w5:.1f}_o5_{w5:.1f}', ct))

# rem + xgb + lv2 (equal) blend
rem_xgb_lv2 = np.clip((rem_t + xgb_t + lv2_t)/3, 0, None)
to_save_configs.append(('rem_xgb_lv2_eq', rem_xgb_lv2))

# oracle_NEW + 10% oracle_5way (conservative)
ct_oracle_lite = np.clip(0.9*oracle_new_t + 0.1*oracle_t_eq, 0, None)
to_save_configs.append(('oracleN90_5w10', ct_oracle_lite))

# mega50+oracle50 (more oracle, less mega) — from varying ratio above
# Recompute
for w_mega2 in [0.60, 0.65, 0.70]:
    w_oracle2 = 1.0 - w_mega2
    wxgb2 = 0.12*w_oracle2/0.36; wlv2_2 = 0.16*w_oracle2/0.36; wrem_2 = 0.08*w_oracle2/0.36
    bb_tt2 = np.clip(w_mega2*fxt + wxgb2*xgb_t + wlv2_2*lv2_t + wrem_2*rem_t, 0, None)
    bb_tt2 = np.clip((1-w_cb)*bb_tt2 + w_cb*cb_test, 0, None)
    fw4_tt2 = np.clip(0.74*bb_tt2 + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)
    to_save_configs.append((f'mega{w_mega2:.2f}_oracle{w_oracle2:.2f}', fw4_tt2))

print(f"\n{'Filename':65s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")

# Build OOF for oracle blends
oracle_oof_map = {
    'oracle5way': oracle_o_eq,
    'rem_only': np.clip(rem_o, 0, None),
    'rem_xgb_lv2_eq': np.clip((rem_o+xgb_o+lv2_o)/3, 0, None),
}

for label, ct in to_save_configs:
    oof_approx = oracle_oof_map.get(label.split('_')[0], None)
    oof_str = f'{mae_fn(oof_approx):9.5f}' if oof_approx is not None else "    ---  "
    fname = f"FINAL_NEW_{label}_OOF_oracle.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  SAVED  {fname[:63]:63s}  {oof_str}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

print("\nDone.")
