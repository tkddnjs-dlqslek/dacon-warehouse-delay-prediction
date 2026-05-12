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

# Load oracle_NEW
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
print(f"oracle_NEW: test={oracle_new_t.mean():.3f}  seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

# Load oracle_C_blend (another oracle variant for reference)
oracle_c_df = pd.read_csv('submission_oracle_C_blend.csv')
oracle_c_df = oracle_c_df.set_index('ID').reindex(id_order).reset_index()
oracle_c_t  = oracle_c_df['avg_delay_minutes_next_30m'].values
print(f"oracle_C:   test={oracle_c_t.mean():.3f}  seen={oracle_c_t[seen_mask].mean():.3f}  unseen={oracle_c_t[unseen_mask].mean():.3f}")

# Rebuild triple_base for reference
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
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
w_rem=1-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
bb_o=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
bb_t=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t,0,None)
bb_o=np.clip((1-w_cb)*bb_o+w_cb*cb_oof,0,None)
bb_t=np.clip((1-w_cb)*bb_t+w_cb*cb_test,0,None)
fw4_o=np.clip(0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
fw4_t=np.clip(0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t,0,None)
dual_o=fw4_o.copy()
sfw=np.sort(fw4_o); sft=np.sort(fw4_t)
dual_o[fw4_o>=sfw[-2000]]=(1-0.15)*fw4_o[fw4_o>=sfw[-2000]]+0.15*rh_o[fw4_o>=sfw[-2000]]
dual_o[fw4_o>=sfw[-5500]]=(1-0.08)*dual_o[fw4_o>=sfw[-5500]]+0.08*slhm_o[fw4_o>=sfw[-5500]]
dual_o=np.clip(dual_o,0,None)
dual_t=fw4_t.copy()
dual_t[fw4_t>=sft[-2000]]=(1-0.15)*fw4_t[fw4_t>=sft[-2000]]+0.15*rh_t[fw4_t>=sft[-2000]]
dual_t[fw4_t>=sft[-5500]]=(1-0.08)*dual_t[fw4_t>=sft[-5500]]+0.08*slhm_t[fw4_t>=sft[-5500]]
dual_t=np.clip(dual_t,0,None)
sft_d=np.sort(dual_t); sfw_d=np.sort(dual_o)
rh_trip_t=dual_t.copy()
rh_trip_t[dual_t>=sft_d[-1000]]=0.90*dual_t[dual_t>=sft_d[-1000]]+0.10*rh_t[dual_t>=sft_d[-1000]]
rh_trip_t=np.clip(rh_trip_t,0,None)
rh_trip_o=dual_o.copy()
rh_trip_o[dual_o>=sfw_d[-1000]]=0.90*dual_o[dual_o>=sfw_d[-1000]]+0.10*rh_o[dual_o>=sfw_d[-1000]]
rh_trip_o=np.clip(rh_trip_o,0,None)
trip_mae=mae_fn(rh_trip_o)
dual_mae=mae_fn(dual_o)
print(f"Triple: OOF={trip_mae:.5f}")

# ============================================================
print("\n" + "="*70)
print("Part 1: oracle_NEW + triple blend (oracle carries better seen, triple contributes)")
print("="*70)
# oracle_NEW achieves better seen (17.046) vs triple (17.640)
# Key: oracle_NEW predictions have lower seen MAE

sub_tmpl = pd.read_csv('sample_submission.csv')

# Blend oracle_NEW test predictions with triple_base test predictions
print(f"\nBase references:")
print(f"  oracle_NEW: OOF=8.38247  seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"  triple_base: OOF=8.37032  seen={rh_trip_t[seen_mask].mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")

print(f"\n{'Config':35s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
print("-"*75)

# oracle oof for blends
oracle_new_oof = None  # we don't have oracle OOF directly

# Blends using available OOF for oracle model (build from component OOFs)
# oracle_NEW OOF components: use xgb_o, lv2_o etc with OOF
# oracle_NEW is: mega+oracle blended. Let's use bb_o (mega+oracle blend) OOF
# Actually we need to reconstruct oracle_NEW's OOF structure

# From the submission file structure, oracle_NEW uses:
# mega blend + oracle models. The OOF would be 8.38247 (as stated in filename).
# We don't have its OOF array directly, but we can approximate with fw4_o (which includes oracle).

# The bb_o already has the right weights. bb_o = wf*mega + oracle
# fw4_o adds slh, xgbc, mono on top. oracle_NEW might be: 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
# Actually that's what fw4_t already is... let me check if oracle_NEW matches fw4_t

r_new_fw4, _ = __import__('scipy').stats.pearsonr(
    np.clip(oracle_new_t,0,None), np.clip(fw4_t,0,None)
)
print(f"Correlation oracle_NEW vs fw4_t: {r_new_fw4:.4f}")
print(f"fw4_t: seen={np.clip(fw4_t,0,None)[seen_mask].mean():.3f}  unseen={np.clip(fw4_t,0,None)[unseen_mask].mean():.3f}")

# oracle_NEW might be fw4_t + extra processing. Let's just use test predictions.
configs = []
for w_oracle in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
    ct = np.clip((1-w_oracle)*rh_trip_t + w_oracle*oracle_new_t, 0, None)
    configs.append((f'oracle{w_oracle:.1f}+triple{1-w_oracle:.1f}', ct))

# Print (no OOF available, just test stats)
for name, ct in configs:
    print(f"  {name:35s}  {'---':>9}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 2: oracle_NEW dual/triple gates (apply cascade to oracle_NEW)")
print("="*70)
# Apply the rh and slhm gates ON TOP of oracle_NEW test predictions

sft_on = np.sort(oracle_new_t)
oracle_gate_t = oracle_new_t.copy()

# Try dual gate on oracle_NEW
for n_top, w_rh in [(2000, 0.15), (1000, 0.10), (500, 0.20)]:
    ct = oracle_new_t.copy()
    thresh = sft_on[-n_top]
    ct[oracle_new_t>=thresh] = (1-w_rh)*oracle_new_t[oracle_new_t>=thresh] + w_rh*rh_t[oracle_new_t>=thresh]
    ct = np.clip(ct, 0, None)
    print(f"  oracle+rh_n{n_top}_w{w_rh:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# slhm gate on oracle_NEW
for n_top, w_slhm in [(5500, 0.08), (3000, 0.10)]:
    ct = oracle_new_t.copy()
    thresh = sft_on[-n_top]
    ct[oracle_new_t>=thresh] = (1-w_slhm)*oracle_new_t[oracle_new_t>=thresh] + w_slhm*slhm_t[oracle_new_t>=thresh]
    ct = np.clip(ct, 0, None)
    print(f"  oracle+slhm_n{n_top}_w{w_slhm:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# Combine dual + triple on oracle_NEW
ct_dual_on = oracle_new_t.copy()
thresh2k = sft_on[-2000]
thresh5k = sft_on[-5500]
ct_dual_on[oracle_new_t>=thresh2k] = (1-0.15)*oracle_new_t[oracle_new_t>=thresh2k] + 0.15*rh_t[oracle_new_t>=thresh2k]
ct_dual_on[oracle_new_t>=thresh5k] = (1-0.08)*ct_dual_on[oracle_new_t>=thresh5k] + 0.08*slhm_t[oracle_new_t>=thresh5k]
ct_dual_on = np.clip(ct_dual_on, 0, None)
print(f"  oracle+dual_gate: seen={ct_dual_on[seen_mask].mean():.3f}  unseen={ct_dual_on[unseen_mask].mean():.3f}")

sft_dod = np.sort(ct_dual_on)
ct_trip_on = ct_dual_on.copy()
ct_trip_on[ct_dual_on>=sft_dod[-1000]] = 0.90*ct_dual_on[ct_dual_on>=sft_dod[-1000]] + 0.10*rh_t[ct_dual_on>=sft_dod[-1000]]
ct_trip_on = np.clip(ct_trip_on, 0, None)
print(f"  oracle+triple_gate: seen={ct_trip_on[seen_mask].mean():.3f}  unseen={ct_trip_on[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: Better oracle blends")
print("="*70)
# The oracle_NEW OOF=8.38247 is good. Can we do better with different oracle weights?

# oracle_C has slightly different structure (seen=17.209, unseen=22.439 from check_oracle_unseen)
# What about blending oracle_C and oracle_NEW?
print(f"\noracle_C: seen={oracle_c_t[seen_mask].mean():.3f}  unseen={oracle_c_t[unseen_mask].mean():.3f}")
print(f"oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

# Try different oracle blends directly
from scipy.stats import pearsonr
r_nc, _ = pearsonr(oracle_new_t, oracle_c_t)
print(f"Correlation oracle_NEW vs oracle_C: {r_nc:.4f}")

# oracle_C + oracle_NEW blend
for w in [0.1, 0.2, 0.3, 0.5]:
    ct = np.clip(w*oracle_c_t + (1-w)*oracle_new_t, 0, None)
    print(f"  oC*{w:.1f}+oN*{1-w:.1f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: oracle_NEW + triple hybrid (oracle for seen, triple for unseen?)")
print("="*70)
# oracle_NEW is BETTER for seen (17.046 vs 17.640)
# Our triple is not great for unseen either
# What if oracle_NEW for seen rows + triple for unseen rows?
ct_hybrid = oracle_new_t.copy()
ct_hybrid[unseen_mask] = rh_trip_t[unseen_mask]
print(f"oracle_seen+triple_unseen: seen={ct_hybrid[seen_mask].mean():.3f}  unseen={ct_hybrid[unseen_mask].mean():.3f}")

# Other way around?
ct_hybrid2 = rh_trip_t.copy()
ct_hybrid2[seen_mask] = oracle_new_t[seen_mask]
print(f"oracle_seen+triple_unseen (v2): seen={ct_hybrid2[seen_mask].mean():.3f}  unseen={ct_hybrid2[unseen_mask].mean():.3f}")
# This is same as ct_hybrid since we replaced seen with oracle

# What about using oracle_NEW for ALL + additive correction for unseen only from triple-oracle gap?
# triple - oracle_new for unseen: this is how much extra triple adds
diff_unseen = rh_trip_t[unseen_mask] - oracle_new_t[unseen_mask]
print(f"\nTriple - oracle_NEW for unseen: mean={diff_unseen.mean():.3f}  std={diff_unseen.std():.3f}")
print(f"  (positive = triple > oracle for unseen rows)")
# If triple > oracle for unseen but oracle has better seen, then:
# we can take oracle_NEW + fraction(triple_unseen - oracle_unseen)
for frac in [0.25, 0.5, 0.75, 1.0]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + frac * diff_unseen
    ct = np.clip(ct, 0, None)
    print(f"  oracle+{frac:.2f}*(triple-oracle) for unseen: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: Improve oracle OOF-based ensemble")
print("="*70)
# Build improved oracle-style OOF blend
# The oracle models individually have OOF=8.42-8.47
# oracle_NEW blends them at OOF=8.38247
# Can we improve?

# Try cascade gates applied to oracle_NEW OOF (using oracle_new → fw4_o as proxy)
# Actually, let's try different oracle component weights for OOF optimization

oracle_models_oof = {
    'xgb': xgb_o, 'lv2': lv2_o, 'rem': rem_o, 'xgbc': xgbc_o, 'mono': mono_o
}
oracle_models_test = {
    'xgb': xgb_t, 'lv2': lv2_t, 'rem': rem_t, 'xgbc': xgbc_t, 'mono': mono_t
}

# Build oracle_NEW-like OOF (same construction as bb_o+fw4_o but matching oracle_NEW structure)
# oracle_NEW was: mega blend + oracle, with fw structure
# We'll match oracle_NEW by starting from bb_o and not applying gates
oracle_new_oof_approx = np.clip(fw4_o, 0, None)
print(f"\nfw4_o (oracle_new OOF approx) MAE: {mae_fn(oracle_new_oof_approx):.5f}")

# Apply different cascade configs to oracle_NEW OOF
sfw_on = np.sort(oracle_new_oof_approx)
for n_top, w_rh in [(2000, 0.10), (2000, 0.15), (1000, 0.10), (500, 0.15)]:
    oof_gate = oracle_new_oof_approx.copy()
    thresh = sfw_on[-n_top]
    oof_gate[oracle_new_oof_approx>=thresh] = (1-w_rh)*oracle_new_oof_approx[oracle_new_oof_approx>=thresh] + w_rh*rh_o[oracle_new_oof_approx>=thresh]
    oof_gate = np.clip(oof_gate, 0, None)
    print(f"  fw4+rh_n{n_top}_w{w_rh:.2f}: OOF={mae_fn(oof_gate):.5f}")

for n_top, w_slhm in [(5500, 0.08), (3000, 0.10), (2000, 0.12)]:
    oof_gate = oracle_new_oof_approx.copy()
    thresh = sfw_on[-n_top]
    oof_gate[oracle_new_oof_approx>=thresh] = (1-w_slhm)*oracle_new_oof_approx[oracle_new_oof_approx>=thresh] + w_slhm*slhm_o[oracle_new_oof_approx>=thresh]
    oof_gate = np.clip(oof_gate, 0, None)
    print(f"  fw4+slhm_n{n_top}_w{w_slhm:.2f}: OOF={mae_fn(oof_gate):.5f}")

# ============================================================
print("\n" + "="*70)
print("Part 6: Save best candidates")
print("="*70)

to_save = [
    # Blends of oracle_NEW and triple
    ('oracleN70_triple30', np.clip(0.7*oracle_new_t + 0.3*rh_trip_t, 0, None)),
    ('oracleN50_triple50', np.clip(0.5*oracle_new_t + 0.5*rh_trip_t, 0, None)),
    ('oracleN80_triple20', np.clip(0.8*oracle_new_t + 0.2*rh_trip_t, 0, None)),
    ('oracle_dual_gate',   ct_dual_on),
    ('oracle_triple_gate', ct_trip_on),
    # Oracle seen + triple unseen
    ('oracleSeen_tripleUnseen', ct_hybrid),
]

print(f"\n{'Filename':65s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for label, ct in to_save:
    fname = f"FINAL_NEW_{label}_OOF_oracle.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  SAVED  {fname[:63]:63s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

print("\nDone.")
