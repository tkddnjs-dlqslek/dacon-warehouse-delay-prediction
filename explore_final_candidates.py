import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
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

mega33_oof  = d33['meta_avg_oof'][id2]; mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]; mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy');       xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy');    lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t  = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t   = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2];   slhm_t = np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]
savg_o = np.load('results/cascade/spec_avg_oof.npy')[id2];            savg_t = np.load('results/cascade/spec_avg_test.npy')[te_id2]
spec_cb_w30_o = np.load('results/cascade/spec_cb_w30_oof.npy')[id2]
spec_cb_w30_t = np.load('results/cascade/spec_cb_w30_test.npy')[te_id2]

mae_fn = lambda p: float(np.mean(np.abs(np.clip(p, 0, None) - y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx  = wm*mega   + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

bb_o, bb_t = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)
fw4_o = np.clip(0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
fw4_t = np.clip(0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)

n1, w1, n2, w2 = 2000, 0.15, 5500, 0.08
sfw = np.sort(fw4_o); sft = np.sort(fw4_t)
m1_o = fw4_o >= sfw[-n1]; m2_o = fw4_o >= sfw[-n2]
dual_o = fw4_o.copy()
dual_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
dual_o[m2_o] = (1-w2)*dual_o[m2_o] + w2*slhm_o[m2_o]
dual_mae = mae_fn(dual_o)

m1_t = fw4_t >= sft[-n1]; m2_t = fw4_t >= sft[-n2]
dual_t = fw4_t.copy()
dual_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
dual_t[m2_t] = (1-w2)*dual_t[m2_t] + w2*slhm_t[m2_t]
dual_t = np.clip(dual_t, 0, None)

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true), dtype=int)
for fi, (_, vi) in enumerate(gkf.split(train_raw, y_true, groups)):
    fold_ids[vi] = fi
fold_mae_fn = lambda p: [float(np.mean(np.abs(p[fold_ids==fi] - y_true[fold_ids==fi]))) for fi in range(5)]

print(f"dual_gate: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

# Triple gate (rh n0=1000, w0=0.10)
sfw_d = np.sort(dual_o); sft_d = np.sort(dual_t)
m0_o = dual_o >= sfw_d[-1000]; m0_t = dual_t >= sft_d[-1000]
rh_trip_o = dual_o.copy()
rh_trip_o[m0_o] = 0.90*dual_o[m0_o] + 0.10*rh_o[m0_o]
rh_trip_o = np.clip(rh_trip_o, 0, None)
rh_trip_t = dual_t.copy()
rh_trip_t[m0_t] = 0.90*dual_t[m0_t] + 0.10*rh_t[m0_t]
rh_trip_t = np.clip(rh_trip_t, 0, None)
trip_mae = mae_fn(rh_trip_o)
print(f"rh_triple: OOF={trip_mae:.5f}  test={rh_trip_t.mean():.3f}  seen={rh_trip_t[seen_mask].mean():.3f}  unseen={rh_trip_t[unseen_mask].mean():.3f}")
print(f"  folds: {fold_mae_fn(rh_trip_o)}")

# Seen gate mask for asymmetric boost
m1_s_t = (fw4_t >= sft[-n1]) & seen_mask
m2_s_t = (fw4_t >= sft[-n2]) & seen_mask

def apply_unseen_boost(base_o, base_t, trip_oof, n1u, w1u, n2u, w2u):
    """Apply asymmetric boost for unseen test rows only, on top of a base ensemble."""
    m1_u = (fw4_t >= sft[-n1u]) & unseen_mask
    m2_u = (fw4_t >= sft[-n2u]) & unseen_mask
    m1f = m1_s_t | m1_u; m2f = m2_s_t | m2_u
    w1a = np.where(unseen_mask, w1u, w1)
    w2a = np.where(unseen_mask, w2u, w2)
    t = base_t.copy()
    t[m1f] = (1-w1a[m1f])*fw4_t[m1f] + w1a[m1f]*rh_t[m1f]
    t[m2f] = (1-w2a[m2f])*t[m2f]     + w2a[m2f]*slhm_t[m2f]
    return np.clip(t, 0, None), trip_oof

print("\n" + "="*70)
print("Part 1: Combined rh_triple + unseen boost")
print("="*70)

configs = [
    ("A_w1u25_w2u12",    2000, 0.25, 5500, 0.12),
    ("A_w1u30_w2u15",    2000, 0.30, 5500, 0.15),
    ("B_n5k_n8k",        5000, 0.15, 8000, 0.08),
    ("AB_n5k_w25_n8k_w12", 5000, 0.25, 8000, 0.12),
    ("AB_n7k_w20_n11k_w10", 7000, 0.20, 11000, 0.10),
    ("AB_n10k_w20_n15k_w10", 10000, 0.20, 15000, 0.10),
]
print(f"{'config':30s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
all_trip_unseen = []
for name, n1u, w1u, n2u, w2u in configs:
    t, oof = apply_unseen_boost(dual_o, rh_trip_t, trip_mae, n1u, w1u, n2u, w2u)
    print(f"  {name:28s}  {oof:>9.5f}  {t.mean():>8.3f}  {t[seen_mask].mean():>8.3f}  {t[unseen_mask].mean():>8.3f}")
    all_trip_unseen.append((name, oof, t, n1u, w1u, n2u, w2u))

print("\n" + "="*70)
print("Part 2: dual + spec_avg unseen boost (corr=0.9426)")
print("= Apply spec_avg specifically to unseen test rows on top of dual_gate")
print("="*70)

print(f"spec_avg: OOF={mae_fn(savg_o):.5f}  unseen={savg_t[unseen_mask].mean():.3f}")
for w_sa in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    t = dual_t.copy()
    t[unseen_mask] = np.clip((1-w_sa)*dual_t[unseen_mask] + w_sa*savg_t[unseen_mask], 0, None)
    print(f"  spec_avg unseen w={w_sa:.2f}: test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}  OOF={dual_mae:.5f}")

print("\n" + "="*70)
print("Part 3: Direct unseen boost via rh only (no slhm for unseen)")
print("(rh alone for unseen top rows)")
print("="*70)

for n1u in [5000, 10000, 20000]:
    for w1u in [0.20, 0.25, 0.30]:
        m1_u = (fw4_t >= sft[-n1u]) & unseen_mask
        t = dual_t.copy()
        t[m1_u] = np.clip((1-w1u)*dual_t[m1_u] + w1u*rh_t[m1_u], 0, None)
        print(f"  rh_only n1u={n1u:6d} w1u={w1u:.2f}: test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}")

print("\n" + "="*70)
print("Part 4: Save final submission candidates")
print("="*70)

sub_template = pd.read_csv('sample_submission.csv')
saved = []

# 1. dual gate
sub = sub_template.copy(); sub['avg_delay_minutes_next_30m'] = dual_t
sub.to_csv('FINAL_dual_gate_OOF8.37050.csv', index=False)
saved.append(('FINAL_dual_gate_OOF8.37050.csv', dual_mae, dual_t.mean(), dual_t[unseen_mask].mean()))

# 2. rh triple gate
sub = sub_template.copy(); sub['avg_delay_minutes_next_30m'] = rh_trip_t
sub.to_csv('FINAL_rh_triple_OOF8.37032.csv', index=False)
saved.append(('FINAL_rh_triple_OOF8.37032.csv', trip_mae, rh_trip_t.mean(), rh_trip_t[unseen_mask].mean()))

# 3. rh_triple + moderate unseen boost (A_w1u25_w2u12)
name3, oof3, t3, n1u3, w1u3, n2u3, w2u3 = all_trip_unseen[0]
sub = sub_template.copy(); sub['avg_delay_minutes_next_30m'] = t3
sub.to_csv(f'FINAL_rh_triple_unsBst_{name3}_OOF{oof3:.5f}.csv', index=False)
saved.append((f'FINAL_rh_triple_unsBst_{name3}_OOF{oof3:.5f}.csv', oof3, t3.mean(), t3[unseen_mask].mean()))

# 4. rh_triple + aggressive unseen boost (AB_n10k_w20_n15k_w10)
name4, oof4, t4, n1u4, w1u4, n2u4, w2u4 = all_trip_unseen[5]
sub = sub_template.copy(); sub['avg_delay_minutes_next_30m'] = t4
sub.to_csv(f'FINAL_rh_triple_unsBst_{name4}_OOF{oof4:.5f}.csv', index=False)
saved.append((f'FINAL_rh_triple_unsBst_{name4}_OOF{oof4:.5f}.csv', oof4, t4.mean(), t4[unseen_mask].mean()))

# 5. dual + spec_avg unseen blend (w=0.15)
t5 = dual_t.copy()
t5[unseen_mask] = np.clip(0.85*dual_t[unseen_mask] + 0.15*savg_t[unseen_mask], 0, None)
sub = sub_template.copy(); sub['avg_delay_minutes_next_30m'] = t5
sub.to_csv(f'FINAL_dual_savg_unseen_w15_OOF{dual_mae:.5f}.csv', index=False)
saved.append((f'FINAL_dual_savg_unseen_w15_OOF{dual_mae:.5f}.csv', dual_mae, t5.mean(), t5[unseen_mask].mean()))

print(f"\n{'File':55s}  {'OOF':>9}  {'test':>8}  {'unseen':>8}")
for fname, oof, tm, uns in saved:
    print(f"  {fname:53s}  {oof:>9.5f}  {tm:>8.3f}  {uns:>8.3f}")

print("\n" + "="*70)
print("SUBMISSION STRATEGY RECOMMENDATION")
print("="*70)
print("""
Priority order (5 submissions/day):
1. FINAL_dual_gate_OOF8.37050         -- anchor, most validated (OOF=8.37050)
2. FINAL_rh_triple_OOF8.37032         -- tiny OOF gain (2/5 folds), higher unseen
3. FINAL_rh_triple_unsBst_A_w1u25..   -- conservative unseen boost (unseen~23.9)
4. FINAL_rh_triple_unsBst_AB_n10k..   -- moderate unseen boost (unseen~25.4)
5. FINAL_dual_savg_unseen_w15         -- spec_avg unseen blend (conservative)
""")
