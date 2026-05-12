import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
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

# dual gate baseline
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
print(f"dual_gate: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

m1_s = (fw4_t >= sft[-n1]) & seen_mask
m2_s = (fw4_t >= sft[-n2]) & seen_mask

def make_asym(n1u, w1u, n2u, w2u):
    """Keep seen gate as-is. For unseen: extend gate + boost weight."""
    m1_u = (fw4_t >= sft[-n1u]) & unseen_mask if n1u < len(fw4_t) else unseen_mask
    m2_u = (fw4_t >= sft[-n2u]) & unseen_mask if n2u < len(fw4_t) else unseen_mask
    m1f = m1_s | m1_u; m2f = m2_s | m2_u
    w1a = np.where(unseen_mask, w1u, w1)
    w2a = np.where(unseen_mask, w2u, w2)
    t = fw4_t.copy()
    t[m1f] = (1-w1a[m1f])*fw4_t[m1f] + w1a[m1f]*rh_t[m1f]
    t[m2f] = (1-w2a[m2f])*t[m2f]     + w2a[m2f]*slhm_t[m2f]
    return np.clip(t, 0, None)

print("\n" + "="*70)
print("Part 1: All cascade files -- unseen performance analysis")
print("="*70)
cascade_dir = 'results/cascade/'
cascade_files = sorted(glob.glob(os.path.join(cascade_dir, '*_oof.npy')))
print(f"{'filename':40s}  {'solo_OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}  {'u/s':>6}")
cmod = {}
for fp in cascade_files:
    fn = os.path.basename(fp)
    test_fn = fn.replace('_oof.npy', '_test.npy')
    test_path = os.path.join(cascade_dir, test_fn)
    if not os.path.exists(test_path): continue
    oof_a = np.load(fp)[id2]
    test_a = np.load(test_path)[te_id2]
    if len(oof_a) != len(y_true) or len(test_a) != len(test_raw): continue
    solo_oof = mae_fn(oof_a)
    t_seen = test_a[seen_mask].mean()
    t_unseen = test_a[unseen_mask].mean()
    print(f"  {fn:38s}  {solo_oof:>9.5f}  {test_a.mean():>8.3f}  {t_seen:>8.3f}  {t_unseen:>8.3f}  {t_unseen/t_seen:.3f}")
    cmod[fn] = (oof_a, test_a, solo_oof)

print("\n" + "="*70)
print("Part 2: Moderate unseen boost candidates")
print("="*70)

candidates = [
    ("A_w1u25_w2u12",   2000, 0.25, 5500, 0.12),
    ("A_w1u30_w2u15",   2000, 0.30, 5500, 0.15),
    ("B_n1u5000_n2u8k", 5000, 0.15, 8000, 0.08),
    ("B_n1u7000_n2u11k",7000, 0.15,11000, 0.08),
    ("AB_n1u4000_w1u25_n2u8k_w2u12", 4000, 0.25, 8000, 0.12),
    ("AB_n1u5000_w1u25_n2u11k_w2u12", 5000, 0.25, 11000, 0.12),
    ("AB_n1u7000_w1u20_n2u15k_w2u10", 7000, 0.20, 15000, 0.10),
    ("AB_n1u10000_w1u20_n2u15k_w2u10",10000, 0.20, 15000, 0.10),
]

saved = []
for name, n1u, w1u, n2u, w2u in candidates:
    t = make_asym(n1u, w1u, n2u, w2u)
    print(f"  {name:40s}  test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}  OOF={dual_mae:.5f}")
    saved.append((name, n1u, w1u, n2u, w2u, t))

print("\n" + "="*70)
print("Part 3: Triple specialist for unseen -- rh + slhm + another")
print("Best cascade for unseen + standard seen gate")
print("="*70)

# Find the cascade model with highest unseen prediction (likely overestimates but informative)
# Filter to ones with reasonable OOF
for fn, (oof_a, test_a, solo_oof) in cmod.items():
    if solo_oof > 30 and test_a[unseen_mask].mean() > rh_t[unseen_mask].mean():
        print(f"  High-unseen extreme model: {fn}  solo_OOF={solo_oof:.3f}  unseen={test_a[unseen_mask].mean():.3f}")

# Triple blend on unseen rows: fw4 + rh + slhm + cb_raw?
cb_cascade_fn = 'spec_cb_raw_oof.npy'
cb_cascade_fp = os.path.join(cascade_dir, cb_cascade_fn)
if os.path.exists(cb_cascade_fp):
    cbo = np.load(cb_cascade_fp)[id2]
    cbt = np.load(os.path.join(cascade_dir, 'spec_cb_raw_test.npy'))[te_id2]
    print(f"\nspec_cb_raw: solo_OOF={mae_fn(cbo):.5f}  test={cbt.mean():.3f}  unseen={cbt[unseen_mask].mean():.3f}")
    for w_cb in [0.05, 0.10, 0.15, 0.20]:
        t = make_asym(7000, 0.20, 15000, 0.10)  # moderate unseen boost
        t[unseen_mask] = np.clip((1-w_cb)*t[unseen_mask] + w_cb*cbt[unseen_mask], 0, None)
        print(f"  AB_7k_20_15k_10 + cb_raw w_unseen={w_cb}: unseen={t[unseen_mask].mean():.3f}  test={t.mean():.3f}")

print("\n" + "="*70)
print("Part 4: Oracle seq models on unseen rows")
print("="*70)
oracle_seq_dir = 'results/oracle_seq/'
print(f"{'filename':45s}  {'unseen':>8}  {'seen':>8}")
for fp in sorted(glob.glob(os.path.join(oracle_seq_dir, 'test_C_*.npy'))):
    fn = os.path.basename(fp)
    t_arr = np.load(fp)
    if len(t_arr) != len(test_raw): continue
    print(f"  {fn:43s}  {t_arr[unseen_mask].mean():>8.3f}  {t_arr[seen_mask].mean():>8.3f}")

print("\n" + "="*70)
print("Part 5: Save top 3 unseen-boosted candidates")
print("="*70)
sub_template = pd.read_csv('sample_submission.csv')
for i, (name, n1u, w1u, n2u, w2u, t) in enumerate(saved[:5]):
    sub = sub_template.copy()
    sub['avg_delay_minutes_next_30m'] = t
    fname = f"submission_unsBst_{name}_OOF{dual_mae:.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  test={t.mean():.3f}  seen={t[seen_mask].mean():.3f}  unseen={t[unseen_mask].mean():.3f}")
