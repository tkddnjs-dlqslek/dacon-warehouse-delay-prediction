"""
Focused follow-up: Part 5-7 from explore_remaining_cascade.py
(Parts 1-3 already confirmed: no remaining cascade file improves 4-way)
Key mission: Part 7 — scan ALL oracle_seq files vs best_base
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls  = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos    = {row['ID']:i for i,row in train_ls.iterrows()}
id2       = [ls_pos[i] for i in train_raw['ID'].values]
test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof      = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test     = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)

rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof    = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof    = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof    = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test   = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test   = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test   = np.load('results/iter_pseudo/round3_test.npy')[te_id2]

xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t  = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t  = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t  = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

def make_pred(w34=0.0, dr2=0.0, dr3=0.0, wf=0.64, w_cb=0.0):
    mega_o = (1-w34)*mega33_oof + w34*mega34_oof
    mega_tt = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3
    w2 = fw['iter_r2'] + dr2
    w3 = fw['iter_r3'] + dr3
    fx  = wm*mega_o  + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_tt + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_o, oracle_t = make_pred(0.0, 0.0, 0.0, 0.64, 0.0)
bb_o,     bb_t     = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)

mae_fn  = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
mae_sub = lambda p, mask: float(np.mean(np.abs(np.clip(p[mask],0,None) - y_true[mask])))
tmean   = lambda t: float(np.mean(t))
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts)
umean   = lambda t: float(np.mean(t[unseen_mask]))

bb_mae = mae_fn(bb_o)
print(f"oracle_NEW: OOF={mae_fn(oracle_o):.5f}")
print(f"best_base:  OOF={bb_mae:.5f}   test={tmean(bb_t):.3f}")

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]

fw4_o  = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t  = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)
print(f"4way best:  OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")

# ── Part 5: raw_huber at extremes ────────────────────────────────────────────
print("\n" + "="*70)
print("Part 5: spec_lgb_raw_huber at extreme y values")
print("="*70)

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
if rh_o_raw.shape[0] == len(train_ls):
    rh_o = np.array(rh_o_raw)[id2]
else:
    rh_o = np.array(rh_o_raw)

for pct in [90, 95, 99]:
    thr = np.percentile(y_true, pct)
    mask = y_true > thr
    n = mask.sum()
    mae_bb     = mae_sub(bb_o, mask)
    mae_oracle = mae_sub(oracle_o, mask)
    mae_slh    = mae_sub(slh_o, mask)
    mae_rh     = mae_sub(rh_o, mask)
    print(f">p{pct} y_true (n={n}):  bb={mae_bb:.2f}  oracle={mae_oracle:.2f}  slh_w30={mae_slh:.2f}  raw_huber={mae_rh:.2f}")

# ── Part 7: Scan ALL oracle_seq files  ───────────────────────────────────────
print("\n" + "="*70)
print("Part 7: All oracle_seq files blended with best_base")
print("="*70)

oseq_dir = 'results/oracle_seq'
oseq_files = sorted([f for f in os.listdir(oseq_dir) if f.startswith('oof_seqC_') and f.endswith('.npy')])
results_oseq = []
for fn in oseq_files:
    tname = fn.replace('oof_seqC_','').replace('.npy','')
    tf = f"{oseq_dir}/test_C_{tname}.npy"
    if not os.path.exists(tf): continue
    o = np.load(f"{oseq_dir}/{fn}")
    t = np.load(tf)
    if o.shape[0] != len(train_raw) or t.shape[0] != len(test_raw): continue
    best_w, best_bl = 0, bb_mae
    for w in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        m = mae_fn((1-w)*bb_o + w*o)
        if m < best_bl: best_bl, best_w = m, w
    bl_t = (1-best_w)*bb_t + best_w*t
    results_oseq.append((best_bl - bb_mae, tname, best_w, best_bl, tmean(bl_t), umean(bl_t)))

results_oseq.sort()
print(f"{'name':<30}  w    blend_OOF    delta    test   unseen")
for delta, name, w, bl, tm, um in results_oseq:
    marker = " <<" if delta < -0.0005 else ""
    print(f"  {name:<28}  {w:.2f}  {bl:.5f}  {delta:+.6f}  {tm:.3f}  {um:.3f}{marker}")

# ── Part 8: Top oracle_seq improvers vs 4-way blend ──────────────────────────
print("\n" + "="*70)
print("Part 8: Top oracle_seq improvers vs 4-way blend")
print("="*70)

# Load all oracle_seq into dict
oseq_data = {}
for fn in oseq_files:
    tname = fn.replace('oof_seqC_','').replace('.npy','')
    tf = f"{oseq_dir}/test_C_{tname}.npy"
    if not os.path.exists(tf): continue
    o = np.load(f"{oseq_dir}/{fn}")
    t = np.load(tf)
    if o.shape[0] == len(train_raw) and t.shape[0] == len(test_raw):
        oseq_data[tname] = (o, t)

# Which oracle_seq files improve BEST_BASE by most?
top_improvers = [(delta, name) for delta,name,*_ in results_oseq if delta < -0.0003]
print(f"Oracle_seq files improving best_base by >0.0003: {len(top_improvers)}")
for delta, name in top_improvers:
    print(f"  {name}: delta={delta:+.6f}")

# Test these top improvers added on 4-way
print("\nAdding top improvers to 4-way blend:")
any_found = False
for delta, name in top_improvers:
    if name not in oseq_data: continue
    o, t = oseq_data[name]
    best_w2, best_m2 = 0, fw4_mae
    for w in np.arange(0.02, 0.20, 0.02):
        norm = 1 + w
        m = mae_fn((fw4_o + w*o) / norm)
        if m < best_m2: best_m2, best_w2 = m, w
    if best_m2 < fw4_mae - 0.00005:
        bl_t2 = (fw4_t + best_w2*t) / (1+best_w2)
        print(f"  [{name}] w={best_w2:.2f}: OOF={best_m2:.5f} ({best_m2-fw4_mae:+.6f})  test={tmean(bl_t2):.3f}  unseen={umean(bl_t2):.3f}")
        any_found = True
if not any_found:
    print("  None of the top oracle_seq improvers help the 4-way blend.")

# ── Part 9: xgbc already used — check other xgb variants on 4-way ────────────
print("\n" + "="*70)
print("Part 9: XGB variant exploration for 5-way blends")
print("="*70)

xgb_variants = [n for n in oseq_data.keys() if 'xgb' in n and n not in ['xgb','xgb_combined','xgb_monotone']]
print(f"Available xgb variants: {xgb_variants}")

for name in xgb_variants:
    o, t = oseq_data[name]
    best_w2, best_m2 = 0, fw4_mae
    for w in [0.04, 0.06, 0.08, 0.10, 0.12]:
        norm = 1 + w
        m = mae_fn((fw4_o + w*o)/norm)
        if m < best_m2: best_m2, best_w2 = m, w
    if best_m2 < fw4_mae - 0.00005:
        bl_t2 = (fw4_t + best_w2*t)/(1+best_w2)
        print(f"[{name}] w={best_w2:.2f}: OOF={best_m2:.5f} ({best_m2-fw4_mae:+.6f})  test={tmean(bl_t2):.3f}  unseen={umean(bl_t2):.3f}")

# ── Part 10: 5-way search: replace xgbc or mono with a better oracle_seq ─────
print("\n" + "="*70)
print("Part 10: Replace xgbc/mono in 4-way with best oracle_seq alternatives")
print("="*70)

# For each replacement target, find the best oracle_seq substitute
for replace_target, target_o, target_t in [
    ("xgbc_o", xgbc_o, xgbc_t),
    ("mono_o",  mono_o,  mono_t),
]:
    best_rep = None; best_rep_mae = fw4_mae
    for name, (o,t) in oseq_data.items():
        if name in ['xgb','xgb_combined','xgb_monotone','log_v2','xgb_remaining']: continue
        # Replace target with this oracle_seq file at same weight
        if replace_target == "xgbc_o":
            bl_o = 0.74*bb_o + 0.08*slh_o + 0.10*o + 0.08*mono_o
            bl_t = 0.74*bb_t + 0.08*slh_t + 0.10*t + 0.08*mono_t
        else:
            bl_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*o
            bl_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*t
        m = mae_fn(bl_o)
        if m < best_rep_mae:
            best_rep_mae = m
            best_rep = (name, tmean(bl_t), umean(bl_t))
    if best_rep:
        name, tm, um = best_rep
        print(f"Replace {replace_target} with [{name}]: OOF={best_rep_mae:.5f} ({best_rep_mae-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
    else:
        print(f"No replacement for {replace_target} improves 4-way")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"best_base:  OOF={bb_mae:.5f}  test={tmean(bb_t):.3f}")
print(f"bb+slh:     OOF={mae_fn(0.92*bb_o+0.08*slh_o):.5f}  test={tmean(0.92*bb_t+0.08*slh_t):.3f}  unseen={umean(0.92*bb_t+0.08*slh_t):.3f}")
print(f"4way best:  OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
