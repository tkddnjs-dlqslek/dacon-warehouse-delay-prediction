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
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3
    w2 = fw['iter_r2'] + dr2
    w3 = fw['iter_r3'] + dr3
    fx  = wm*mega_o + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof, 0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_o, oracle_t = make_pred(0.0, 0.0, 0.0, 0.64, 0.0)
bb_o,     bb_t     = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)

mae   = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
tmean = lambda t: float(np.mean(t))
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts)
umean = lambda t: float(np.mean(t[unseen_mask]))

print(f"oracle_NEW: OOF={mae(oracle_o):.5f}  test={tmean(oracle_t):.3f}")
print(f"best_base:  OOF={mae(bb_o):.5f}   test={tmean(bb_t):.3f}")

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
bb_slh_o = 0.92*bb_o + 0.08*slh_o
bb_slh_t = 0.92*bb_t + 0.08*slh_t
print(f"bb+slh0.08: OOF={mae(bb_slh_o):.5f}  test={tmean(bb_slh_t):.3f}")

fw4_o  = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t  = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae(fw4_o)
print(f"4way best:  OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")

# ── Load remaining cascade files ─────────────────────────────────────────────
cascade_dir = 'results/cascade'
names = ["spec_cb_raw","spec_cb_w30","spec_lgb_raw_huber","spec_lgb_raw_mae",
         "spec_lgb_w30_mae","clf","clf_v2","spec_avg","spec_v2_avg"]

print("\n" + "="*70)
print("Part 1: Cascade files blended with best_base")
print(f"{'name':<22}  solo_OOF   test    unseen  best_w blend_OOF   delta")
print("="*70)

cascade_data = {}
for name in names:
    oof_f = f"{cascade_dir}/{name}_oof.npy"
    if not os.path.exists(oof_f): print(f"MISSING: {name}"); continue
    o_raw = np.load(oof_f); t_raw = np.load(f"{cascade_dir}/{name}_test.npy")
    if o_raw.shape[0] == len(train_ls):
        o = np.array(o_raw)[id2]; t = np.array(t_raw)[te_id2]
    else:
        o = np.array(o_raw); t = np.array(t_raw)
    cascade_data[name] = (o, t)
    solo = mae(o)
    best_w, best_bl = 0, mae(bb_o)
    for w in [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20]:
        m = mae((1-w)*bb_o + w*o)
        if m < best_bl: best_bl, best_w = m, w
    print(f"{name:<22}  {solo:8.5f}  {tmean(t):6.3f}  {umean(t):7.3f}  w={best_w:.2f}  {best_bl:.5f}  {best_bl-mae(bb_o):+.6f}")

# ── Part 2: Detailed analysis for small improvements ─────────────────────────
print("\n" + "="*70)
print("Part 2: Fine-grained analysis of all improving cascade files")
print("="*70)

bb_mae = mae(bb_o)
for name, (o, t) in cascade_data.items():
    best_w, best_bl = 0, bb_mae
    for w in np.arange(0.005, 0.25, 0.005):
        m = mae((1-w)*bb_o + w*o)
        if m < best_bl: best_bl, best_w = m, w
    if best_bl < bb_mae - 0.00010:
        bl_t = (1-best_w)*bb_t + best_w*t
        print(f"\n[{name}] best w={best_w:.3f}: OOF={best_bl:.5f} ({best_bl-bb_mae:+.6f})  test={tmean(bl_t):.3f}  unseen={umean(bl_t):.3f}")

# ── Part 3: Adding to 4-way blend ────────────────────────────────────────────
print("\n" + "="*70)
print("Part 3: Cascade files added on top of 4-way blend")
print("="*70)

any_improvement = False
for name, (o, t) in cascade_data.items():
    best_w2, best_m2 = 0, fw4_mae
    for w in np.arange(0.005, 0.15, 0.005):
        norm = 1 + w
        m = mae((fw4_o + w*o) / norm)
        if m < best_m2: best_m2, best_w2 = m, w
    if best_m2 < fw4_mae - 0.00005:
        bl_t2 = (fw4_t + best_w2*t) / (1+best_w2)
        print(f"[{name}] w={best_w2:.3f}: OOF={best_m2:.5f} ({best_m2-fw4_mae:+.6f})  test={tmean(bl_t2):.3f}  unseen={umean(bl_t2):.3f}")
        any_improvement = True
if not any_improvement:
    print("None of the cascade files improve the 4-way blend.")

# ── Part 4: spec_lgb_raw_huber correlation analysis ──────────────────────────
print("\n" + "="*70)
print("Part 4: spec_lgb_raw_huber analysis")
print("="*70)

if "spec_lgb_raw_huber" in cascade_data:
    rh_o, rh_t = cascade_data["spec_lgb_raw_huber"]
    print(f"Corr(slh_w30, raw_huber):   {np.corrcoef(slh_o, rh_o)[0,1]:.4f}")
    print(f"Corr(raw_huber, oracle_NEW): {np.corrcoef(rh_o, oracle_o)[0,1]:.4f}")

    # Even though solo OOF is bad, maybe residuals at extremes help?
    y_p95 = np.percentile(y_true, 95)
    for label, mask in [
        (">p95 y_true", y_true > y_p95),
        (">p90 y_true", y_true > np.percentile(y_true, 90)),
    ]:
        n = mask.sum()
        mae_oracle = mae(oracle_o[mask])
        mae_slh    = mae(slh_o[mask])
        mae_rh     = mae(rh_o[mask])
        mae_bb     = mae(bb_o[mask])
        print(f"  [{label}] n={n}  bb={mae_bb:.2f}  oracle={mae_oracle:.2f}  slh_w30={mae_slh:.2f}  raw_huber={mae_rh:.2f}")

# ── Part 5: CLF-gated SLH boost ──────────────────────────────────────────────
print("\n" + "="*70)
print("Part 5: CLF-gated analysis")
print("="*70)

for clf_name in ["clf", "clf_v2"]:
    if clf_name not in cascade_data: continue
    clf_o, clf_t = cascade_data[clf_name]
    print(f"\n[{clf_name}] mean={clf_o.mean():.4f}  range=[{clf_o.min():.3f},{clf_o.max():.3f}]")

    # What does CLF predict vs y_true?
    high_clf = clf_o > np.percentile(clf_o, 90)
    print(f"  When clf>p90: mean y_true={np.mean(y_true[high_clf]):.2f}  mean oracle_res={(np.mean((oracle_o-y_true)[high_clf])):+.2f}  mean slh_res={(np.mean((slh_o-y_true)[high_clf])):+.2f}")
    low_clf = clf_o < np.percentile(clf_o, 10)
    print(f"  When clf<p10: mean y_true={np.mean(y_true[low_clf]):.2f}  mean oracle_res={(np.mean((oracle_o-y_true)[low_clf])):+.2f}  mean slh_res={(np.mean((slh_o-y_true)[low_clf])):+.2f}")

    # Try using clf as soft gate for SLH
    # Normalize clf to [0,1] range
    clf_norm_o = (clf_o - clf_o.min()) / (clf_o.max() - clf_o.min() + 1e-9)
    clf_norm_t = (clf_t - clf_t.min()) / (clf_t.max() - clf_t.min() + 1e-9)

    print("  Soft gate: slh_w = max_slh * clf_norm")
    for max_slh in [0.05, 0.08, 0.10, 0.15]:
        slh_w_o = max_slh * clf_norm_o
        slh_w_t = max_slh * clf_norm_t
        mean_w = slh_w_o.mean()
        bl_o = (1 - slh_w_o)*bb_o + slh_w_o*slh_o
        bl_t = (1 - slh_w_t)*bb_t + slh_w_t*slh_t
        m = mae(bl_o)
        print(f"    max_slh={max_slh:.2f} (mean_w={mean_w:.4f}): OOF={m:.5f} ({m-bb_mae:+.6f})  test={tmean(bl_t):.3f}")

# ── Part 6: Orthogonality search — combine two low-corr cascade files ─────────
print("\n" + "="*70)
print("Part 6: Pairwise cascade combination on top of bb+slh")
print("="*70)

bb_slh_mae = mae(bb_slh_o)
print(f"Reference bb+slh: OOF={bb_slh_mae:.5f}  test={tmean(bb_slh_t):.3f}")

# Among files that are less correlated with SLH, check pairs
pairs_checked = 0
for n1, (o1, t1) in cascade_data.items():
    c1 = np.corrcoef(slh_o, o1)[0,1]
    if c1 > 0.95: continue  # too correlated with slh
    for w1 in [0.02, 0.04, 0.06]:
        bl_o = (1-w1)*bb_slh_o + w1*o1
        bl_t = (1-w1)*bb_slh_t + w1*t1
        m = mae(bl_o)
        if m < bb_slh_mae - 0.0002:
            print(f"  bb+slh+{n1}(w={w1:.2f}): OOF={m:.5f} ({m-bb_slh_mae:+.6f})  test={tmean(bl_t):.3f}  corr_slh={c1:.3f}")

# ── Part 7: Try all oracle_seq files individually  ───────────────────────────
print("\n" + "="*70)
print("Part 7: All oracle_seq files vs best_base (top 5 improvers)")
print("="*70)

oseq_dir = 'results/oracle_seq'
oseq_files = [f for f in os.listdir(oseq_dir) if f.startswith('oof_seqC_') and f.endswith('.npy')]
results_oseq = []
for fn in oseq_files:
    o = np.load(f"{oseq_dir}/{fn}")
    tname = fn.replace('oof_seqC_','').replace('.npy','')
    tf = f"{oseq_dir}/test_C_{tname}.npy"
    if not os.path.exists(tf): continue
    t = np.load(tf)
    if o.shape[0] != len(train_raw) or t.shape[0] != len(test_raw): continue
    best_w, best_bl = 0, bb_mae
    for w in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        m = mae((1-w)*bb_o + w*o)
        if m < best_bl: best_bl, best_w = m, w
    if best_bl < bb_mae:
        results_oseq.append((best_bl - bb_mae, tname, best_w, best_bl, tmean((1-best_w)*bb_t+best_w*t), umean((1-best_w)*bb_t+best_w*t)))

results_oseq.sort()
print(f"{'name':<28}  w   blend_OOF    delta    test   unseen")
for delta, name, w, bl, tm, um in results_oseq[:10]:
    print(f"  {name:<26}  {w:.2f}  {bl:.5f}  {delta:+.6f}  {tm:.3f}  {um:.3f}")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"oracle_NEW:    OOF={mae(oracle_o):.5f}  test={tmean(oracle_t):.3f}")
print(f"best_base:     OOF={mae(bb_o):.5f}  test={tmean(bb_t):.3f}")
print(f"bb+slh0.08:    OOF={mae(bb_slh_o):.5f}  test={tmean(bb_slh_t):.3f}  unseen={umean(bb_slh_t):.3f}")
print(f"4way best:     OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
