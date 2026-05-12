"""
Final combination analysis:
1. 5-way (4way + xgb_v31) + tail gate
2. tail gate on different base blends
3. spec_lgb_w30_mae as tail gate (very high test predictions)
4. Combined 6-way blend
5. Save best candidates for submission
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

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

mega33_oof  = d33['meta_avg_oof'][id2]; mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]; mega34_test = d34['meta_avg_test'][te_id2]
cb_oof = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test= np.clip(d33['meta_tests']['cb'][te_id2], 0, None)

rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]

xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
v31_o  = np.load('results/oracle_seq/oof_seqC_xgb_v31.npy')
v31_t  = np.load('results/oracle_seq/test_C_xgb_v31.npy')

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

def make_pred(w34=0.0, dr2=0.0, dr3=0.0, wf=0.64, w_cb=0.0):
    m_o = (1-w34)*mega33_oof + w34*mega34_oof
    m_t = (1-w34)*mega33_test + w34*mega34_test
    wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx  = wm*m_o + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*m_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    wr=1-wf; wxgb=0.12*wr/0.36; wlv2=0.16*wr/0.36; wrem=0.08*wr/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

bb_o, bb_t = make_pred(0.25,-0.04,-0.02,0.72,0.12)
mae_fn  = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
tmean   = lambda t: float(np.mean(t))
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts)
umean   = lambda t: float(np.mean(t[unseen_mask]))

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw = np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o = np.array(rh_o_raw)[id2] if rh_o_raw.shape[0] == len(train_ls) else np.array(rh_o_raw)
rh_t = np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0] == len(test_ls) else np.array(rh_t_raw)

slhm_o_raw = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')
slhm_t_raw = np.load('results/cascade/spec_lgb_w30_mae_test.npy')
slhm_o = np.array(slhm_o_raw)[id2] if slhm_o_raw.shape[0] == len(train_ls) else np.array(slhm_o_raw)
slhm_t = np.array(slhm_t_raw)[te_id2] if slhm_t_raw.shape[0] == len(test_ls) else np.array(slhm_t_raw)

# Known baselines
fw4_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)

# 5-way: replace xgbc with v31 (from Part 10: slight improvement)
fw5_o = 0.74*bb_o + 0.08*slh_o + 0.10*v31_o + 0.08*mono_o
fw5_t = 0.74*bb_t + 0.08*slh_t + 0.10*v31_t + 0.08*mono_t
fw5_mae = mae_fn(fw5_o)

print(f"4way:  OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"5way:  OOF={fw5_mae:.5f}  test={tmean(fw5_t):.3f}  unseen={umean(fw5_t):.3f}")

def apply_tail_gate(base_o, base_t, gate_o, gate_t, top_n, w):
    thresh_o = np.sort(gate_o)[-top_n]
    thresh_t = np.sort(gate_t)[-top_n]
    mask_h = gate_o >= thresh_o
    mask_t = gate_t >= thresh_t
    adj_o = base_o.copy()
    adj_t = base_t.copy()
    adj_o[mask_h] = (1-w)*base_o[mask_h] + w*rh_o[mask_h]
    adj_t[mask_t] = (1-w)*base_t[mask_t] + w*rh_t[mask_t]
    return adj_o, adj_t

# ── Part 1: Tail gate on different base blends ───────────────────────────────
print("\n" + "="*70)
print("Part 1: Tail gate (top_n=2000, w=0.20) on different base blends")
print("="*70)

bases = [
    ("4way",  fw4_o, fw4_t),
    ("5way",  fw5_o, fw5_t),
    ("bb+slh", 0.92*bb_o+0.08*slh_o, 0.92*bb_t+0.08*slh_t),
    ("best_base", bb_o, bb_t),
]
for name, bo, bt in bases:
    for top_n, w in [(2000,0.20), (1000,0.30), (2500,0.15)]:
        adj_o, adj_t = apply_tail_gate(bo, bt, bo, bt, top_n, w)
        m = mae_fn(adj_o)
        base_m = mae_fn(bo)
        print(f"  [{name}]+tail(n={top_n},w={w:.2f}): OOF={m:.5f} ({m-base_m:+.6f})  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

# ── Part 2: Use spec_lgb_w30_mae as tail specialist ──────────────────────────
print("\n" + "="*70)
print("Part 2: spec_lgb_w30_mae as tail specialist (instead of raw_huber)")
print("="*70)

print(f"slhm predictions: train_mean={slhm_o.mean():.2f}  test_mean={slhm_t.mean():.2f}  unseen={umean(slhm_t):.3f}")
# Check extreme performance
for pct in [90, 95, 99]:
    thr = np.percentile(y_true, pct)
    mask = y_true > thr
    print(f"  >p{pct}: fw4_mae={np.mean(np.abs((fw4_o-y_true)[mask])):.2f}  slhm_mae={np.mean(np.abs((slhm_o-y_true)[mask])):.2f}  rh_mae={np.mean(np.abs((rh_o-y_true)[mask])):.2f}")

print("\nslhm tail gate (gate on fw4 predictions):")
best_slhm = fw4_mae; best_slhm_c = None
for top_n in [500, 1000, 1500, 2000, 2500]:
    thresh_o = np.sort(fw4_o)[-top_n]
    thresh_t = np.sort(fw4_t)[-top_n]
    mask_h = fw4_o >= thresh_o
    mask_t = fw4_t >= thresh_t
    for w in [0.01, 0.02, 0.03, 0.05]:
        adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
        adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*slhm_o[mask_h]
        adj_t[mask_t] = (1-w)*fw4_t[mask_t] + w*slhm_t[mask_t]
        m = mae_fn(adj_o)
        if m < best_slhm:
            best_slhm = m
            best_slhm_c = (top_n, w, tmean(adj_t), umean(adj_t))

if best_slhm_c:
    top_n, w, tm, um = best_slhm_c
    print(f"  BEST slhm tail: n={top_n}, w={w:.2f}: OOF={best_slhm:.5f} ({best_slhm-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
else:
    print("  slhm tail gate does not improve 4-way")

# ── Part 3: Combined tail gate: rh AND slhm together ─────────────────────────
print("\n" + "="*70)
print("Part 3: Combined tail gate (raw_huber + w30_mae together)")
print("="*70)

best_combo2 = fw4_mae; best_combo2_c = None
top_n_fixed = 2000; thresh_o = np.sort(fw4_o)[-top_n_fixed]; thresh_t = np.sort(fw4_t)[-top_n_fixed]
mask_h = fw4_o >= thresh_o; mask_t = fw4_t >= thresh_t

for wr in [0.15, 0.20, 0.25]:
    for wm in [0.01, 0.02, 0.03, 0.05]:
        wt = wr + wm
        adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
        adj_o[mask_h] = (1-wt)*fw4_o[mask_h] + wr*rh_o[mask_h] + wm*slhm_o[mask_h]
        adj_t[mask_t] = (1-wt)*fw4_t[mask_t] + wr*rh_t[mask_t] + wm*slhm_t[mask_t]
        m = mae_fn(adj_o)
        if m < best_combo2:
            best_combo2 = m
            best_combo2_c = (wr, wm, tmean(adj_t), umean(adj_t))

if best_combo2_c:
    wr, wm, tm, um = best_combo2_c
    print(f"BEST combo: rh_w={wr:.2f}+slhm_w={wm:.2f}: OOF={best_combo2:.5f} ({best_combo2-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
else:
    print("No improvement from combining both tail specialists")

# ── Part 4: Apply tail gate using raw_huber predictions as gate (not fw4) ────
print("\n" + "="*70)
print("Part 4: Use raw_huber itself as the gate signal")
print("="*70)

# rh has mean=45, so gating on rh>threshold selects rows where rh predicts high delay
# These may be different from rows where fw4 predicts high delay
print(f"rh_o: p90={np.percentile(rh_o,90):.1f}  p95={np.percentile(rh_o,95):.1f}  p99={np.percentile(rh_o,99):.1f}")
print(f"fw4_o: p90={np.percentile(fw4_o,90):.1f}  p95={np.percentile(fw4_o,95):.1f}  p99={np.percentile(fw4_o,99):.1f}")

# Overlap
gate_fw4 = set(np.where(fw4_o >= np.sort(fw4_o)[-2000])[0])
gate_rh  = set(np.where(rh_o  >= np.sort(rh_o)[-2000])[0])
print(f"fw4 top-2000 ∩ rh top-2000: {len(gate_fw4 & gate_rh)} rows")

# Gate on rh predictions
best_rh_gate = fw4_mae; best_rh_gate_c = None
for top_n in [1000, 1500, 2000, 2500]:
    thresh_o = np.sort(rh_o)[-top_n]
    thresh_t = np.sort(rh_t)[-top_n]
    mask_h = rh_o >= thresh_o
    mask_t = rh_t >= thresh_t
    for w in [0.10, 0.15, 0.20, 0.25]:
        adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
        adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]
        adj_t[mask_t] = (1-w)*fw4_t[mask_t] + w*rh_t[mask_t]
        m = mae_fn(adj_o)
        if m < best_rh_gate:
            best_rh_gate = m
            best_rh_gate_c = (top_n, w, tmean(adj_t), umean(adj_t))
        if m < fw4_mae - 0.001:
            print(f"  rh-gate top_n={top_n}, w={w:.2f}: OOF={m:.5f} ({m-fw4_mae:+.6f})  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

if best_rh_gate_c:
    top_n, w, tm, um = best_rh_gate_c
    print(f"BEST rh-gate: n={top_n}, w={w:.2f}: OOF={best_rh_gate:.5f} ({best_rh_gate-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")

# ── Part 5: Full fold-stability check for all saved submissions ───────────────
print("\n" + "="*70)
print("Part 5: Fold stability for top-3 tail gate configs")
print("="*70)

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

configs = [
    ("4way (baseline)",      fw4_o),
    ("4way+tail(2000,0.20)", None),
    ("4way+tail(2000,0.15)", None),
    ("4way+tail(1000,0.30)", None),
]

for name, bo in configs:
    if bo is None:
        # parse name
        parts = name.split("(")[1].rstrip(")")
        tn, tw = int(parts.split(",")[0]), float(parts.split(",")[1])
        thresh = np.sort(fw4_o)[-tn]
        mask_h = fw4_o >= thresh
        bo = fw4_o.copy()
        bo[mask_h] = (1-tw)*fw4_o[mask_h] + tw*rh_o[mask_h]

    fold_maes = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_raw, groups=groups)):
        fold_mae = np.mean(np.abs(np.clip(bo[va_idx],0,None) - y_true[va_idx]))
        fold_maes.append(fold_mae)
    overall_mae = mae_fn(bo)
    print(f"\n{name}: OOF={overall_mae:.5f}")
    print(f"  Folds: {[f'{m:.4f}' for m in fold_maes]}")
    print(f"  std={np.std(fold_maes):.5f}")

# ── Part 6: Save best new submissions ─────────────────────────────────────────
print("\n" + "="*70)
print("Part 6: Save best new combinations")
print("="*70)

sample = pd.read_csv('sample_submission.csv')

submissions_to_save = []

# Best OOF tail gate
best_tn, best_tw = 2000, 0.20
thresh_o = np.sort(fw4_o)[-best_tn]; thresh_t = np.sort(fw4_t)[-best_tn]
mask_h = fw4_o >= thresh_o; mask_t = fw4_t >= thresh_t
adj_t = fw4_t.copy(); adj_o2 = fw4_o.copy()
adj_t[mask_t] = (1-best_tw)*fw4_t[mask_t] + best_tw*rh_t[mask_t]
adj_o2[mask_h] = (1-best_tw)*fw4_o[mask_h] + best_tw*rh_o[mask_h]
m_best = mae_fn(adj_o2)
submissions_to_save.append((f"submission_4way_tailRH_n{best_tn}_w{int(best_tw*100):02d}_OOF{m_best:.5f}.csv", adj_t, m_best, tmean(adj_t), umean(adj_t)))

# Conservative tail gate (fewer rows)
for tn, tw in [(1000, 0.25), (250, 0.30), (500, 0.30)]:
    thresh_o2 = np.sort(fw4_o)[-tn]; thresh_t2 = np.sort(fw4_t)[-tn]
    mask_h2 = fw4_o >= thresh_o2; mask_t2 = fw4_t >= thresh_t2
    adj_o3 = fw4_o.copy(); adj_t3 = fw4_t.copy()
    adj_o3[mask_h2] = (1-tw)*fw4_o[mask_h2] + tw*rh_o[mask_h2]
    adj_t3[mask_t2] = (1-tw)*fw4_t[mask_t2] + tw*rh_t[mask_t2]
    m3 = mae_fn(adj_o3)
    submissions_to_save.append((f"submission_4way_tailRH_n{tn}_w{int(tw*100):02d}_OOF{m3:.5f}.csv", adj_t3, m3, tmean(adj_t3), umean(adj_t3)))

for fname, t_preds, m, tm, um in submissions_to_save:
    sample['answer'] = np.clip(t_preds, 0, None)
    sample.to_csv(fname, index=False)
    print(f"Saved: {fname}  OOF={m:.5f}  test={tm:.3f}  unseen={um:.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"4way:                OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
for fname, t_preds, m, tm, um in submissions_to_save:
    print(f"{fname[:50]}: OOF={m:.5f}  test={tm:.3f}  unseen={um:.3f}")
