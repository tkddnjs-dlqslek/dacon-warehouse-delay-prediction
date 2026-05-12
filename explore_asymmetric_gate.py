"""
Asymmetric tail gate: different raw_huber weights for seen vs unseen test rows.
Key insight: raw_huber's high predictions (unseen_mean=61.163) better match
the domain shift for unseen layouts. Seen layouts may not need the same boost.
Also: try a pure unseen-only boost strategy.
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
cb_oof = np.clip(d33['meta_oofs']['cb'][id2],0,None)
cb_test= np.clip(d33['meta_tests']['cb'][te_id2],0,None)

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

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

def make_pred(w34=0.0, dr2=0.0, dr3=0.0, wf=0.64, w_cb=0.0):
    m_o=(1-w34)*mega33_oof+w34*mega34_oof; m_t=(1-w34)*mega33_test+w34*mega34_test
    wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx  = wm*m_o+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
    fxt = wm*m_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
    wr=1-wf; wxgb=0.12*wr/0.36; wlv2=0.16*wr/0.36; wrem=0.08*wr/0.36
    oo = np.clip(wf*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo+w_cb*cb_oof, 0, None)
        ot = np.clip((1-w_cb)*ot+w_cb*cb_test, 0, None)
    return oo, ot

bb_o, bb_t = make_pred(0.25,-0.04,-0.02,0.72,0.12)
mae_fn  = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
tmean   = lambda t: float(np.mean(t))
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts)
seen_mask_t   = test_raw['layout_id'].isin(train_layouts).values
unseen_mask_t = ~seen_mask_t
umean = lambda t: float(np.mean(t[unseen_mask_t]))
smean = lambda t: float(np.mean(t[seen_mask_t]))

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
fw4_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw = np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o = np.array(rh_o_raw)[id2] if rh_o_raw.shape[0] == len(train_ls) else np.array(rh_o_raw)
rh_t = np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0] == len(test_ls) else np.array(rh_t_raw)

print(f"4way: OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  seen={smean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"rh:   test={tmean(rh_t):.3f}  seen={smean(rh_t):.3f}  unseen={umean(rh_t):.3f}")
n_unseen = unseen_mask_t.sum(); n_seen = (~unseen_mask_t).sum()
print(f"Test: n_seen={n_seen}, n_unseen={n_unseen}")

# Best tail gate (reference)
thresh_o2000 = np.sort(fw4_o)[-2000]; thresh_t2000 = np.sort(fw4_t)[-2000]
mask_h = fw4_o >= thresh_o2000; mask_t = fw4_t >= thresh_t2000
best_tail_o = fw4_o.copy(); best_tail_t = fw4_t.copy()
best_tail_o[mask_h] = 0.80*fw4_o[mask_h] + 0.20*rh_o[mask_h]
best_tail_t[mask_t] = 0.80*fw4_t[mask_t] + 0.20*rh_t[mask_t]
print(f"tail_2000_w20: OOF={mae_fn(best_tail_o):.5f}  test={tmean(best_tail_t):.3f}  seen={smean(best_tail_t):.3f}  unseen={umean(best_tail_t):.3f}")

# ── Part 1: Asymmetric test gate ─────────────────────────────────────────────
print("\n" + "="*70)
print("Part 1: Asymmetric gate — different weights for seen/unseen TEST rows")
print("OOF uses seen-layout training data, so no asymmetry there.")
print("="*70)

# For test: apply rh differently to seen vs unseen
# OOF stays the same as tail_2000_w20 (only changes test predictions)
print("Strategy: fw4 tail gate training, but asymmetric on test")
print(f"{'ws_seen':>8}  {'ws_unseen':>10}  {'test':>7}  {'seen_t':>7}  {'unseen_t':>9}  note")

for w_seen in [0.05, 0.10, 0.15, 0.20]:
    for w_unseen in [0.20, 0.30, 0.40, 0.50]:
        # Apply different weight to seen vs unseen test rows in the gate
        adj_t = fw4_t.copy()
        # Seen test rows in gate
        gate_seen   = mask_t & seen_mask_t
        gate_unseen = mask_t & unseen_mask_t
        adj_t[gate_seen]   = (1-w_seen)*fw4_t[gate_seen]   + w_seen*rh_t[gate_seen]
        adj_t[gate_unseen] = (1-w_unseen)*fw4_t[gate_unseen] + w_unseen*rh_t[gate_unseen]
        tm = tmean(adj_t)
        sm = smean(adj_t)
        um = umean(adj_t)
        note = ""
        if um > 23.5: note = "HIGH_UNSEEN"
        elif um < 22.9: note = "LOW_UNSEEN"
        print(f"  {w_seen:8.2f}  {w_unseen:10.2f}  {tm:7.3f}  {sm:7.3f}  {um:9.3f}  {note}")

# ── Part 2: Unseen-only boost (no training/OOF change) ───────────────────────
print("\n" + "="*70)
print("Part 2: Unseen-only test boost with raw_huber")
print("(OOF = 4way OOF, only test changes)")
print("="*70)

# Simple: for unseen test rows above prediction threshold, blend in rh
for w_unseen in [0.10, 0.20, 0.30, 0.40, 0.50]:
    # Apply to ALL unseen test rows
    adj_t = fw4_t.copy()
    adj_t[unseen_mask_t] = (1-w_unseen)*fw4_t[unseen_mask_t] + w_unseen*rh_t[unseen_mask_t]
    print(f"  w_unseen={w_unseen:.2f} (all unseen): test={tmean(adj_t):.3f}  seen={smean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

print()
# Apply to unseen above threshold
for pct_gate in [0, 25, 50, 75]:
    if pct_gate == 0:
        gate_unseen = unseen_mask_t
    else:
        thresh_u = np.percentile(fw4_t[unseen_mask_t], pct_gate)
        gate_unseen = unseen_mask_t & (fw4_t > thresh_u)
    n_gated = gate_unseen.sum()
    for w_unseen in [0.20, 0.30, 0.40]:
        adj_t = fw4_t.copy()
        adj_t[gate_unseen] = (1-w_unseen)*fw4_t[gate_unseen] + w_unseen*rh_t[gate_unseen]
        print(f"  unseen_gate>p{pct_gate:2d}(n={n_gated:4d}), w={w_unseen:.2f}: test={tmean(adj_t):.3f}  seen={smean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

# ── Part 3: Blend tail gate with slhm gate for conservative option ────────────
print("\n" + "="*70)
print("Part 3: slhm tail gate only (conservative, less test inflation)")
print("="*70)

slhm_o_raw = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')
slhm_t_raw = np.load('results/cascade/spec_lgb_w30_mae_test.npy')
slhm_o = np.array(slhm_o_raw)[id2] if slhm_o_raw.shape[0] == len(train_ls) else np.array(slhm_o_raw)
slhm_t = np.array(slhm_t_raw)[te_id2] if slhm_t_raw.shape[0] == len(test_ls) else np.array(slhm_t_raw)

print(f"slhm: test={tmean(slhm_t):.3f}  seen={smean(slhm_t):.3f}  unseen={umean(slhm_t):.3f}")

best_slhm = fw4_mae; best_slhm_c = None
for top_n in [500, 1000, 1500, 2000, 2500, 3000]:
    thresh_o = np.sort(fw4_o)[-top_n]
    thresh_t_s = np.sort(fw4_t)[-top_n]
    mask_h2 = fw4_o >= thresh_o; mask_t2 = fw4_t >= thresh_t_s
    for w in np.arange(0.02, 0.20, 0.02):
        adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
        adj_o[mask_h2] = (1-w)*fw4_o[mask_h2] + w*slhm_o[mask_h2]
        adj_t[mask_t2] = (1-w)*fw4_t[mask_t2] + w*slhm_t[mask_t2]
        m = mae_fn(adj_o)
        if m < best_slhm:
            best_slhm = m
            best_slhm_c = (top_n, w, tmean(adj_t), umean(adj_t))

if best_slhm_c:
    top_n, w, tm, um = best_slhm_c
    print(f"BEST slhm tail: n={top_n}, w={w:.2f}: OOF={best_slhm:.5f} ({best_slhm-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")

# ── Part 4: Summary of all options and LB strategy ───────────────────────────
print("\n" + "="*70)
print("Part 4: Submission candidates ranked by expected LB impact")
print("="*70)

# Standard tail gate: n=2000, w=0.20
thresh_o = np.sort(fw4_o)[-2000]; thresh_t = np.sort(fw4_t)[-2000]
mask_h = fw4_o >= thresh_o; mask_t = fw4_t >= thresh_t
tg_o = fw4_o.copy(); tg_t = fw4_t.copy()
tg_o[mask_h] = 0.80*fw4_o[mask_h] + 0.20*rh_o[mask_h]
tg_t[mask_t] = 0.80*fw4_t[mask_t] + 0.20*rh_t[mask_t]

# Asymmetric: seen w=0.10, unseen w=0.30, gate n=2000
asym_t = fw4_t.copy()
gate_seen   = mask_t & seen_mask_t; gate_unseen = mask_t & unseen_mask_t
asym_t[gate_seen]   = 0.90*fw4_t[gate_seen]   + 0.10*rh_t[gate_seen]
asym_t[gate_unseen] = 0.70*fw4_t[gate_unseen] + 0.30*rh_t[gate_unseen]

# Conservative: n=1000, w=0.25
thresh_o_c = np.sort(fw4_o)[-1000]; thresh_t_c = np.sort(fw4_t)[-1000]
mask_h_c = fw4_o >= thresh_o_c; mask_t_c = fw4_t >= thresh_t_c
tg_c_o = fw4_o.copy(); tg_c_t = fw4_t.copy()
tg_c_o[mask_h_c] = 0.75*fw4_o[mask_h_c] + 0.25*rh_o[mask_h_c]
tg_c_t[mask_t_c] = 0.75*fw4_t[mask_t_c] + 0.25*rh_t[mask_t_c]

configs_final = [
    ("4way baseline",                 fw4_o, fw4_t),
    ("tail_n2000_w0.20",              tg_o,  tg_t),
    ("asym_n2000_seen0.10_un0.30",    tg_o,  asym_t),  # OOF same as tg_o
    ("tail_n1000_w0.25",              tg_c_o, tg_c_t),
]
print(f"{'name':<38}  OOF      test   seen_t  unseen_t")
for name, co, ct in configs_final:
    print(f"  {name:<36}  {mae_fn(co):.5f}  {tmean(ct):.3f}  {smean(ct):.3f}  {umean(ct):.3f}")

# Save asymmetric submission
sample = pd.read_csv('sample_submission.csv')
sample['answer'] = np.clip(asym_t, 0, None)
sample.to_csv("submission_4way_tailRH_asym_seen10_unseen30_OOF8.37205.csv", index=False)
print(f"\nSaved: submission_4way_tailRH_asym_seen10_unseen30_OOF8.37205.csv")
print(f"  test={tmean(asym_t):.3f}  seen={smean(asym_t):.3f}  unseen={umean(asym_t):.3f}")

# Save slhm tail gate
if best_slhm_c:
    top_n, w, tm, um = best_slhm_c
    thresh_o_s = np.sort(fw4_o)[-top_n]
    thresh_t_s = np.sort(fw4_t)[-top_n]
    mask_h_s = fw4_o >= thresh_o_s; mask_t_s = fw4_t >= thresh_t_s
    slhm_final_o = fw4_o.copy(); slhm_final_t = fw4_t.copy()
    slhm_final_o[mask_h_s] = (1-w)*fw4_o[mask_h_s] + w*slhm_o[mask_h_s]
    slhm_final_t[mask_t_s] = (1-w)*fw4_t[mask_t_s] + w*slhm_t[mask_t_s]
    sample['answer'] = np.clip(slhm_final_t, 0, None)
    fname_slhm = f"submission_4way_tailSLHM_n{top_n}_w{int(w*100):02d}_OOF{best_slhm:.5f}.csv"
    sample.to_csv(fname_slhm, index=False)
    print(f"Saved: {fname_slhm}  test={tm:.3f}  unseen={um:.3f}")

print("\n" + "="*70)
print("FINAL SUMMARY — candidates for LB submission")
print("="*70)
print("Ranked by OOF improvement (best first):")
print(f"  1. tail_n2000_w0.20:    OOF=8.37205  test=19.863  unseen=23.396  +test_inflation")
print(f"  2. tail_n1000_w0.25:    OOF=8.37414  test=19.664  unseen=23.157  moderate")
print(f"  3. 4way baseline:       OOF=8.37624  test=19.413  unseen=22.802  conservative")
print(f"  4. slhm tail (if good): OOF={best_slhm:.5f}  conservative test_mean")
