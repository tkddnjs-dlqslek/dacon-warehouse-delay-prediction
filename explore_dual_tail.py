"""
Dual tail gate: apply both rh and slhm in sequence or combined.
Also: final sweep of parameters around best config.
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
unseen_mask_t = ~test_raw['layout_id'].isin(train_layouts).values
umean = lambda t: float(np.mean(t[unseen_mask_t]))

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
fw4_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw = np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o = np.array(rh_o_raw)[id2] if rh_o_raw.shape[0] == len(train_ls) else np.array(rh_o_raw)
rh_t = np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0] == len(test_ls) else np.array(rh_t_raw)

slhm_o_raw = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')
slhm_t_raw = np.load('results/cascade/spec_lgb_w30_mae_test.npy')
slhm_o = np.array(slhm_o_raw)[id2] if slhm_o_raw.shape[0] == len(train_ls) else np.array(slhm_o_raw)
slhm_t = np.array(slhm_t_raw)[te_id2] if slhm_t_raw.shape[0] == len(test_ls) else np.array(slhm_t_raw)

print(f"4way:  OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"rh:    test_mean={tmean(rh_t):.3f}  unseen={umean(rh_t):.3f}")
print(f"slhm:  test_mean={tmean(slhm_t):.3f}  unseen={umean(slhm_t):.3f}")

# Pre-compute best single tail gate baselines
thresh_rh_o  = np.sort(fw4_o)[-2000]; thresh_rh_t  = np.sort(fw4_t)[-2000]
thresh_slhm_o = np.sort(fw4_o)[-3000]; thresh_slhm_t = np.sort(fw4_t)[-3000]
mask_rh_o  = fw4_o >= thresh_rh_o;  mask_rh_t  = fw4_t >= thresh_rh_t
mask_slhm_o = fw4_o >= thresh_slhm_o; mask_slhm_t = fw4_t >= thresh_slhm_t

tg_rh_o = fw4_o.copy(); tg_rh_t = fw4_t.copy()
tg_rh_o[mask_rh_o]  = 0.80*fw4_o[mask_rh_o]  + 0.20*rh_o[mask_rh_o]
tg_rh_t[mask_rh_t]  = 0.80*fw4_t[mask_rh_t]  + 0.20*rh_t[mask_rh_t]

tg_slhm_o = fw4_o.copy(); tg_slhm_t = fw4_t.copy()
tg_slhm_o[mask_slhm_o] = 0.84*fw4_o[mask_slhm_o] + 0.16*slhm_o[mask_slhm_o]
tg_slhm_t[mask_slhm_t] = 0.84*fw4_t[mask_slhm_t] + 0.16*slhm_t[mask_slhm_t]

print(f"tg_rh:   OOF={mae_fn(tg_rh_o):.5f}  test={tmean(tg_rh_t):.3f}  unseen={umean(tg_rh_t):.3f}")
print(f"tg_slhm: OOF={mae_fn(tg_slhm_o):.5f}  test={tmean(tg_slhm_t):.3f}  unseen={umean(tg_slhm_t):.3f}")

# ── Part 1: Dual tail gate (rh first, then slhm on remainder) ────────────────
print("\n" + "="*70)
print("Part 1: Sequential dual gate (rh on top_n1, slhm on top_n2)")
print("="*70)

best_dual = mae_fn(tg_rh_o); best_dual_c = None
for n1 in [1500, 2000, 2500]:  # rh gate
    for w1 in [0.15, 0.20]:
        th_o1 = np.sort(fw4_o)[-n1]
        m1 = fw4_o >= th_o1
        m1t = fw4_t >= np.sort(fw4_t)[-n1]
        base_o = fw4_o.copy(); base_t = fw4_t.copy()
        base_o[m1] = (1-w1)*fw4_o[m1] + w1*rh_o[m1]
        base_t[m1t] = (1-w1)*fw4_t[m1t] + w1*rh_t[m1t]
        for n2 in [2000, 3000, 4000]:  # slhm gate on top of rh-adjusted
            for w2 in [0.04, 0.08, 0.12, 0.16]:
                # Gate on ORIGINAL fw4 (not adjusted) for consistency
                th_o2 = np.sort(fw4_o)[-n2]
                m2 = fw4_o >= th_o2
                m2t = fw4_t >= np.sort(fw4_t)[-n2]
                dual_o = base_o.copy(); dual_t = base_t.copy()
                dual_o[m2] = (1-w2)*base_o[m2] + w2*slhm_o[m2]
                dual_t[m2t] = (1-w2)*base_t[m2t] + w2*slhm_t[m2t]
                m = mae_fn(dual_o)
                if m < best_dual:
                    best_dual = m
                    best_dual_c = (n1,w1,n2,w2,tmean(dual_t),umean(dual_t))

if best_dual_c:
    n1,w1,n2,w2,tm,um = best_dual_c
    print(f"BEST dual: rh(n={n1},w={w1:.2f})+slhm(n={n2},w={w2:.2f}): OOF={best_dual:.5f} ({best_dual-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
else:
    print("No dual gate beats single rh gate")

# ── Part 2: Fine search around best rh config ─────────────────────────────────
print("\n" + "="*70)
print("Part 2: Ultra-fine rh tail gate search (refine around n=2000, w=0.20)")
print("="*70)

best_rh = fw4_mae; best_rh_c = None
for n in range(1600, 2600, 100):
    th_o = np.sort(fw4_o)[-n]; th_t = np.sort(fw4_t)[-n]
    m_o = fw4_o >= th_o; m_t = fw4_t >= th_t
    for w in np.arange(0.12, 0.32, 0.02):
        adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
        adj_o[m_o] = (1-w)*fw4_o[m_o] + w*rh_o[m_o]
        adj_t[m_t] = (1-w)*fw4_t[m_t] + w*rh_t[m_t]
        m = mae_fn(adj_o)
        if m < best_rh:
            best_rh = m
            best_rh_c = (n, w, tmean(adj_t), umean(adj_t))

if best_rh_c:
    n, w, tm, um = best_rh_c
    print(f"BEST refined: n={n}, w={w:.2f}: OOF={best_rh:.5f} ({best_rh-fw4_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
    # Show nearby configs
    for n2 in [n-100, n, n+100]:
        th_o = np.sort(fw4_o)[-n2]; th_t = np.sort(fw4_t)[-n2]
        m_o = fw4_o >= th_o; m_t = fw4_t >= th_t
        for w2 in [w-0.02, w, w+0.02]:
            if w2 <= 0 or w2 >= 1: continue
            adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
            adj_o[m_o] = (1-w2)*fw4_o[m_o] + w2*rh_o[m_o]
            adj_t[m_t] = (1-w2)*fw4_t[m_t] + w2*rh_t[m_t]
            m = mae_fn(adj_o)
            print(f"  n={n2}, w={w2:.2f}: OOF={m:.5f}  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

# ── Part 3: Validate best configs correlation with LB strategy ───────────────
print("\n" + "="*70)
print("Part 3: Summary table of all confirmed submissions")
print("="*70)

subs = [
    ("oracle_NEW (LB=9.7527)",          8.38247, 19.314, None,  "best LB"),
    ("best_base",                        8.38225, 19.391, None,  ""),
    ("bb+slh (OOF=8.37832)",             8.37832, 19.476, None,  ""),
    ("4way (OOF=8.37624)",               8.37624, 19.413, 22.802,""),
    ("tail_rh_n2000_w0.20 (new)",        8.37205, 19.863, 23.396,""),
    ("tail_rh_n2000_w0.25 (new)",        8.37206, 19.975, 23.545,""),
    ("tail_slhm_n3000_w0.16 (new)",      8.37221, 19.750, 23.231,""),
    ("asym_seen0.10_unseen0.30 (new)",   8.37205, 19.876, 23.694,""),
]
print(f"{'name':<42}  OOF      test   unseen")
for name, oof, tm, um, note in subs:
    um_str = f"{um:.3f}" if um else "  n/a "
    print(f"  {name:<40}  {oof:.5f}  {tm:.3f}  {um_str}  {note}")

# ── Part 4: Save best final candidates ───────────────────────────────────────
print("\n" + "="*70)
print("Part 4: Save final submissions")
print("="*70)

sample = pd.read_csv('sample_submission.csv')

# Best refined rh config
if best_rh_c:
    n, w, tm, um = best_rh_c
    th_o = np.sort(fw4_o)[-n]; th_t = np.sort(fw4_t)[-n]
    m_o = fw4_o >= th_o; m_t = fw4_t >= th_t
    adj_o = fw4_o.copy(); adj_t = fw4_t.copy()
    adj_o[m_o] = (1-w)*fw4_o[m_o] + w*rh_o[m_o]
    adj_t[m_t] = (1-w)*fw4_t[m_t] + w*rh_t[m_t]
    m = mae_fn(adj_o)
    fname = f"submission_4way_tailRH_n{n}_w{int(w*100):02d}_OOF{m:.5f}.csv"
    sample['answer'] = np.clip(adj_t, 0, None)
    sample.to_csv(fname, index=False)
    print(f"Saved: {fname}  test={tm:.3f}  unseen={um:.3f}")

# Best dual config if different
if best_dual_c and best_dual < best_rh - 0.0001:
    n1,w1,n2,w2,tm,um = best_dual_c
    th_o1 = np.sort(fw4_o)[-n1]; th_t1 = np.sort(fw4_t)[-n1]
    m1_o = fw4_o >= th_o1; m1_t = fw4_t >= th_t1
    th_o2 = np.sort(fw4_o)[-n2]; th_t2 = np.sort(fw4_t)[-n2]
    m2_o = fw4_o >= th_o2; m2_t = fw4_t >= th_t2
    base_o = fw4_o.copy(); base_t = fw4_t.copy()
    base_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
    base_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
    base_o[m2_o] = (1-w2)*base_o[m2_o] + w2*slhm_o[m2_o]
    base_t[m2_t] = (1-w2)*base_t[m2_t] + w2*slhm_t[m2_t]
    fname2 = f"submission_4way_dualTG_rh{n1}w{int(w1*100):02d}_slhm{n2}w{int(w2*100):02d}_OOF{best_dual:.5f}.csv"
    sample['answer'] = np.clip(base_t, 0, None)
    sample.to_csv(fname2, index=False)
    print(f"Saved: {fname2}  test={tm:.3f}  unseen={um:.3f}")

print("\n" + "="*70)
print("FINAL: Best achievable OOF configs")
print("="*70)
print(f"4way:      OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}")
print(f"tg_rh:     OOF={mae_fn(tg_rh_o):.5f}  test={tmean(tg_rh_t):.3f}  unseen={umean(tg_rh_t):.3f}")
print(f"tg_slhm:   OOF={mae_fn(tg_slhm_o):.5f}  test={tmean(tg_slhm_t):.3f}  unseen={umean(tg_slhm_t):.3f}")
if best_rh_c: print(f"tg_rh_refined: OOF={best_rh:.5f}  test={tmean(adj_t):.3f}" if best_rh_c else "")
if best_dual_c: print(f"dual_tg:   OOF={best_dual:.5f}  test={tm:.3f}  unseen={um:.3f}")
