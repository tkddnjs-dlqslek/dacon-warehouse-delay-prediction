"""
spec_lgb_raw_huber tail-specific blending strategy.
Key insight: raw_huber is dramatically better at extreme y_true (>p95: MAE=31.51 vs bb=64.35)
but terrible overall (solo_OOF=28.99).

Strategy: use raw_huber only for rows where our current prediction is HIGH.
This avoids applying it to rows where it would hurt.
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
    mega_o = (1-w34)*mega33_oof + w34*mega34_oof
    mega_tt = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3; w2 = fw['iter_r2'] + dr2; w3 = fw['iter_r3'] + dr3
    fx  = wm*mega_o  + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_tt + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_o, oracle_t = make_pred(0.0,0.0,0.0,0.64,0.0)
bb_o, bb_t = make_pred(0.25,-0.04,-0.02,0.72,0.12)

mae_fn  = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
tmean   = lambda t: float(np.mean(t))
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts)
umean   = lambda t: float(np.mean(t[unseen_mask]))

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]

fw4_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw = np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o = np.array(rh_o_raw)[id2] if rh_o_raw.shape[0] == len(train_ls) else np.array(rh_o_raw)
rh_t = np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0] == len(test_ls) else np.array(rh_t_raw)

print(f"best_base OOF:  {mae_fn(bb_o):.5f}")
print(f"bb+slh OOF:     {mae_fn(0.92*bb_o+0.08*slh_o):.5f}")
print(f"4way OOF:       {fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"raw_huber solo: {mae_fn(rh_o):.5f}")

# ── Part A: Tail-gated blend ─────────────────────────────────────────────────
# Use predictions as gate: when fw4_pred > threshold, blend in raw_huber
print("\n" + "="*70)
print("Part A: Gate raw_huber by predicted value (fw4 prediction)")
print("="*70)

for thr_pct in [90, 93, 95, 97, 99]:
    thr = np.percentile(fw4_o, thr_pct)
    gate = (fw4_o > thr).astype(float)
    n_gated = gate.sum()
    for w in [0.10, 0.20, 0.30, 0.50]:
        # Only blend raw_huber where prediction is high
        adj_o = fw4_o.copy()
        adj_o[fw4_o > thr] = (1-w)*fw4_o[fw4_o>thr] + w*rh_o[fw4_o>thr]
        m = mae_fn(adj_o)
        delta = m - fw4_mae
        if delta < -0.0001:
            # For test: use fw4_t as gate (we apply to test too)
            thr_t = np.percentile(fw4_t, thr_pct)
            adj_t = fw4_t.copy()
            adj_t[fw4_t > thr_t] = (1-w)*fw4_t[fw4_t>thr_t] + w*rh_t[fw4_t>thr_t]
            print(f"  p{thr_pct} gate (n={int(n_gated):5d}), rh_w={w:.2f}: OOF={m:.5f} ({delta:+.6f})  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

# ── Part B: Smooth gate using sigmoid ────────────────────────────────────────
print("\n" + "="*70)
print("Part B: Smooth sigmoid gate for raw_huber")
print("="*70)

def sigmoid_gate(preds, center, scale, max_w):
    return max_w / (1 + np.exp(-(preds - center) / scale))

for center_pct in [90, 95]:
    center = np.percentile(fw4_o, center_pct)
    center_t = np.percentile(fw4_t, center_pct)
    for scale in [5.0, 10.0, 20.0]:
        for max_w in [0.10, 0.20, 0.30]:
            w_o = sigmoid_gate(fw4_o, center, scale, max_w)
            adj_o = (1-w_o)*fw4_o + w_o*rh_o
            m = mae_fn(adj_o)
            if m < fw4_mae - 0.0001:
                w_t = sigmoid_gate(fw4_t, center_t, scale, max_w)
                adj_t = (1-w_t)*fw4_t + w_t*rh_t
                delta = m - fw4_mae
                print(f"  c=p{center_pct}, scale={scale:.0f}, max_w={max_w:.2f}: OOF={m:.5f} ({delta:+.6f})  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}  mean_gate_w={w_o.mean():.4f}")

# ── Part C: Analyze why raw_huber is so good at extremes ─────────────────────
print("\n" + "="*70)
print("Part C: raw_huber residual distribution at extremes")
print("="*70)

for pct_lo, pct_hi in [(0,50),(50,75),(75,90),(90,95),(95,99),(99,100)]:
    lo = np.percentile(y_true, pct_lo)
    hi = np.percentile(y_true, pct_hi) if pct_hi < 100 else y_true.max()+1
    mask = (y_true >= lo) & (y_true < hi)
    n = mask.sum()
    if n == 0: continue
    r_bb  = np.mean((bb_o  - y_true)[mask])
    r_fw4 = np.mean((fw4_o - y_true)[mask])
    r_slh = np.mean((slh_o - y_true)[mask])
    r_rh  = np.mean((rh_o  - y_true)[mask])
    mae_rh_here = np.mean(np.abs((rh_o - y_true)[mask]))
    mae_fw4_here = np.mean(np.abs((fw4_o - y_true)[mask]))
    print(f"  p{pct_lo:2d}-p{pct_hi:3d} (n={n:5d}): bb_r={r_bb:+.2f}  fw4_r={r_fw4:+.2f}  slh_r={r_slh:+.2f}  rh_r={r_rh:+.2f}  rh_mae={mae_rh_here:.2f}  fw4_mae={mae_fw4_here:.2f}")

# ── Part D: What is raw_huber predicting? ────────────────────────────────────
print("\n" + "="*70)
print("Part D: raw_huber prediction analysis")
print("="*70)

print(f"raw_huber train: mean={rh_o.mean():.2f}  std={rh_o.std():.2f}  min={rh_o.min():.2f}  max={rh_o.max():.2f}")
print(f"raw_huber test:  mean={rh_t.mean():.2f}  std={rh_t.std():.2f}  min={rh_t.min():.2f}  max={rh_t.max():.2f}")
print(f"y_true:          mean={y_true.mean():.2f}  std={y_true.std():.2f}  min={y_true.min():.2f}  max={y_true.max():.2f}")
print(f"fw4 train preds: mean={fw4_o.mean():.2f}  max={fw4_o.max():.2f}")
print(f"fw4 test preds:  mean={fw4_t.mean():.2f}  max={fw4_t.max():.2f}")
print(f"rh_t unseen:     mean={umean(rh_t):.3f}")

# Top extreme y_true rows vs predictions
top_indices = np.argsort(y_true)[-20:]
print("\nTop 20 highest delay rows:")
print(f"  {'y_true':>8}  {'fw4':>8}  {'slh':>8}  {'raw_h':>8}  {'bb':>8}")
for i in top_indices:
    print(f"  {y_true[i]:8.1f}  {fw4_o[i]:8.1f}  {slh_o[i]:8.1f}  {rh_o[i]:8.1f}  {bb_o[i]:8.1f}")

# ── Part E: Hard gate at top-N rows ──────────────────────────────────────────
print("\n" + "="*70)
print("Part E: Hard gate — replace fw4 with rh at highest-predicted rows")
print("="*70)

N_total = len(fw4_o)
for top_n in [100, 250, 500, 1000, 2500, 5000]:
    thresh = np.sort(fw4_o)[-top_n]
    mask_h = fw4_o >= thresh
    for w in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
        adj_o = fw4_o.copy()
        adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]
        m = mae_fn(adj_o)
        delta = m - fw4_mae
        if delta < -0.00005:
            # Apply same to test (gate on fw4_t predicted value)
            thresh_t = np.sort(fw4_t)[-top_n]
            mask_t = fw4_t >= thresh_t
            adj_t = fw4_t.copy()
            adj_t[mask_t] = (1-w)*fw4_t[mask_t] + w*rh_t[mask_t]
            print(f"  top_n={top_n:5d} (p{100*(1-top_n/N_total):.1f}+), rh_w={w:.2f}: OOF={m:.5f} ({delta:+.6f})  test={tmean(adj_t):.3f}  unseen={umean(adj_t):.3f}")

# ── Part F: Best hard gate — save submission ──────────────────────────────────
print("\n" + "="*70)
print("Part F: Best configuration search & save")
print("="*70)

best_mae2 = fw4_mae
best_config = None
for top_n in [50,100,200,500,1000,2000,3000,5000]:
    thresh_o = np.sort(fw4_o)[-top_n]
    mask_h = fw4_o >= thresh_o
    for w in np.arange(0.05, 1.05, 0.05):
        adj_o = fw4_o.copy()
        adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]
        m = mae_fn(adj_o)
        if m < best_mae2:
            best_mae2 = m
            best_config = (top_n, w, thresh_o)

if best_config:
    top_n, w, thresh_o = best_config
    thresh_t = np.sort(fw4_t)[-top_n]
    mask_h = fw4_o >= thresh_o
    mask_t = fw4_t >= thresh_t
    final_o = fw4_o.copy()
    final_t = fw4_t.copy()
    final_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]
    final_t[mask_t] = (1-w)*fw4_t[mask_t] + w*rh_t[mask_t]
    print(f"BEST: top_n={top_n}, rh_w={w:.2f}: OOF={best_mae2:.5f} ({best_mae2-fw4_mae:+.6f})  test={tmean(final_t):.3f}  unseen={umean(final_t):.3f}")

    # Save submission
    sample = pd.read_csv('sample_submission.csv')
    sample['answer'] = np.clip(final_t, 0, None)
    fname = f"submission_tail_rh_top{top_n}_w{int(w*100):02d}_OOF{best_mae2:.5f}.csv"
    sample.to_csv(fname, index=False)
    print(f"Saved: {fname}")
else:
    print("No improvement found with tail-gate raw_huber strategy.")
    print("The 4-way blend remains optimal.")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"4way OOF: {fw4_mae:.5f}  (baseline)")
print(f"Best tail-gate: {best_mae2:.5f}" if best_config else "No improvement")
