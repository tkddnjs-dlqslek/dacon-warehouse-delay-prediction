"""
Deep analysis of tail-gate raw_huber strategy.
Goal: verify the OOF gain is real (not overfit), check test behavior, refine parameters.
Best so far: top_n=2000, w=0.20: OOF=8.37205, test=19.863, unseen=23.396
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
    mega_o = (1-w34)*mega33_oof + w34*mega34_oof
    mega_tt = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33']-dr2-dr3; w2 = fw['iter_r2']+dr2; w3 = fw['iter_r3']+dr3
    fx  = wm*mega_o  + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_tt + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    wr = 1.0-wf; wxgb=0.12*wr/0.36; wlv2=0.16*wr/0.36; wrem=0.08*wr/0.36
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
seen_mask     = ~unseen_mask
umean   = lambda t: float(np.mean(t[unseen_mask]))
smean   = lambda t: float(np.mean(t[seen_mask]))

slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
fw4_o = 0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o
fw4_t = 0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t
fw4_mae = mae_fn(fw4_o)

rh_o_raw = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw = np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o = np.array(rh_o_raw)[id2] if rh_o_raw.shape[0] == len(train_ls) else np.array(rh_o_raw)
rh_t = np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0] == len(test_ls) else np.array(rh_t_raw)

print(f"4way: OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}  seen={smean(fw4_t):.3f}")
print(f"rh: mean_train={rh_o.mean():.2f}  mean_test={rh_t.mean():.2f}  unseen={umean(rh_t):.3f}  seen={smean(rh_t):.3f}")

# ── Part A: Where does top-N gate fire? ──────────────────────────────────────
print("\n" + "="*70)
print("Part A: Anatomy of top-N gate on training and test")
print("="*70)

for top_n in [250, 500, 1000, 2000, 2500, 5000]:
    # Training
    thresh_o = np.sort(fw4_o)[-top_n]
    mask_h = fw4_o >= thresh_o
    n_gated = mask_h.sum()
    y_in_gate = y_true[mask_h]
    mean_y_gated = y_in_gate.mean()
    p95_y = np.percentile(y_true, 95)

    # Test
    thresh_t = np.sort(fw4_t)[-top_n]
    mask_t = fw4_t >= thresh_t
    n_unseen_gated = unseen_mask[mask_t].sum()
    pct_unseen = n_unseen_gated / mask_t.sum() * 100

    fw4_in_gate = fw4_o[mask_h].mean()
    rh_in_gate  = rh_o[mask_h].mean()
    y_above_p95 = (y_in_gate > p95_y).sum()

    print(f"  top_n={top_n:5d}: fw4_thresh={thresh_o:.1f}  mean_y={mean_y_gated:.1f}  y>p95={y_above_p95}  rh_mean={rh_in_gate:.1f}  test_unseen_pct={pct_unseen:.1f}%  fw4_t_thresh={thresh_t:.1f}")

# ── Part B: Fold-level analysis of tail gate ─────────────────────────────────
print("\n" + "="*70)
print("Part B: Per-fold OOF improvement from tail gate (top_n=2000, w=0.20)")
print("="*70)

from sklearn.model_selection import GroupKFold
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

thresh_o = np.sort(fw4_o)[-2000]
mask_h = fw4_o >= thresh_o
adj_o = fw4_o.copy()
adj_o[mask_h] = (1-0.20)*fw4_o[mask_h] + 0.20*rh_o[mask_h]

print(f"Overall: fw4={fw4_mae:.5f}  tail_gate={mae_fn(adj_o):.5f}  delta={mae_fn(adj_o)-fw4_mae:+.6f}")
for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_raw, groups=groups)):
    fw4_fold   = np.mean(np.abs(np.clip(fw4_o[va_idx],0,None) - y_true[va_idx]))
    adj_fold   = np.mean(np.abs(np.clip(adj_o[va_idx],0,None) - y_true[va_idx]))
    n_gated_fold = mask_h[va_idx].sum()
    print(f"  Fold {fold_idx}: fw4={fw4_fold:.5f}  adj={adj_fold:.5f}  delta={adj_fold-fw4_fold:+.6f}  n_gated={n_gated_fold}")

# ── Part C: Fine search for optimal top_n and w ──────────────────────────────
print("\n" + "="*70)
print("Part C: Grid search top_n × w for tail gate")
print("="*70)

print(f"{'top_n':>6}  {'w':>5}  {'OOF':>9}  {'delta':>9}  {'test':>7}  {'unseen':>7}")
results = []
for top_n in [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]:
    thresh_o = np.sort(fw4_o)[-top_n]
    thresh_t = np.sort(fw4_t)[-top_n]
    mask_h = fw4_o >= thresh_o
    mask_t = fw4_t >= thresh_t
    for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        adj_o = fw4_o.copy()
        adj_t = fw4_t.copy()
        adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]
        adj_t[mask_t] = (1-w)*fw4_t[mask_t] + w*rh_t[mask_t]
        m = mae_fn(adj_o)
        delta = m - fw4_mae
        tm = tmean(adj_t)
        um = umean(adj_t)
        results.append((m, top_n, w, delta, tm, um))
        if delta < -0.002:
            print(f"{top_n:6d}  {w:5.2f}  {m:9.5f}  {delta:+9.6f}  {tm:7.3f}  {um:7.3f}")

print("\nTop 10 by OOF:")
results.sort()
for m, top_n, w, delta, tm, um in results[:10]:
    print(f"  top_n={top_n:5d}, w={w:.2f}: OOF={m:.5f} ({delta:+.6f})  test={tm:.3f}  unseen={um:.3f}")

# ── Part D: Compare with fold validation ─────────────────────────────────────
print("\n" + "="*70)
print("Part D: Verify top candidate with per-fold stability")
print("="*70)

# Take top 3 OOF performers
for m, top_n, w, delta, tm, um in results[:3]:
    thresh_o = np.sort(fw4_o)[-top_n]
    mask_h = fw4_o >= thresh_o
    adj_o = fw4_o.copy()
    adj_o[mask_h] = (1-w)*fw4_o[mask_h] + w*rh_o[mask_h]

    fold_deltas = []
    for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_raw, groups=groups)):
        fw4_fold = np.mean(np.abs(np.clip(fw4_o[va_idx],0,None) - y_true[va_idx]))
        adj_fold = np.mean(np.abs(np.clip(adj_o[va_idx],0,None) - y_true[va_idx]))
        fold_deltas.append(adj_fold - fw4_fold)

    n_neg = sum(1 for d in fold_deltas if d < 0)
    print(f"\ntop_n={top_n}, w={w:.2f}: OOF={m:.5f} ({delta:+.6f})")
    print(f"  Fold deltas: {['%+.4f'%d for d in fold_deltas]}  n_improved={n_neg}/5")

# ── Part E: Save best configurations ─────────────────────────────────────────
print("\n" + "="*70)
print("Part E: Save best tail-gate submissions")
print("="*70)

sample = pd.read_csv('sample_submission.csv')

# Best OOF
best_m, best_top_n, best_w, best_delta, best_tm, best_um = results[0]
thresh_o = np.sort(fw4_o)[-best_top_n]
thresh_t = np.sort(fw4_t)[-best_top_n]
mask_h = fw4_o >= thresh_o
mask_t = fw4_t >= thresh_t
final_t = fw4_t.copy()
final_t[mask_t] = (1-best_w)*fw4_t[mask_t] + best_w*rh_t[mask_t]
sample['answer'] = np.clip(final_t, 0, None)
fname = f"submission_tail_rh_top{best_top_n}_w{int(best_w*100):02d}_OOF{best_m:.5f}.csv"
sample.to_csv(fname, index=False)
print(f"Saved best OOF: {fname}  test={best_tm:.3f}  unseen={best_um:.3f}")

# Best OOF with test_mean close to 4-way baseline
# Find best OOF where test_mean >= 19.4
constrained = [(m,top_n,w,delta,tm,um) for m,top_n,w,delta,tm,um in results if tm >= 19.4 and delta < 0]
if constrained:
    cm, ctop, cw, cdelta, ctm, cum = constrained[0]
    thresh_o2 = np.sort(fw4_o)[-ctop]
    thresh_t2 = np.sort(fw4_t)[-ctop]
    mask_h2 = fw4_o >= thresh_o2
    mask_t2 = fw4_t >= thresh_t2
    final_t2 = fw4_t.copy()
    final_t2[mask_t2] = (1-cw)*fw4_t[mask_t2] + cw*rh_t[mask_t2]
    sample['answer'] = np.clip(final_t2, 0, None)
    fname2 = f"submission_tail_rh_top{ctop}_w{int(cw*100):02d}_constrained_OOF{cm:.5f}.csv"
    sample.to_csv(fname2, index=False)
    print(f"Saved constrained (test>=19.4): {fname2}  test={ctm:.3f}  unseen={cum:.3f}")

# bb_slh baseline + tail raw_huber (to verify chain effect)
bb_slh_o = 0.92*bb_o + 0.08*slh_o
bb_slh_t = 0.92*bb_t + 0.08*slh_t
thresh_bs = np.sort(bb_slh_o)[-best_top_n]
mask_bs   = bb_slh_o >= thresh_bs
thresh_bt = np.sort(bb_slh_t)[-best_top_n]
mask_bt   = bb_slh_t >= thresh_bt
bs_adj_o  = bb_slh_o.copy(); bs_adj_o[mask_bs] = (1-best_w)*bb_slh_o[mask_bs] + best_w*rh_o[mask_bs]
bs_adj_t  = bb_slh_t.copy(); bs_adj_t[mask_bt] = (1-best_w)*bb_slh_t[mask_bt] + best_w*rh_t[mask_bt]
bs_mae = mae_fn(bs_adj_o)
print(f"bb+slh+tailRH(same params): OOF={bs_mae:.5f} ({bs_mae-mae_fn(bb_slh_o):+.6f})  test={tmean(bs_adj_t):.3f}  unseen={umean(bs_adj_t):.3f}")

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print(f"4way:         OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"Best tailgate: OOF={best_m:.5f}  test={best_tm:.3f}  unseen={best_um:.3f}")
if constrained:
    print(f"Constrained:   OOF={cm:.5f}  test={ctm:.3f}  unseen={cum:.3f}")
