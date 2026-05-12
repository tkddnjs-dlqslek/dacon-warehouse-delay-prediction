"""
meta_resid is a high-quality OOF=8.36433 model.
Combined with dual gate (OOF=8.37050), achieves OOF=8.36558 at w=0.20.
Key question: how far can we push OOF? And is this beneficial for LB?
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
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
te_id2   = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
m33_o=d33['meta_avg_oof'][id2]; m33_t=d33['meta_avg_test'][te_id2]
m34_o=d34['meta_avg_oof'][id2]; m34_t=d34['meta_avg_test'][te_id2]
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

fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
mo=(1-0.25)*m33_o+0.25*m34_o; mt=(1-0.25)*m33_t+0.25*m34_t
wm=fw['mega33']+0.04+0.02; w2=fw['iter_r2']-0.04; w3r=fw['iter_r3']-0.02
fx=wm*mo+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3r*r3_oof
fxt=wm*mt+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3r*r3_test
wr=0.28; wxgb=0.12*wr/0.36; wlv2=0.16*wr/0.36; wrem=0.08*wr/0.36
bb_o=np.clip(0.88*(0.72*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o)+0.12*cb_oof,0,None)
bb_t=np.clip(0.88*(0.72*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t)+0.12*cb_test,0,None)
fw4_o=0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o
fw4_t=0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t

mae_fn=lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
tmean=lambda t: float(np.mean(t))
train_layouts=set(train_raw['layout_id'].unique()); unseen_t=~test_raw['layout_id'].isin(train_layouts).values
umean=lambda t: float(np.mean(t[unseen_t])); smean=lambda t: float(np.mean(t[~unseen_t]))

rh_o_raw=np.load('results/cascade/spec_lgb_raw_huber_oof.npy'); rh_t_raw=np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o=np.array(rh_o_raw)[id2] if rh_o_raw.shape[0]==len(train_ls) else np.array(rh_o_raw)
rh_t=np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0]==len(test_ls) else np.array(rh_t_raw)
slhm_o_raw=np.load('results/cascade/spec_lgb_w30_mae_oof.npy'); slhm_t_raw=np.load('results/cascade/spec_lgb_w30_mae_test.npy')
slhm_o=np.array(slhm_o_raw)[id2] if slhm_o_raw.shape[0]==len(train_ls) else np.array(slhm_o_raw)
slhm_t=np.array(slhm_t_raw)[te_id2] if slhm_t_raw.shape[0]==len(test_ls) else np.array(slhm_t_raw)
mr_o=np.load('results/meta/meta_resid_oof.npy'); mr_t=np.load('results/meta/meta_resid_test.npy')
mb_o=np.load('results/meta/meta_blend_oof.npy'); mb_t=np.load('results/meta/meta_blend_test.npy')

# Build dual gate
sfw=np.sort(fw4_o); sft=np.sort(fw4_t)
n1,w1,n2,w2=2000,0.15,5500,0.08
m1_o=fw4_o>=sfw[-n1]; m1_t=fw4_t>=sft[-n1]; m2_o=fw4_o>=sfw[-n2]; m2_t=fw4_t>=sft[-n2]
dual_o=fw4_o.copy(); dual_t=fw4_t.copy()
dual_o[m1_o]=(1-w1)*fw4_o[m1_o]+w1*rh_o[m1_o]; dual_t[m1_t]=(1-w1)*fw4_t[m1_t]+w1*rh_t[m1_t]
dual_o[m2_o]=(1-w2)*dual_o[m2_o]+w2*slhm_o[m2_o]; dual_t[m2_t]=(1-w2)*dual_t[m2_t]+w2*slhm_t[m2_t]
dual_mae=mae_fn(dual_o)

print(f"4way:       OOF={mae_fn(fw4_o):.5f}  test={tmean(fw4_t):.3f}  seen={smean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"dual_gate:  OOF={dual_mae:.5f}  test={tmean(dual_t):.3f}  seen={smean(dual_t):.3f}  unseen={umean(dual_t):.3f}")
print(f"meta_resid: OOF={mae_fn(mr_o):.5f}  test={tmean(mr_t):.3f}  seen={smean(mr_t):.3f}  unseen={umean(mr_t):.3f}")
print(f"meta_blend: OOF={mae_fn(mb_o):.5f}  test={tmean(mb_t):.3f}  seen={smean(mb_t):.3f}  unseen={umean(mb_t):.3f}")

# ── Part 1: Full grid search — dual + meta_resid ──────────────────────────────
print("\n" + "="*70)
print("Part 1: Grid search dual_gate × meta_resid blend")
print("="*70)
results = []
for wmr in np.arange(0.02, 0.50, 0.02):
    bl_o=(1-wmr)*dual_o+wmr*mr_o; bl_t=(1-wmr)*dual_t+wmr*mr_t
    m=mae_fn(bl_o); tm=tmean(bl_t); um=umean(bl_t); sm=smean(bl_t)
    results.append((m, wmr, tm, um, sm))

print(f"{'wmr':>6}  OOF        delta     test    seen    unseen")
for m, wmr, tm, um, sm in results:
    d=m-dual_mae
    print(f"  {wmr:.2f}  {m:.5f}  {d:+.6f}  {tm:.3f}  {sm:.3f}  {um:.3f}")

best_m, best_wmr, best_tm, best_um, best_sm = min(results)
print(f"\nBEST: wmr={best_wmr:.2f}: OOF={best_m:.5f} ({best_m-dual_mae:+.6f})  test={best_tm:.3f}  unseen={best_um:.3f}")

# ── Part 2: Fold stability of best meta combo ─────────────────────────────────
print("\n" + "="*70)
print("Part 2: Fold stability — dual+meta_resid vs dual vs 4way")
print("="*70)

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

# Best meta combo
best_bl_o = (1-best_wmr)*dual_o + best_wmr*mr_o
# moderate meta combo w=0.10 (for comparison)
mod_bl_o = 0.90*dual_o + 0.10*mr_o

for name, pred in [("4way", fw4_o), ("dual", dual_o), (f"dual+mr{best_wmr:.2f}", best_bl_o), ("dual+mr0.10", mod_bl_o)]:
    fold_maes = []
    for tr_idx, va_idx in gkf.split(train_raw, groups=groups):
        fold_maes.append(np.mean(np.abs(np.clip(pred[va_idx],0,None)-y_true[va_idx])))
    print(f"{name:<22}: OOF={mae_fn(pred):.5f}  folds={['%.4f'%m for m in fold_maes]}  std={np.std(fold_maes):.5f}")

# ── Part 3: meta_blend analysis ───────────────────────────────────────────────
print("\n" + "="*70)
print("Part 3: meta_blend analysis")
print("="*70)

print(f"Corr(meta_blend, dual): {np.corrcoef(mb_o, dual_o)[0,1]:.4f}")
results_mb = []
for wmb in np.arange(0.02, 0.40, 0.02):
    bl_o=(1-wmb)*dual_o+wmb*mb_o; bl_t=(1-wmb)*dual_t+wmb*mb_t
    m=mae_fn(bl_o)
    results_mb.append((m, wmb, tmean(bl_t), umean(bl_t)))

best_mb = min(results_mb)
print(f"BEST meta_blend blend: wmb={best_mb[1]:.2f}: OOF={best_mb[0]:.5f} ({best_mb[0]-dual_mae:+.6f})  test={best_mb[2]:.3f}  unseen={best_mb[3]:.3f}")

# ── Part 4: Triple combo — dual + meta_resid + meta_blend ────────────────────
print("\n" + "="*70)
print("Part 4: Triple combo — dual + meta_resid + meta_blend")
print("="*70)

best_triple = dual_mae; best_triple_c = None
for wmr2 in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for wmb2 in [0.02, 0.04, 0.06, 0.08, 0.10]:
        wt = 1 - wmr2 - wmb2
        if wt <= 0: continue
        bl_o = wt*dual_o + wmr2*mr_o + wmb2*mb_o
        bl_t = wt*dual_t + wmr2*mr_t + wmb2*mb_t
        m = mae_fn(bl_o)
        if m < best_triple:
            best_triple = m
            best_triple_c = (wmr2, wmb2, tmean(bl_t), umean(bl_t))

if best_triple_c:
    wmr2, wmb2, tm, um = best_triple_c
    print(f"BEST triple: wmr={wmr2:.2f}, wmb={wmb2:.2f}: OOF={best_triple:.5f} ({best_triple-dual_mae:+.6f})  test={tm:.3f}  unseen={um:.3f}")
else:
    print("No triple combo improves dual")

# ── Part 5: Can we push meta_resid further? ───────────────────────────────────
print("\n" + "="*70)
print("Part 5: Direct meta_resid + tail gate combination")
print("(Use meta_resid AS BASE, apply dual tail gate on top)")
print("="*70)

# meta_resid as base, tail gate on top
sfw_mr = np.sort(mr_o); sft_mr = np.sort(mr_t)
for n1_mr in [1000, 1500, 2000]:
    th1 = sfw_mr[-n1_mr]; th1_t = sft_mr[-n1_mr]
    m1_mr = mr_o >= th1; m1_mt = mr_t >= th1_t
    for w1_mr in [0.10, 0.15, 0.20]:
        base_mr_o = mr_o.copy(); base_mr_t = mr_t.copy()
        base_mr_o[m1_mr] = (1-w1_mr)*mr_o[m1_mr] + w1_mr*rh_o[m1_mr]
        base_mr_t[m1_mt] = (1-w1_mr)*mr_t[m1_mt] + w1_mr*rh_t[m1_mt]
        for n2_mr in [3000, 4000, 5000]:
            th2 = sfw_mr[-n2_mr]; th2_t = sft_mr[-n2_mr]
            m2_mr = mr_o >= th2; m2_mt = mr_t >= th2_t
            for w2_mr in [0.06, 0.08, 0.10, 0.12]:
                bl_o = base_mr_o.copy(); bl_t = base_mr_t.copy()
                bl_o[m2_mr] = (1-w2_mr)*base_mr_o[m2_mr] + w2_mr*slhm_o[m2_mr]
                bl_t[m2_mt] = (1-w2_mr)*base_mr_t[m2_mt] + w2_mr*slhm_t[m2_mt]
                m = mae_fn(bl_o)
                if m < mae_fn(mr_o) - 0.003:
                    print(f"  meta+rh(n={n1_mr},w={w1_mr:.2f})+slhm(n={n2_mr},w={w2_mr:.2f}): OOF={m:.5f} ({m-mae_fn(mr_o):+.6f})  test={tmean(bl_t):.3f}  unseen={umean(bl_t):.3f}")

# ── Part 6: Save best candidates ─────────────────────────────────────────────
print("\n" + "="*70)
print("Part 6: Save new best submissions")
print("="*70)

sample = pd.read_csv('sample_submission.csv')

# Best dual+meta
best_m_dmr, best_w_dmr, best_tm_dmr, best_um_dmr, best_sm_dmr = min(results)
final_o_dmr = (1-best_w_dmr)*dual_o + best_w_dmr*mr_o
final_t_dmr = (1-best_w_dmr)*dual_t + best_w_dmr*mr_t
fname_dmr = f"submission_dualTG_meta{int(best_w_dmr*100):02d}_OOF{best_m_dmr:.5f}.csv"
sample['answer'] = np.clip(final_t_dmr, 0, None)
sample.to_csv(fname_dmr, index=False)
print(f"Saved: {fname_dmr}  test={best_tm_dmr:.3f}  seen={best_sm_dmr:.3f}  unseen={best_um_dmr:.3f}")

# Moderate meta (w=0.10) for more conservative option
mod_o = 0.90*dual_o + 0.10*mr_o; mod_t = 0.90*dual_t + 0.10*mr_t
fname_mod = f"submission_dualTG_meta10_OOF{mae_fn(mod_o):.5f}.csv"
sample['answer'] = np.clip(mod_t, 0, None)
sample.to_csv(fname_mod, index=False)
print(f"Saved: {fname_mod}  OOF={mae_fn(mod_o):.5f}  test={tmean(mod_t):.3f}  unseen={umean(mod_t):.3f}")

# meta_resid standalone (for comparison if not already saved)
if not os.path.exists('submission_meta_resid_OOF8.36433.csv'):
    sample['answer'] = np.clip(mr_t, 0, None)
    sample.to_csv('submission_meta_resid_OOF8.36433.csv', index=False)
    print(f"Saved meta_resid standalone")

print("\n" + "="*70)
print("COMPLETE SUMMARY")
print("="*70)
print(f"oracle_NEW (LB=9.7527): OOF=8.38247  test=19.314")
print(f"4way:                   OOF={mae_fn(fw4_o):.5f}  test={tmean(fw4_t):.3f}")
print(f"dual_gate:              OOF={dual_mae:.5f}  test={tmean(dual_t):.3f}  unseen={umean(dual_t):.3f}")
print(f"best dual+meta:         OOF={best_m_dmr:.5f}  test={best_tm_dmr:.3f}  unseen={best_um_dmr:.3f}")
if best_triple_c:
    wmr2, wmb2, tm, um = best_triple_c
    print(f"triple combo:           OOF={best_triple:.5f}  test={tm:.3f}  unseen={um:.3f}")
