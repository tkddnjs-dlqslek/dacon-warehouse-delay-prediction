"""
Exhaustive dual tail gate refinement.
Best so far: rh(n=2000,w=0.15)+slhm(n=4000,w=0.12): OOF=8.37058
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
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
def make_pred(w34=0.0,dr2=0.0,dr3=0.0,wf=0.64,w_cb=0.0):
    mo=(1-w34)*mega33_oof+w34*mega34_oof; mt=(1-w34)*mega33_test+w34*mega34_test
    wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
    fx=wm*mo+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
    fxt=wm*mt+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test
    wr=1-wf; wxgb=0.12*wr/0.36; wlv2=0.16*wr/0.36; wrem=0.08*wr/0.36
    oo=np.clip(wf*fx+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
    ot=np.clip(wf*fxt+wxgb*xgb_t+wlv2*lv2_t+wrem*rem_t,0,None)
    if w_cb>0:
        oo=np.clip((1-w_cb)*oo+w_cb*cb_oof,0,None)
        ot=np.clip((1-w_cb)*ot+w_cb*cb_test,0,None)
    return oo, ot

bb_o, bb_t = make_pred(0.25,-0.04,-0.02,0.72,0.12)
mae_fn=lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
tmean=lambda t: float(np.mean(t))
train_layouts=set(train_raw['layout_id'].unique())
unseen_mask_t=~test_raw['layout_id'].isin(train_layouts).values
umean=lambda t: float(np.mean(t[unseen_mask_t]))

slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
fw4_o=0.74*bb_o+0.08*slh_o+0.10*xgbc_o+0.08*mono_o
fw4_t=0.74*bb_t+0.08*slh_t+0.10*xgbc_t+0.08*mono_t
fw4_mae=mae_fn(fw4_o)

rh_o_raw=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')
rh_t_raw=np.load('results/cascade/spec_lgb_raw_huber_test.npy')
rh_o=np.array(rh_o_raw)[id2] if rh_o_raw.shape[0]==len(train_ls) else np.array(rh_o_raw)
rh_t=np.array(rh_t_raw)[te_id2] if rh_t_raw.shape[0]==len(test_ls) else np.array(rh_t_raw)

slhm_o_raw=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')
slhm_t_raw=np.load('results/cascade/spec_lgb_w30_mae_test.npy')
slhm_o=np.array(slhm_o_raw)[id2] if slhm_o_raw.shape[0]==len(train_ls) else np.array(slhm_o_raw)
slhm_t=np.array(slhm_t_raw)[te_id2] if slhm_t_raw.shape[0]==len(test_ls) else np.array(slhm_t_raw)

print(f"4way: OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")

# ── Part 1: Exhaustive dual gate search ──────────────────────────────────────
print("\n" + "="*70)
print("Part 1: Exhaustive dual gate search")
print("="*70)

all_results = []
n1_vals = list(range(1200, 3200, 200))
w1_vals = [0.10, 0.12, 0.15, 0.18, 0.20, 0.25]
n2_vals = list(range(2000, 6000, 500))
w2_vals = [0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20]

# Pre-sort fw4_o to avoid repeated sorting
sorted_fw4_o = np.sort(fw4_o)
sorted_fw4_t = np.sort(fw4_t)

for n1 in n1_vals:
    th1_o = sorted_fw4_o[-n1]; th1_t = sorted_fw4_t[-n1]
    m1_o = fw4_o >= th1_o; m1_t = fw4_t >= th1_t
    for w1 in w1_vals:
        base_o = fw4_o.copy(); base_t = fw4_t.copy()
        base_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
        base_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
        m1_mae = mae_fn(base_o)
        for n2 in n2_vals:
            th2_o = sorted_fw4_o[-n2]; th2_t = sorted_fw4_t[-n2]
            m2_o = fw4_o >= th2_o; m2_t = fw4_t >= th2_t
            for w2 in w2_vals:
                dual_o = base_o.copy(); dual_t = base_t.copy()
                dual_o[m2_o] = (1-w2)*base_o[m2_o] + w2*slhm_o[m2_o]
                dual_t[m2_t] = (1-w2)*base_t[m2_t] + w2*slhm_t[m2_t]
                m = mae_fn(dual_o)
                all_results.append((m, n1, w1, n2, w2, tmean(dual_t), umean(dual_t)))

all_results.sort()
print(f"{'OOF':>9}  {'delta':>9}  {'n1':>5}  {'w1':>5}  {'n2':>5}  {'w2':>5}  {'test':>7}  {'unseen':>7}")
for m, n1, w1, n2, w2, tm, um in all_results[:15]:
    print(f"  {m:.5f}  {m-fw4_mae:+.6f}  {n1:5d}  {w1:.2f}  {n2:5d}  {w2:.2f}  {tm:.3f}  {um:.3f}")

# ── Part 2: Fold stability of best dual config ───────────────────────────────
print("\n" + "="*70)
print("Part 2: Fold stability analysis")
print("="*70)

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)

best_m, n1, w1, n2, w2, best_tm, best_um = all_results[0]
th1_o = sorted_fw4_o[-n1]; m1_o = fw4_o >= th1_o
th2_o = sorted_fw4_o[-n2]; m2_o = fw4_o >= th2_o
dual_o = fw4_o.copy()
dual_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
dual_o[m2_o] = (1-w2)*dual_o[m2_o] + w2*slhm_o[m2_o]

print(f"Best dual: n1={n1},w1={w1:.2f},n2={n2},w2={w2:.2f}: OOF={best_m:.5f} ({best_m-fw4_mae:+.6f})")
fold_deltas = []
for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_raw, groups=groups)):
    fw4_fold = np.mean(np.abs(np.clip(fw4_o[va_idx],0,None) - y_true[va_idx]))
    dual_fold = np.mean(np.abs(np.clip(dual_o[va_idx],0,None) - y_true[va_idx]))
    fold_deltas.append(dual_fold - fw4_fold)
    print(f"  Fold {fold_idx}: fw4={fw4_fold:.5f}  dual={dual_fold:.5f}  delta={dual_fold-fw4_fold:+.6f}")
n_improved = sum(1 for d in fold_deltas if d < 0)
print(f"  Folds improved: {n_improved}/5  std_delta={np.std(fold_deltas):.5f}")

# Compare to single rh gate
rh_o_ref = fw4_o.copy()
rh_o_ref[fw4_o >= sorted_fw4_o[-2000]] = 0.80*fw4_o[fw4_o >= sorted_fw4_o[-2000]] + 0.20*rh_o[fw4_o >= sorted_fw4_o[-2000]]
print(f"\nSingle rh (n=2000,w=0.20): OOF={mae_fn(rh_o_ref):.5f}")
fold_deltas2 = []
for fold_idx, (tr_idx, va_idx) in enumerate(gkf.split(train_raw, groups=groups)):
    fw4_fold = np.mean(np.abs(np.clip(fw4_o[va_idx],0,None) - y_true[va_idx]))
    rh_fold = np.mean(np.abs(np.clip(rh_o_ref[va_idx],0,None) - y_true[va_idx]))
    fold_deltas2.append(rh_fold - fw4_fold)
n_improved2 = sum(1 for d in fold_deltas2 if d < 0)
print(f"  Folds improved: {n_improved2}/5  deltas={['%+.4f'%d for d in fold_deltas2]}")

# ── Part 3: Save best dual and top configs ───────────────────────────────────
print("\n" + "="*70)
print("Part 3: Save submissions")
print("="*70)

sample = pd.read_csv('sample_submission.csv')

# Top 3 unique configurations by OOF
saved_configs = set()
saved_count = 0
for m, n1, w1, n2, w2, tm, um in all_results:
    config_key = (n1//200, round(w1,2), n2//500, round(w2,2))
    if config_key in saved_configs: continue
    saved_configs.add(config_key)

    th1_o = sorted_fw4_o[-n1]; th1_t = sorted_fw4_t[-n1]
    th2_o = sorted_fw4_o[-n2]; th2_t = sorted_fw4_t[-n2]
    m1_o2 = fw4_o >= th1_o; m1_t2 = fw4_t >= th1_t
    m2_o2 = fw4_o >= th2_o; m2_t2 = fw4_t >= th2_t
    d_o = fw4_o.copy(); d_t = fw4_t.copy()
    d_o[m1_o2] = (1-w1)*fw4_o[m1_o2] + w1*rh_o[m1_o2]
    d_t[m1_t2] = (1-w1)*fw4_t[m1_t2] + w1*rh_t[m1_t2]
    d_o[m2_o2] = (1-w2)*d_o[m2_o2] + w2*slhm_o[m2_o2]
    d_t[m2_t2] = (1-w2)*d_t[m2_t2] + w2*slhm_t[m2_t2]
    m_check = mae_fn(d_o)
    fname = f"submission_dualTG_rh{n1}w{int(w1*100):02d}_slhm{n2}w{int(w2*100):02d}_OOF{m_check:.5f}.csv"
    sample['answer'] = np.clip(d_t, 0, None)
    sample.to_csv(fname, index=False)
    print(f"Saved: {fname}  test={tmean(d_t):.3f}  unseen={umean(d_t):.3f}")
    saved_count += 1
    if saved_count >= 3: break

print("\n" + "="*70)
print("COMPLETE SUMMARY")
print("="*70)
print(f"oracle_NEW (LB=9.7527): OOF=8.38247  test=19.314")
print(f"4way:                   OOF={fw4_mae:.5f}  test={tmean(fw4_t):.3f}  unseen={umean(fw4_t):.3f}")
print(f"single rh:              OOF={mae_fn(rh_o_ref):.5f}  test=19.863  unseen=23.396")
print(f"dual (best):            OOF={best_m:.5f}  test={best_tm:.3f}  unseen={best_um:.3f}  n1={n1},w1={w1:.2f},n2={n2},w2={w2:.2f}")
