import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
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

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true), dtype=int)
for fi, (_, vi) in enumerate(gkf.split(train_raw, y_true, groups)):
    fold_ids[vi] = fi
fold_mae_fn = lambda p: [float(np.mean(np.abs(p[fold_ids==fi] - y_true[fold_ids==fi]))) for fi in range(5)]

print("\n" + "="*70)
print("Part 1: Per-layout bias calibration (seen layouts only)")
print("="*70)

# For each training layout, compute bias = mean(y - pred) over all scenarios
train_raw['pred_dual'] = dual_o
layout_bias = train_raw.groupby('layout_id').apply(
    lambda df: (df['avg_delay_minutes_next_30m'] - df['pred_dual']).mean()
).to_dict()

print(f"Layout bias stats:")
bias_vals = np.array(list(layout_bias.values()))
print(f"  mean={bias_vals.mean():.3f}  std={bias_vals.std():.3f}  min={bias_vals.min():.3f}  max={bias_vals.max():.3f}")
print(f"  % positive (underprediction): {(bias_vals>0).mean()*100:.1f}%")

# Check which test layouts are seen and what their training bias is
seen_test_layouts = [l for l in test_raw['layout_id'].unique() if l in train_layouts]
print(f"\n50 seen test layouts -- training bias stats:")
test_seen_biases = {l: layout_bias[l] for l in seen_test_layouts}
bvals = np.array(list(test_seen_biases.values()))
print(f"  mean={bvals.mean():.3f}  std={bvals.std():.3f}  min={bvals.min():.3f}  max={bvals.max():.3f}")
print(f"  Positive bias (underprediction): {(bvals>0).sum()}/50")
print(f"  |bias| > 2: {(np.abs(bvals)>2).sum()}/50")
print(f"  |bias| > 5: {(np.abs(bvals)>5).sum()}/50")

# Apply per-layout bias correction to seen test predictions
t_calib = dual_t.copy()
test_raw['pred_dual'] = dual_t
test_raw['layout_bias'] = test_raw['layout_id'].map(layout_bias).fillna(0)
t_calib += test_raw['layout_bias'].values  # add bias directly
t_calib = np.clip(t_calib, 0, None)
print(f"\nPure bias correction: test={t_calib.mean():.3f}  seen={t_calib[seen_mask].mean():.3f}  unseen={t_calib[unseen_mask].mean():.3f}")

# But we're ADDING the training residual to test predictions, which is leaking training info
# The right way: use leave-one-layout-out bias estimation
print("\n-- Leave-one-layout-out bias calibration (no leakage) --")
loo_bias = {}
for layout in train_layouts:
    # Compute bias on all OTHER layouts
    other_mask = train_raw['layout_id'] != layout
    other_pred = dual_o[other_mask]
    other_y    = y_true[other_mask]
    # But we want the bias FOR this layout, not others
    # So use the within-layout OOF residual
    this_mask = train_raw['layout_id'] == layout
    this_pred = dual_o[this_mask]
    this_y    = y_true[this_mask]
    loo_bias[layout] = float(np.mean(this_y - this_pred))

# Now the question: does adding loo_bias improve OOF?
# OOF for this layout uses predictions from OTHER folds
# Actually, the per-layout bias IS the loo residual
# Adding loo_bias to oof would make OOF = 0 for each layout (trivially perfect)
# But for test, we're adding training bias to test predictions

# The proper validation: apply bias from all OTHER layouts to each test row
# For OOF validation: leave one layout out, apply bias from other layouts to that layout's OOF rows
# But that's cross-validated bias estimation, which is essentially what the oracle_seq models do

# Instead, let's check if the bias is correlated with any features we can use for unseen layouts
print("\nBias correlation with features:")
feature_cols = ['order_inflow_15m', 'unique_sku_15m', 'avg_items_per_order', 'urgent_order_ratio']
train_raw_layout_stats = train_raw.groupby('layout_id').agg(
    bias=('avg_delay_minutes_next_30m', lambda x: (x - dual_o[x.index]).mean()),
    **{fc: (fc, 'mean') for fc in feature_cols}
).reset_index()
for fc in feature_cols:
    corr = np.corrcoef(train_raw_layout_stats['bias'].values, train_raw_layout_stats[fc].values)[0,1]
    print(f"  bias ~ {fc}: r={corr:.3f}")

print("\n" + "="*70)
print("Part 2: Layout-level ratio calibration with GroupKFold validation")
print("="*70)

# Group-k-fold: for each fold, compute layout bias from the OTHER 4 folds and apply to held-out fold
loo_corrected_o = dual_o.copy()
for fi, (train_idx, val_idx) in enumerate(gkf.split(train_raw, y_true, groups)):
    # layouts in val fold
    val_layouts = set(train_raw.iloc[val_idx]['layout_id'].unique())
    # for each val layout, compute bias from TRAIN folds (same layout)
    for layout in val_layouts:
        train_same_layout = [i for i in train_idx if groups[i] == layout]
        if not train_same_layout: continue
        bias_est = float(np.mean(y_true[train_same_layout] - dual_o[train_same_layout]))
        val_same_layout = [i for i in val_idx if groups[i] == layout]
        loo_corrected_o[val_same_layout] = dual_o[val_same_layout] + bias_est

loo_corrected_o = np.clip(loo_corrected_o, 0, None)
print(f"LOO bias-corrected OOF: {mae_fn(loo_corrected_o):.5f} (dual: {dual_mae:.5f})")

# Blend between uncorrected and bias-corrected
for alpha in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    blended_o = (1-alpha)*dual_o + alpha*loo_corrected_o
    blended_o = np.clip(blended_o, 0, None)
    print(f"  alpha={alpha:.1f}: OOF={mae_fn(blended_o):.5f}  delta={mae_fn(blended_o)-dual_mae:+.6f}")

print("\n" + "="*70)
print("Part 3: Apply LOO bias correction to test (seen layouts only)")
print("="*70)

# Apply per-layout bias to seen test layouts using layout bias from full training
t_bias_corrected = dual_t.copy()
for layout in seen_test_layouts:
    layout_mask_test = test_raw['layout_id'] == layout
    bias = layout_bias.get(layout, 0.0)
    t_bias_corrected[layout_mask_test] += bias

t_bias_corrected = np.clip(t_bias_corrected, 0, None)
print(f"Full bias correction on test: test={t_bias_corrected.mean():.3f}  seen={t_bias_corrected[seen_mask].mean():.3f}  unseen={t_bias_corrected[unseen_mask].mean():.3f}")

# Blended
for alpha in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    t_blend = (1-alpha)*dual_t + alpha*t_bias_corrected
    t_blend = np.clip(t_blend, 0, None)
    print(f"  alpha={alpha:.1f}: test={t_blend.mean():.3f}  seen={t_blend[seen_mask].mean():.3f}  unseen={t_blend[unseen_mask].mean():.3f}")

print("\n" + "="*70)
print("Part 4: spec_cb_w30 triple gate (best fold stability: 4/5)")
print("+ dual gate + save submission")
print("="*70)

spec_cb_w30_o = np.load('results/cascade/spec_cb_w30_oof.npy')[id2]
spec_cb_w30_t = np.load('results/cascade/spec_cb_w30_test.npy')[te_id2]

sfw_d = np.sort(dual_o); sft_d = np.sort(dual_t)
m0_o = dual_o >= sfw_d[-500]; m0_t = dual_t >= sft_d[-500]
n0_best, w0_best = 500, 0.10

triple_cb_o = dual_o.copy()
triple_cb_o[m0_o] = (1-w0_best)*dual_o[m0_o] + w0_best*spec_cb_w30_o[m0_o]
triple_cb_o = np.clip(triple_cb_o, 0, None)
triple_cb_t = dual_t.copy()
triple_cb_t[m0_t] = (1-w0_best)*dual_t[m0_t] + w0_best*spec_cb_w30_t[m0_t]
triple_cb_t = np.clip(triple_cb_t, 0, None)
triple_mae = mae_fn(triple_cb_o)
triple_folds = fold_mae_fn(triple_cb_o)
print(f"triple_cb_w30 (n0=500, w0=0.10):")
print(f"  OOF={triple_mae:.5f} ({triple_mae-dual_mae:+.6f})")
print(f"  test={triple_cb_t.mean():.3f}  seen={triple_cb_t[seen_mask].mean():.3f}  unseen={triple_cb_t[unseen_mask].mean():.3f}")
print(f"  folds={[f'{x:.4f}' for x in triple_folds]}")
print(f"  n_improved={sum(1 for fi,fd in zip(triple_folds,fold_mae_fn(dual_o)) if fi<fd)}/5")

# rh triple gate
rh_triple_o = dual_o.copy()
rh_triple_o[dual_o >= sfw_d[-1000]] = 0.90*dual_o[dual_o >= sfw_d[-1000]] + 0.10*rh_o[dual_o >= sfw_d[-1000]]
rh_triple_o = np.clip(rh_triple_o, 0, None)
rh_triple_t = dual_t.copy()
rh_triple_t[dual_t >= sft_d[-1000]] = 0.90*dual_t[dual_t >= sft_d[-1000]] + 0.10*rh_t[dual_t >= sft_d[-1000]]
rh_triple_t = np.clip(rh_triple_t, 0, None)
rh_triple_mae = mae_fn(rh_triple_o)
rh_triple_folds = fold_mae_fn(rh_triple_o)
print(f"\nrh_triple (n0=1000, w0=0.10):")
print(f"  OOF={rh_triple_mae:.5f} ({rh_triple_mae-dual_mae:+.6f})")
print(f"  test={rh_triple_t.mean():.3f}  seen={rh_triple_t[seen_mask].mean():.3f}  unseen={rh_triple_t[unseen_mask].mean():.3f}")
print(f"  folds={[f'{x:.4f}' for x in rh_triple_folds]}")
print(f"  n_improved={sum(1 for fi,fd in zip(rh_triple_folds,fold_mae_fn(dual_o)) if fi<fd)}/5")

sub_template = pd.read_csv('sample_submission.csv')

sub = sub_template.copy()
sub['avg_delay_minutes_next_30m'] = triple_cb_t
fname = f"submission_tripleTG_cb500w10_OOF{triple_mae:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"\nSaved: {fname}")

sub2 = sub_template.copy()
sub2['avg_delay_minutes_next_30m'] = rh_triple_t
fname2 = f"submission_tripleTG_rh1000w10_OOF{rh_triple_mae:.5f}.csv"
sub2.to_csv(fname2, index=False)
print(f"Saved: {fname2}")

print("\n" + "="*70)
print("FINAL SUMMARY -- All candidate submissions")
print("="*70)
candidates = [
    ("oracle_NEW (LB=9.7527)",  8.38247, 19.314, None),
    ("4way",                    8.37624, 19.413, 22.802),
    ("dual_gate",               dual_mae, dual_t.mean(), dual_t[unseen_mask].mean()),
    ("triple_cb_w30",           triple_mae, triple_cb_t.mean(), triple_cb_t[unseen_mask].mean()),
    ("rh_triple",               rh_triple_mae, rh_triple_t.mean(), rh_triple_t[unseen_mask].mean()),
]
print(f"{'Name':25s}  {'OOF':>9}  {'test':>8}  {'unseen':>8}")
for name, oof, tm, uns in candidates:
    uns_str = f"{uns:8.3f}" if uns is not None else "      -"
    print(f"  {name:25s}  {oof:>9.5f}  {tm:>8.3f}  {uns_str}")
