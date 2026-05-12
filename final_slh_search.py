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
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t   = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t   = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgb_comb_o_raw = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgb_comb_t_raw = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o_raw = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t_raw = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
# oracle_seq files are already in train_raw/test_raw order (no re-indexing needed)
xgb_comb_o = xgb_comb_o_raw
xgb_comb_t = xgb_comb_t_raw
mono_o = mono_o_raw
mono_t = mono_t_raw

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3; w2 = fw['iter_r2'] + dr2; w3 = fw['iter_r3'] + dr3
    fx  = wm*mega + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof, 0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

oracle_oof, oracle_test = make_pred(0.0, 0.0, 0.0, 0.64, 0.0)
best_base_oof, best_base_test = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)
base_oof = mae(oracle_oof)
best_base_v = mae(best_base_oof)
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts)
seen_mask = test_raw['layout_id'].isin(train_layouts)
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
val_indices = [(fi, np.sort(vi)) for fi, (_, vi) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups))]

slh_o_raw = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')
slh_t_raw = np.load('results/cascade/spec_lgb_w30_huber_test.npy')
slh_o = slh_o_raw[id2] if slh_o_raw.shape[0] == len(train_ls) else slh_o_raw
slh_t = slh_t_raw[te_id2] if slh_t_raw.shape[0] == len(test_ls) else slh_t_raw

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')
print(f'slh: OOF={mae(np.clip(slh_o,0,None)):.5f}  test={slh_t.mean():.3f}  unseen={slh_t[unseen_mask].mean():.3f}')

# === Part 1: Exhaustive three-way search ===
print('\n=== Fine-grained three-way search: bb + slh + xgb_comb ===')
results_3way = []
for w_slh in np.arange(0.04, 0.16, 0.01):
    for w_xgb in np.arange(0.04, 0.16, 0.01):
        w_bb = 1.0 - w_slh - w_xgb
        if w_bb < 0.70 or w_bb > 0.95: continue
        b_oof = np.clip(w_bb*best_base_oof + w_slh*np.clip(slh_o,0,None) + w_xgb*np.clip(xgb_comb_o,0,None), 0, None)
        b_test = np.clip(w_bb*best_base_test + w_slh*np.clip(slh_t,0,None) + w_xgb*np.clip(xgb_comb_t,0,None), 0, None)
        results_3way.append((mae(b_oof), w_bb, w_slh, w_xgb, b_test.mean(), b_test[unseen_mask].mean(), b_oof, b_test))

results_3way.sort()
print(f'Top 10 three-way (bb+slh+xgb):')
print(f'{"w_bb":>7}  {"w_slh":>7}  {"w_xgb":>7}  {"OOF":>9}  {"ΔOOF":>8}  {"test":>8}  {"unseen":>9}')
for oof_v, w_bb, w_slh, w_xgb, tm, tu, _, _ in results_3way[:10]:
    print(f'  {w_bb:.2f}     {w_slh:.2f}     {w_xgb:.2f}  {oof_v:.5f}  {oof_v-base_oof:+.6f}  {tm:.3f}  {tu:.3f}')

print('\n=== Fine-grained three-way search: bb + slh + mono ===')
results_3way_m = []
for w_slh in np.arange(0.04, 0.16, 0.01):
    for w_mono in np.arange(0.04, 0.16, 0.01):
        w_bb = 1.0 - w_slh - w_mono
        if w_bb < 0.70 or w_bb > 0.95: continue
        b_oof = np.clip(w_bb*best_base_oof + w_slh*np.clip(slh_o,0,None) + w_mono*np.clip(mono_o,0,None), 0, None)
        b_test = np.clip(w_bb*best_base_test + w_slh*np.clip(slh_t,0,None) + w_mono*np.clip(mono_t,0,None), 0, None)
        results_3way_m.append((mae(b_oof), w_bb, w_slh, w_mono, b_test.mean(), b_test[unseen_mask].mean(), b_oof, b_test))

results_3way_m.sort()
print(f'Top 10 three-way (bb+slh+mono):')
print(f'{"w_bb":>7}  {"w_slh":>7}  {"w_mono":>7}  {"OOF":>9}  {"ΔOOF":>8}  {"test":>8}  {"unseen":>9}')
for oof_v, w_bb, w_slh, w_mono, tm, tu, _, _ in results_3way_m[:10]:
    print(f'  {w_bb:.2f}     {w_slh:.2f}     {w_mono:.2f}  {oof_v:.5f}  {oof_v-base_oof:+.6f}  {tm:.3f}  {tu:.3f}')

# === Part 2: Four-way blend ===
print('\n=== Four-way blend: bb + slh + xgb_comb + mono ===')
results_4way = []
for w_slh in [0.05, 0.08, 0.10]:
    for w_xgb in [0.05, 0.08, 0.10]:
        for w_mono in [0.05, 0.08, 0.10]:
            w_bb = 1.0 - w_slh - w_xgb - w_mono
            if w_bb < 0.65 or w_bb > 0.92: continue
            b_oof = np.clip(w_bb*best_base_oof + w_slh*np.clip(slh_o,0,None) + w_xgb*np.clip(xgb_comb_o,0,None) + w_mono*np.clip(mono_o,0,None), 0, None)
            b_test = np.clip(w_bb*best_base_test + w_slh*np.clip(slh_t,0,None) + w_xgb*np.clip(xgb_comb_t,0,None) + w_mono*np.clip(mono_t,0,None), 0, None)
            results_4way.append((mae(b_oof), w_bb, w_slh, w_xgb, w_mono, b_test.mean(), b_test[unseen_mask].mean(), b_oof, b_test))

results_4way.sort()
print(f'Top 10 four-way:')
for oof_v, w_bb, w_slh, w_xgb, w_mono, tm, tu, _, _ in results_4way[:10]:
    print(f'  bb={w_bb:.2f}+slh={w_slh:.2f}+xgb={w_xgb:.2f}+mono={w_mono:.2f}: OOF={oof_v:.5f} ({oof_v-base_oof:+.6f})  test={tm:.3f}  unseen={tu:.3f}')

# === Part 3: Fold-level analysis for best configs ===
print('\n=== Fold-level analysis for best three-way configs ===')
best_3xgb = results_3way[0]
best_3mono = results_3way_m[0]
best_4way = results_4way[0]

configs_check = [
    ('oracle_NEW', oracle_oof, oracle_test),
    ('best_base', best_base_oof, best_base_test),
    (f'bb+slh_w0.08', np.clip(0.92*best_base_oof + 0.08*np.clip(slh_o,0,None), 0, None),
     np.clip(0.92*best_base_test + 0.08*np.clip(slh_t,0,None), 0, None)),
    (f'3way_bb+slh+xgb_best', best_3xgb[6], best_3xgb[7]),
    (f'3way_bb+slh+mono_best', best_3mono[6], best_3mono[7]),
    (f'4way_best', best_4way[7], best_4way[8]),
]
print(f'{"Fold":>5}', end='')
for name, _, _ in configs_check:
    print(f'  {name[:18]:>18s}', end='')
print()
for fi, vi in val_indices:
    print(f'{fi+1:>5}', end='')
    for cname, pred, _ in configs_check:
        fv = float(np.mean(np.abs(pred[vi]-y_true[vi])))
        base_fv = float(np.mean(np.abs(np.clip(oracle_oof[vi],0,None)-y_true[vi])))
        print(f'  {fv:>13.5f}({fv-base_fv:>+7.5f})', end='')
    print()
print(f'{"Global":>5}', end='')
for cname, pred, _ in configs_check:
    print(f'  {mae(pred):>13.5f}({mae(pred)-base_oof:>+7.5f})', end='')
print()

# === Part 4: Save final submissions ===
print('\n=== Saving final submissions ===')
sub = pd.read_csv('sample_submission.csv')

saved_list = []

# 1. bb+slh w=0.08 (best two-way)
w=0.08
b_oof = np.clip((1-w)*best_base_oof + w*np.clip(slh_o,0,None), 0, None)
b_test = np.clip((1-w)*best_base_test + w*np.clip(slh_t,0,None), 0, None)
sub['avg_delay_minutes_next_30m'] = b_test
fname = f'submission_bb_slh_w{w:.2f}_OOF{mae(b_oof):.5f}.csv'
sub.to_csv(fname, index=False)
saved_list.append((fname, mae(b_oof), b_test.mean()))
print(f'Saved: {fname}  OOF={mae(b_oof):.5f}  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

# 2. Best three-way bb+slh+xgb
oof_v, w_bb, w_slh, w_xgb, tm, tu, b_oof3, b_test3 = results_3way[0]
sub['avg_delay_minutes_next_30m'] = b_test3
fname = f'submission_3way_bb{w_bb:.2f}_slh{w_slh:.2f}_xgb{w_xgb:.2f}_OOF{oof_v:.5f}.csv'
sub.to_csv(fname, index=False)
saved_list.append((fname, oof_v, tm))
print(f'Saved: {fname}  OOF={oof_v:.5f}  test={tm:.3f}  unseen={tu:.3f}')

# 3. Best three-way bb+slh+mono
oof_v, w_bb, w_slh, w_mono, tm, tu, b_oof3m, b_test3m = results_3way_m[0]
sub['avg_delay_minutes_next_30m'] = b_test3m
fname = f'submission_3way_bb{w_bb:.2f}_slh{w_slh:.2f}_mono{w_mono:.2f}_OOF{oof_v:.5f}.csv'
sub.to_csv(fname, index=False)
saved_list.append((fname, oof_v, tm))
print(f'Saved: {fname}  OOF={oof_v:.5f}  test={tm:.3f}  unseen={tu:.3f}')

# 4. Best four-way
oof_v, w_bb, w_slh, w_xgb, w_mono, tm, tu, b_oof4, b_test4 = results_4way[0]
sub['avg_delay_minutes_next_30m'] = b_test4
fname = f'submission_4way_bb{w_bb:.2f}_slh{w_slh:.2f}_xgb{w_xgb:.2f}_mono{w_mono:.2f}_OOF{oof_v:.5f}.csv'
sub.to_csv(fname, index=False)
saved_list.append((fname, oof_v, tm))
print(f'Saved: {fname}  OOF={oof_v:.5f}  test={tm:.3f}  unseen={tu:.3f}')

# 5. Also save oracle+slh w=0.08 (oracle base + slh)
w=0.08
b_oof_o = np.clip((1-w)*oracle_oof + w*np.clip(slh_o,0,None), 0, None)
b_test_o = np.clip((1-w)*oracle_test + w*np.clip(slh_t,0,None), 0, None)
sub['avg_delay_minutes_next_30m'] = b_test_o
fname = f'submission_oracle_slh_w{w:.2f}_OOF{mae(b_oof_o):.5f}.csv'
sub.to_csv(fname, index=False)
saved_list.append((fname, mae(b_oof_o), b_test_o.mean()))
print(f'Saved: {fname}  OOF={mae(b_oof_o):.5f}  test={b_test_o.mean():.3f}  unseen={b_test_o[unseen_mask].mean():.3f}')

# === Part 5: Summary ===
print('\n=== FINAL SUBMISSION RANKING ===')
print(f'{"filename":65s}  {"OOF":>9}  {"ΔOOF":>8}  {"test":>8}')
print(f'  {"oracle_NEW (LB=9.7527 BEST)":65s}  {base_oof:.5f}  +0.000000  {oracle_test.mean():.3f}')
for fname, oof_v, tm in sorted(saved_list, key=lambda x: x[1]):
    print(f'  {fname:65s}  {oof_v:.5f}  {oof_v-base_oof:+.6f}  {tm:.3f}')

print('\nDone.')
