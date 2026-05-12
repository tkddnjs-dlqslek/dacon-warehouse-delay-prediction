import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
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
xgb_comb_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgb_comb_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')

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

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')

# Load all cascade files
print('\n=== All cascade files ===')
cascade_files = []
for fp in sorted(glob.glob('results/cascade/*_oof.npy')):
    fn = os.path.basename(fp)
    test_fp = fp.replace('_oof.', '_test.')
    if not os.path.exists(test_fp): continue
    oo = np.load(fp)
    ot = np.load(test_fp)
    if oo.shape[0] not in [len(train_raw), len(train_ls)]: continue
    if ot.shape[0] not in [len(test_raw), len(test_ls)]: continue
    if oo.shape[0] == len(train_ls): oo = oo[id2]
    if ot.shape[0] == len(test_ls): ot = ot[te_id2]
    oof_v = mae(np.clip(oo,0,None))
    best_w = None; best_blend_oof = base_oof
    for w in np.arange(0.02, 0.30, 0.01):
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(oo,0,None), 0, None)
        if mae(b_oof) < best_blend_oof:
            best_blend_oof = mae(b_oof)
            best_w = w
    if best_w is None: best_w = 0.0
    b_test = np.clip((1-best_w)*best_base_test + best_w*np.clip(ot,0,None), 0, None)
    cascade_files.append((best_blend_oof, fn, oof_v, ot.mean(), ot[unseen_mask].mean(), best_w, oo, ot, b_test))
    print(f'  {fn:40s}: standalone={oof_v:.5f}  t={ot.mean():.3f}  t_u={ot[unseen_mask].mean():.3f}  best_w={best_w:.2f}  blend_OOF={best_blend_oof:.5f} ({best_blend_oof-base_oof:+.6f})')

cascade_files.sort()
print(f'\nTop cascade blends (best OOF):')
for bf, fn, standalone, tm, tu, bw, oo, ot, bt in cascade_files[:5]:
    print(f'  {fn:40s}: blend_OOF={bf:.5f} ({bf-base_oof:+.6f})  w={bw:.2f}  test={bt.mean():.3f}  unseen={bt[unseen_mask].mean():.3f}')

# === Deep dive into spec_lgb_w30_huber ===
print('\n=== spec_lgb_w30_huber deep dive ===')
slh_o = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')
slh_t = np.load('results/cascade/spec_lgb_w30_huber_test.npy')
if slh_o.shape[0] == len(train_ls): slh_o = slh_o[id2]
if slh_t.shape[0] == len(test_ls): slh_t = slh_t[te_id2]
print(f'spec_lgb_w30_huber: OOF={mae(np.clip(slh_o,0,None)):.5f}  test={slh_t.mean():.3f}  unseen={slh_t[unseen_mask].mean():.3f}  seen={slh_t[seen_mask].mean():.3f}')
print(f'  train mean={slh_o.mean():.3f}  test mean={slh_t.mean():.3f}')
print(f'  corr with oracle_oof: {np.corrcoef(oracle_oof, np.clip(slh_o,0,None))[0,1]:.4f}')
print(f'  corr with best_base_oof: {np.corrcoef(best_base_oof, np.clip(slh_o,0,None))[0,1]:.4f}')

# Fold-level breakdown
print(f'\nFold-level for bb+slh blends:')
for w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
    b_oof = np.clip((1-w)*best_base_oof + w*np.clip(slh_o,0,None), 0, None)
    b_test = np.clip((1-w)*best_base_test + w*np.clip(slh_t,0,None), 0, None)
    print(f'  w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}', end='')
    for fi, vi in val_indices:
        fv = float(np.mean(np.abs(b_oof[vi]-y_true[vi])))
        base_fv = float(np.mean(np.abs(np.clip(oracle_oof[vi],0,None)-y_true[vi])))
        print(f'  F{fi+1}:{fv:.5f}({fv-base_fv:+.5f})', end='')
    print()

# === Three-way blends: best_base + slh + (xgb_combined or mono) ===
print('\n=== Three-way: bb + slh + xgb_combined ===')
for w_slh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    for w_xgb in [0.05, 0.08, 0.10, 0.12]:
        w_bb = 1.0 - w_slh - w_xgb
        if w_bb < 0.5: continue
        b_oof = np.clip(w_bb*best_base_oof + w_slh*np.clip(slh_o,0,None) + w_xgb*np.clip(xgb_comb_o,0,None), 0, None)
        b_test = np.clip(w_bb*best_base_test + w_slh*np.clip(slh_t,0,None) + w_xgb*np.clip(xgb_comb_t,0,None), 0, None)
        if mae(b_oof) < base_oof - 0.002:
            marker = '**' if mae(b_oof) < base_oof - 0.003 else '*'
            print(f'  {marker} bb({w_bb:.2f})+slh({w_slh:.2f})+xgb({w_xgb:.2f}): OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

print('\n=== Three-way: bb + slh + mono ===')
for w_slh in [0.05, 0.08, 0.10, 0.12, 0.15]:
    for w_mono in [0.05, 0.08, 0.10, 0.12]:
        w_bb = 1.0 - w_slh - w_mono
        if w_bb < 0.5: continue
        b_oof = np.clip(w_bb*best_base_oof + w_slh*np.clip(slh_o,0,None) + w_mono*np.clip(mono_o,0,None), 0, None)
        b_test = np.clip(w_bb*best_base_test + w_slh*np.clip(slh_t,0,None) + w_mono*np.clip(mono_t,0,None), 0, None)
        if mae(b_oof) < base_oof - 0.003:
            print(f'  bb({w_bb:.2f})+slh({w_slh:.2f})+mono({w_mono:.2f}): OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

# === Other cascade files analysis ===
print('\n=== All cascade files with other oracle_seq components ===')
# Load spec_avg which might be the average
spec_avg_o = np.load('results/cascade/spec_avg_oof.npy')
spec_avg_t = np.load('results/cascade/spec_avg_test.npy')
if spec_avg_o.shape[0] == len(train_ls): spec_avg_o = spec_avg_o[id2]
if spec_avg_t.shape[0] == len(test_ls): spec_avg_t = spec_avg_t[te_id2]
print(f'spec_avg: OOF={mae(np.clip(spec_avg_o,0,None)):.5f}  test={spec_avg_t.mean():.3f}  unseen={spec_avg_t[unseen_mask].mean():.3f}')
for w in [0.05, 0.10, 0.15]:
    b_oof = np.clip((1-w)*best_base_oof + w*np.clip(spec_avg_o,0,None), 0, None)
    b_test = np.clip((1-w)*best_base_test + w*np.clip(spec_avg_t,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  bb+spec_avg w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')

# === oracle+slh blend (using oracle_NEW as base) ===
print('\n=== oracle_NEW + slh blend ===')
best_w_ora = None; best_oof_ora = base_oof
for w in np.arange(0.02, 0.25, 0.01):
    b_oof = np.clip((1-w)*oracle_oof + w*np.clip(slh_o,0,None), 0, None)
    b_test = np.clip((1-w)*oracle_test + w*np.clip(slh_t,0,None), 0, None)
    if mae(b_oof) < best_oof_ora:
        best_oof_ora = mae(b_oof)
        best_w_ora = w

if best_w_ora:
    b_oof_o = np.clip((1-best_w_ora)*oracle_oof + best_w_ora*np.clip(slh_o,0,None), 0, None)
    b_test_o = np.clip((1-best_w_ora)*oracle_test + best_w_ora*np.clip(slh_t,0,None), 0, None)
    print(f'Best: oracle+slh w={best_w_ora:.2f}: OOF={best_oof_ora:.5f} ({best_oof_ora-base_oof:+.6f})  test={b_test_o.mean():.3f}  unseen={b_test_o[unseen_mask].mean():.3f}')
else:
    print('No OOF improvement for oracle+slh')

# Show full grid
for w in [0.05, 0.10, 0.12, 0.15, 0.20]:
    b_oof = np.clip((1-w)*oracle_oof + w*np.clip(slh_o,0,None), 0, None)
    b_test = np.clip((1-w)*oracle_test + w*np.clip(slh_t,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  oracle+slh w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')

# === Full candidate summary ===
print('\n=== FULL CANDIDATE SUMMARY ===')
sub = pd.read_csv('sample_submission.csv')
candidates = []

# 1. oracle_NEW (baseline)
candidates.append(('oracle_NEW', oracle_oof, oracle_test, base_oof))

# 2. best_base
candidates.append(('best_base', best_base_oof, best_base_test, best_base_v))

# 3. bb+slh w=0.10 (OOF=8.37858, test=19.497)
for w in [0.08, 0.10, 0.12, 0.15]:
    b_oof = np.clip((1-w)*best_base_oof + w*np.clip(slh_o,0,None), 0, None)
    b_test = np.clip((1-w)*best_base_test + w*np.clip(slh_t,0,None), 0, None)
    candidates.append((f'bb+slh_w{w:.2f}', b_oof, b_test, mae(b_oof)))

# 4. bb+xgb_comb w=0.15 (OOF=8.38041)
w = 0.15
b_oof = np.clip((1-w)*best_base_oof + w*np.clip(xgb_comb_o,0,None), 0, None)
b_test = np.clip((1-w)*best_base_test + w*np.clip(xgb_comb_t,0,None), 0, None)
candidates.append(('bb+xgb_comb_w15', b_oof, b_test, mae(b_oof)))

# 5. bb+mono w=0.14
w = 0.14
b_oof = np.clip((1-w)*best_base_oof + w*np.clip(mono_o,0,None), 0, None)
b_test = np.clip((1-w)*best_base_test + w*np.clip(mono_t,0,None), 0, None)
candidates.append(('bb+mono_w14', b_oof, b_test, mae(b_oof)))

print(f'{"config":25s}  {"OOF":>9}  {"ΔOOF":>8}  {"test":>8}  {"Δtest":>7}  {"unseen":>9}  {"seen":>9}')
for name, oo, ot, oof_v in candidates:
    d_oof = oof_v - base_oof
    d_test = ot.mean() - oracle_test.mean()
    marker = '**' if d_oof < -0.003 else '*' if d_oof < 0 else ''
    print(f'  {name:25s}  {oof_v:.5f}  {d_oof:+.6f}  {ot.mean():.3f}  {d_test:+.4f}  {ot[unseen_mask].mean():.3f}  {ot[seen_mask].mean():.3f} {marker}')

# Save best new submission
print('\n--- Saving ---')
for name, oo, ot, oof_v in candidates:
    if 'slh' in name and oof_v < base_oof - 0.002:
        sub['avg_delay_minutes_next_30m'] = ot
        fname = f'submission_{name.replace("/","_").replace("+","_")}_OOF{oof_v:.5f}.csv'
        sub.to_csv(fname, index=False)
        print(f'Saved: {fname}  OOF={oof_v:.5f}  test={ot.mean():.3f}')

print('\nDone.')
