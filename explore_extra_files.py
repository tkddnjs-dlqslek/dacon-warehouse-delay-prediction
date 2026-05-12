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

xgb_comb_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgb_comb_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')

print(f'oracle_NEW: OOF={base_oof:.5f}  test={oracle_test.mean():.3f}')
print(f'best_base:  OOF={best_base_v:.5f}  test={best_base_test.mean():.3f}')

# Best combos confirmed
bb_comb_oof = np.clip(0.85*best_base_oof + 0.15*np.clip(xgb_comb_o,0,None), 0, None)
bb_comb_test = np.clip(0.85*best_base_test + 0.15*np.clip(xgb_comb_t,0,None), 0, None)
print(f'bb+xgb_comb w=0.15: OOF={mae(bb_comb_oof):.5f}  test={bb_comb_test.mean():.3f}  unseen={bb_comb_test[unseen_mask].mean():.3f}')

bb_mono_oof = np.clip(0.86*best_base_oof + 0.14*np.clip(mono_o,0,None), 0, None)
bb_mono_test = np.clip(0.86*best_base_test + 0.14*np.clip(mono_t,0,None), 0, None)
print(f'bb+mono w=0.14: OOF={mae(bb_mono_oof):.5f}  test={bb_mono_test.mean():.3f}  unseen={bb_mono_test[unseen_mask].mean():.3f}')

# === Fold-level analysis ===
print('\n=== Fold-level analysis of best combos ===')
val_indices = [(fi, np.sort(vi)) for fi, (_, vi) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups))]

configs = [
    ('oracle_NEW', oracle_oof),
    ('best_base', best_base_oof),
    ('bb+xgb_comb_w15', bb_comb_oof),
    ('bb+mono_w14', bb_mono_oof),
]
print(f'{"Fold":>5}', end='')
for name, _ in configs:
    print(f'  {name[:15]:>15s}', end='')
print()
for fi, vi in val_indices:
    print(f'{fi+1:>5}', end='')
    for cname, pred in configs:
        fv = float(np.mean(np.abs(pred[vi] - y_true[vi])))
        print(f'  {fv:>15.5f}', end='')
    print()
print(f'{"Global":>5}', end='')
for cname, pred in configs:
    print(f'  {mae(pred):>15.5f}', end='')
print()

# === Explore extra files ===
print('\n=== Extra files exploration ===')

def check_arr(name, oof_path, test_path, base_oof_arr=best_base_oof, base_test_arr=best_base_test):
    if not os.path.exists(oof_path) or not os.path.exists(test_path): return None
    oo = np.load(oof_path)
    ot = np.load(test_path)
    if oo.shape[0] != len(train_raw) or ot.shape[0] != len(test_raw): return None
    oof_v = mae(np.clip(oo, 0, None))
    unseen_m = ot[unseen_mask].mean()
    seen_m = ot[seen_mask].mean()
    results = []
    for w in [0.05, 0.10, 0.15]:
        b_oof = np.clip((1-w)*base_oof_arr + w*np.clip(oo,0,None), 0, None)
        b_test = np.clip((1-w)*base_test_arr + w*np.clip(ot,0,None), 0, None)
        marker = '*' if mae(b_oof) < base_oof else ''
        results.append((mae(b_oof), w, b_test.mean(), b_test[unseen_mask].mean(), marker))
    best = min(results, key=lambda x: x[0])
    print(f'{name}: standalone_OOF={oof_v:.5f}  unseen={unseen_m:.3f}  seen={seen_m:.3f}')
    print(f'  best_blend w={best[1]:.2f}: OOF={best[0]:.5f} ({best[0]-base_oof:+.6f})  test={best[2]:.3f}  unseen_blend={best[3]:.3f} {best[4]}')
    return (best[0], name, best[1], oo, ot)

# base_v31 files
print('\n--- base_v31/ ---')
for name in ['cb_v31', 'lgb_v31', 'xgb_v31']:
    check_arr(name, f'results/base_v31/{name}_oof.npy', f'results/base_v31/{name}_test.npy')

# ranking variants
print('\n--- ranking_variants/ ---')
rank_ens_o = np.load('results/ranking_variants/rank_ens_oof.npy')[id2]
rank_ens_t = np.load('results/ranking_variants/rank_ens_test.npy')[te_id2]
print(f'rank_ens: OOF={mae(rank_ens_o):.5f}  test={rank_ens_t.mean():.3f}  unseen={rank_ens_t[unseen_mask].mean():.3f}')
for w in [0.05, 0.10, 0.15]:
    b_oof = np.clip((1-w)*best_base_oof + w*np.clip(rank_ens_o,0,None), 0, None)
    b_test = np.clip((1-w)*best_base_test + w*np.clip(rank_ens_t,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  bb+rank_ens w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f} {marker}')

# Individual ranking variants
print('\n--- individual ranking variants ---')
for vnum in range(6):
    for obj in ['lambdarank', 'rank_xendcg']:
        fn = f'results/ranking_variants/rank_adj_oof_v{vnum}_s42_{obj}.npy' if vnum==0 else f'results/ranking_variants/rank_adj_oof_v{vnum}_{"s42" if vnum in [0,1] else "s123" if vnum in [2,3] else "s2024"}_{obj}.npy'
        # Actually let me just check all oof files in the dir
        break
    break

for fp in sorted(glob.glob('results/ranking_variants/rank_adj_oof_*.npy')):
    fn = os.path.basename(fp)
    oo = np.load(fp)[id2]
    # Find test counterpart
    test_fp = fp.replace('oof', 'test')
    if not os.path.exists(test_fp): continue
    ot = np.load(test_fp)[te_id2]
    oof_v = mae(np.clip(oo,0,None))
    unseen_m = ot[unseen_mask].mean()
    b_oof = np.clip(0.85*best_base_oof + 0.15*np.clip(oo,0,None), 0, None)
    b_test = np.clip(0.85*best_base_test + 0.15*np.clip(ot,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  {fn:50s}: OOF={oof_v:.5f}  unseen={unseen_m:.3f}  blend_OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f}) {marker}')

# residual_ranking
print('\n--- residual_ranking/ ---')
res_rank_o = np.load('results/residual_ranking/res_rank_oof.npy')
res_rank_t = np.load('results/residual_ranking/res_rank_test.npy')
print(f'res_rank: shape oof={res_rank_o.shape}  test={res_rank_t.shape}')
if res_rank_o.shape[0] == len(train_raw):
    print(f'  OOF={mae(np.clip(res_rank_o,0,None)):.5f}  test={res_rank_t.mean():.3f}  unseen={res_rank_t[unseen_mask].mean():.3f}')
elif res_rank_o.shape[0] == len(train_ls):
    res_o = res_rank_o[id2]
    res_t = res_rank_t[te_id2]
    print(f'  OOF={mae(np.clip(res_o,0,None)):.5f}  test={res_t.mean():.3f}  unseen={res_t[unseen_mask].mean():.3f}')
    for w in [0.05, 0.10]:
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(res_o,0,None), 0, None)
        b_test = np.clip((1-w)*best_base_test + w*np.clip(res_t,0,None), 0, None)
        marker = '*' if mae(b_oof) < base_oof else ''
        print(f'  bb+res_rank w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f} {marker}')

# cascade
print('\n--- cascade/ ---')
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
    for w in [0.05, 0.10]:
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(oo,0,None), 0, None)
        b_test = np.clip((1-w)*best_base_test + w*np.clip(ot,0,None), 0, None)
        if mae(b_oof) < base_oof:
            print(f'  * {fn[:35]:35s} w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f}')

# layer2
print('\n--- layer2/ ---')
for fp in sorted(glob.glob('results/layer2/*.npy')):
    fn = os.path.basename(fp)
    arr = np.load(fp)
    if arr.shape[0] == len(train_raw):
        print(f'  {fn}: OOF={mae(np.clip(arr,0,None)):.5f}  mean={arr.mean():.3f}')
    elif arr.shape[0] == len(test_raw):
        print(f'  {fn}: test={arr.mean():.3f}  unseen={arr[unseen_mask].mean():.3f}')

# mega33_v31_final.pkl
print('\n--- mega33_v31_final.pkl ---')
with open('results/mega33_v31_final.pkl','rb') as f: d33v31 = pickle.load(f)
print(f'keys: {list(d33v31.keys())[:5]}')
if 'meta_avg_oof' in d33v31:
    v31_oof = d33v31['meta_avg_oof'][id2]
    v31_test = d33v31['meta_avg_test'][te_id2]
    print(f'v31 avg: OOF={mae(v31_oof):.5f}  test={v31_test.mean():.3f}  unseen={v31_test[unseen_mask].mean():.3f}')
    for w in [0.05, 0.10, 0.15, 0.20, 0.25]:
        b_oof = np.clip((1-w)*best_base_oof + w*np.clip(v31_oof,0,None), 0, None)
        b_test = np.clip((1-w)*best_base_test + w*np.clip(v31_test,0,None), 0, None)
        marker = '*' if mae(b_oof) < base_oof else ''
        print(f'  bb+v31 w={w:.2f}: OOF={mae(b_oof):.5f} ({mae(b_oof)-base_oof:+.6f})  test={b_test.mean():.3f}  unseen={b_test[unseen_mask].mean():.3f} {marker}')

# Summary of top candidates (fold-level)
print('\n=== Summary of best OOF candidates ===')
best_candidates = [
    ('oracle_NEW', oracle_oof, oracle_test),
    ('best_base', best_base_oof, best_base_test),
    ('bb+xgb_comb_w15', bb_comb_oof, bb_comb_test),
    ('bb+mono_w14', bb_mono_oof, bb_mono_test),
]
print(f'{"config":30s}  {"OOF":>9}  {"test":>8}  {"unseen":>9}  {"seen":>9}')
for name, oo, ot in best_candidates:
    print(f'  {name:30s}  {mae(oo):>9.5f}  {ot.mean():>8.3f}  {ot[unseen_mask].mean():>9.3f}  {ot[seen_mask].mean():>9.3f}')

print('\nDone.')
