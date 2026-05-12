import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

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
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
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

fixed_rw_oof  = wm_best*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
fixed_rw_test = wm_best*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test
fixed_orig_oof  = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fixed_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test

oracle_oof  = np.clip(0.64*fixed_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)
rw_oof  = np.clip(0.64*fixed_rw_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
rw_test = np.clip(0.64*fixed_rw_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'fixed_rw:   OOF={mae(rw_oof):.5f}  test_mean={rw_test.mean():.3f}')

# 1. Non-linear blends: geometric mean
print('\n=== Geometric mean blend (oracle_NEW, rw) ===')
for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
    geo_oof = np.clip(oracle_oof, 1e-6, None)**w * np.clip(rw_oof, 1e-6, None)**(1-w)
    geo_test = np.clip(oracle_test, 1e-6, None)**w * np.clip(rw_test, 1e-6, None)**(1-w)
    print(f'  w_oracle={w:.1f}: OOF={mae(geo_oof):.5f}  delta={mae(geo_oof)-base_oof:+.6f}  test={geo_test.mean():.3f}')

# 2. Rank-based blend
print('\n=== Rank blend (oracle_NEW, rw) ===')
from scipy.stats import rankdata
for w in [0.3, 0.5, 0.7]:
    r_oracle = rankdata(oracle_oof) / len(oracle_oof)
    r_rw = rankdata(rw_oof) / len(rw_oof)
    r_blend = w*r_oracle + (1-w)*r_rw
    # Map back to predictions via interpolation
    sorted_oracle = np.sort(oracle_oof)
    blend_vals = np.interp(r_blend, np.linspace(0, 1, len(sorted_oracle)), sorted_oracle)
    r_oracle_t = rankdata(oracle_test) / len(oracle_test)
    r_rw_t = rankdata(rw_test) / len(rw_test)
    r_blend_t = w*r_oracle_t + (1-w)*r_rw_t
    sorted_oracle_t = np.sort(oracle_test)
    blend_vals_t = np.interp(r_blend_t, np.linspace(0, 1, len(sorted_oracle_t)), sorted_oracle_t)
    print(f'  w_oracle={w:.1f}: OOF={mae(blend_vals):.5f}  delta={mae(blend_vals)-base_oof:+.6f}  test={blend_vals_t.mean():.3f}')

# 3. Explore full oracle_seq files not yet tried
print('\n=== Checking all oracle_seq files ===')
oracle_dir = 'results/oracle_seq'
all_oof_files = sorted([f for f in os.listdir(oracle_dir) if f.startswith('oof_') and f.endswith('.npy')])
print(f'Total OOF files: {len(all_oof_files)}')

# Load with fixed_rw base
best_blend_oof = base_oof
best_blend_file = None
for f in all_oof_files:
    o_path = os.path.join(oracle_dir, f)
    t_path = os.path.join(oracle_dir, f.replace('oof_','test_').replace('oof_seqC_','test_C_'))
    # Try matching test file
    if not os.path.exists(t_path):
        # Try alternative naming
        stem = f.replace('oof_','').replace('.npy','')
        t_candidates = [x for x in os.listdir(oracle_dir) if stem in x and x.startswith('test_')]
        if not t_candidates:
            continue
        t_path = os.path.join(oracle_dir, t_candidates[0])
    try:
        oo = np.load(o_path)
        ot = np.load(t_path)
        # Try blending with rw at small weight
        for w_new in [0.05, 0.10]:
            blend_oof = np.clip((1-w_new)*rw_oof + w_new*np.clip(oo,0,None), 0, None)
            blend_test = np.clip((1-w_new)*rw_test + w_new*np.clip(ot,0,None), 0, None)
            if mae(blend_oof) < best_blend_oof:
                best_blend_oof = mae(blend_oof)
                best_blend_file = (f, w_new, mae(blend_oof), blend_test.mean())
    except Exception as e:
        pass

if best_blend_file:
    print(f'BEST: {best_blend_file}')
else:
    print('No oracle file improves rw_oof through blending')

# Show all oracle files that get within 0.001 of best when blended at w=0.05
print('\nOracle files giving OOF < baseline-0.0001 when blended at w=0.10 with rw:')
for f in all_oof_files:
    o_path = os.path.join(oracle_dir, f)
    try:
        oo = np.load(o_path)
        w_new = 0.10
        blend_oof = np.clip((1-w_new)*rw_oof + w_new*np.clip(oo,0,None), 0, None)
        if mae(blend_oof) < base_oof - 0.0001:
            print(f'  {f}: OOF={mae(np.clip(oo,0,None)):.5f}  blend_OOF={mae(blend_oof):.5f}  delta={mae(blend_oof)-base_oof:+.6f}')
    except:
        pass

# 4. What about the mega33 pkl's other keys?
print('\n=== mega33_final.pkl keys ===')
print(list(d33.keys()))
for k, v in d33.items():
    if isinstance(v, np.ndarray):
        print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')
    elif isinstance(v, list):
        print(f'  {k}: list len={len(v)}')
    else:
        print(f'  {k}: {type(v).__name__}')

# 5. Check other pkl files
print('\n=== Other pkl files in results/ ===')
for root, dirs, files in os.walk('results'):
    for f in files:
        if f.endswith('.pkl') and f != 'mega33_final.pkl':
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'rb') as fp:
                    d = pickle.load(fp)
                if isinstance(d, dict):
                    print(f'  {fpath}: keys={list(d.keys())[:5]}')
            except:
                print(f'  {fpath}: could not load')

# 6. Pack-corrected + fixed_rw combined
print('\n=== Pack-corrected + fixed_rw combined ===')
# Use the targeted pack correction on the rw base
train_raw['oracle_pred'] = rw_oof
test_raw['oracle_pred'] = rw_test

# WH_217 pack=1.0 reference stats
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
sc217 = wh217.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('oracle_pred','mean'),
).reset_index()
sc217_hi = sc217[sc217['pack_mean'] >= 0.99]
wh217_hi_y = sc217_hi['y_mean'].mean()
wh217_hi_pred = sc217_hi['pred_mean'].mean()
print(f'WH_217 pack=1.0: y_mean={wh217_hi_y:.2f}  pred_mean={wh217_hi_pred:.2f}  residual={wh217_hi_y-wh217_hi_pred:+.2f}')

# Apply correction to 3 extreme test layouts
for correction_scale in [0.3, 0.5, 0.7, 1.0]:
    preds = rw_test.copy()
    for wh_id in ['WH_201', 'WH_246', 'WH_283']:
        df = test_raw[test_raw['layout_id']==wh_id].copy()
        sc_stats = df.groupby('scenario_id')['pack_utilization'].mean()
        n_sc = len(sc_stats)
        n_hi_sc = (sc_stats >= 0.99).sum()
        correction = correction_scale * (n_hi_sc / n_sc) * (wh217_hi_y - wh217_hi_pred)
        mask = test_raw['layout_id'] == wh_id
        preds[mask] = np.clip(rw_test[mask] + correction, 0, None)
    print(f'  scale={correction_scale:.1f}: test_mean={preds.mean():.3f}  delta={preds.mean()-rw_test.mean():+.3f}')

# Best scale=0.5 submission
correction_scale = 0.5
preds = rw_test.copy()
for wh_id in ['WH_201', 'WH_246', 'WH_283']:
    df = test_raw[test_raw['layout_id']==wh_id].copy()
    sc_stats = df.groupby('scenario_id')['pack_utilization'].mean()
    n_sc = len(sc_stats)
    n_hi_sc = (sc_stats >= 0.99).sum()
    correction = correction_scale * (n_hi_sc / n_sc) * (wh217_hi_y - wh217_hi_pred)
    mask = test_raw['layout_id'] == wh_id
    preds[mask] = np.clip(rw_test[mask] + correction, 0, None)
    print(f'  {wh_id}: correction={correction:+.2f}  new_mean={preds[mask].mean():.2f}')

sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = preds
fname = f'submission_rw_pack05_OOF{mae(rw_oof):.5f}.csv'
sub.to_csv(fname, index=False)
print(f'Saved: {fname}  test_mean={preds.mean():.3f}')

print('\nDone.')
