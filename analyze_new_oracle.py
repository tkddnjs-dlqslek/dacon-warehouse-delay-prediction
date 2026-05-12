import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

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

# Load the promising oracle files
oracle_dir = 'results/oracle_seq'
candidates = {
    'xgb_combined': ('oof_seqC_xgb_combined.npy', 'test_C_xgb_combined.npy'),
    'xgb_v31':      ('oof_seqC_xgb_v31.npy',      'test_C_xgb_v31.npy'),
    'xgb_bestproxy':('oof_seqC_xgb_bestproxy.npy', 'test_C_xgb_bestproxy.npy'),
    'xgb_monotone': ('oof_seqC_xgb_monotone.npy',  'test_C_xgb_monotone.npy'),
    'cb':           ('oof_seqC_cb.npy',             'test_C_cb.npy'),
}

loaded = {}
for name, (of, tf) in candidates.items():
    op = os.path.join(oracle_dir, of)
    tp = os.path.join(oracle_dir, tf)
    if os.path.exists(op) and os.path.exists(tp):
        loaded[name] = (np.load(op), np.load(tp))
    elif os.path.exists(op):
        # Try to find matching test file
        stem = of.replace('oof_seqC_','').replace('.npy','')
        t_cands = [x for x in os.listdir(oracle_dir) if stem in x and x.startswith('test_')]
        if t_cands:
            loaded[name] = (np.load(op), np.load(os.path.join(oracle_dir, t_cands[0])))
        else:
            print(f'  {name}: OOF file found but no test file')

print(f'\n=== Loaded {len(loaded)} candidate oracle files ===')
for name, (oo, ot) in loaded.items():
    corr_w = np.corrcoef(oracle_oof, np.clip(oo,0,None))[0,1]
    corr_rw = np.corrcoef(rw_oof, np.clip(oo,0,None))[0,1]
    print(f'  {name:20s}: OOF={mae(oo):.5f}  test_mean={ot.mean():.3f}  corr_oracle={corr_w:.4f}  corr_rw={corr_rw:.4f}')

# Layout-level analysis: do these files have lower residuals on high-pack layouts?
print('\n=== Layout-level residual analysis for new oracle files ===')
train_raw['oracle_pred'] = oracle_oof
train_raw['rw_pred'] = rw_oof
for name, (oo, ot) in loaded.items():
    train_raw[f'{name}_pred'] = np.clip(oo, 0, None)

ly_stats = train_raw.groupby('layout_id').agg(
    pack_mean=('pack_utilization','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    oracle_mean=('oracle_pred','mean'),
    rw_mean=('rw_pred','mean'),
).reset_index()
ly_stats['oracle_resid'] = ly_stats['y_mean'] - ly_stats['oracle_mean']
ly_stats['rw_resid'] = ly_stats['y_mean'] - ly_stats['rw_mean']

for name, (oo, ot) in loaded.items():
    ly_stats[f'{name}_mean'] = train_raw.groupby('layout_id')[f'{name}_pred'].mean().values
    ly_stats[f'{name}_resid'] = ly_stats['y_mean'] - ly_stats[f'{name}_mean']

hi_pack = ly_stats[ly_stats['pack_mean'] > 0.65].sort_values('pack_mean', ascending=False)
print(f'High-pack layouts (pack>0.65): n={len(hi_pack)}')
print(f'{"layout_id":12s}  {"pack":6s}  {"oracle_r":9s}  {"rw_r":9s}', end='')
for name in loaded.keys():
    print(f'  {name[:8]:>8s}', end='')
print()
for _, r in hi_pack.iterrows():
    print(f'{r["layout_id"]:12s}  {r["pack_mean"]:.4f}  {r["oracle_resid"]:+9.2f}  {r["rw_resid"]:+9.2f}', end='')
    for name in loaded.keys():
        print(f'  {r[f"{name}_resid"]:+8.2f}', end='')
    print()

# Correlation with pack residual
print(f'\nCorr(residual, pack_mean):')
print(f'  oracle: {np.corrcoef(ly_stats["pack_mean"], ly_stats["oracle_resid"])[0,1]:.4f}')
print(f'  rw:     {np.corrcoef(ly_stats["pack_mean"], ly_stats["rw_resid"])[0,1]:.4f}')
for name in loaded.keys():
    print(f'  {name:20s}: {np.corrcoef(ly_stats["pack_mean"], ly_stats[f"{name}_resid"])[0,1]:.4f}')

# Weight sweep for most promising files
print('\n=== Weight sweep for xgb_combined and xgb_v31 ===')
for name in ['xgb_combined', 'xgb_v31']:
    if name not in loaded:
        continue
    oo, ot = loaded[name]
    print(f'\n{name}:')
    print(f'{"w":>6}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')
    for w in [0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25]:
        b_oof = np.clip((1-w)*rw_oof + w*np.clip(oo,0,None), 0, None)
        b_test = np.clip((1-w)*rw_test + w*np.clip(ot,0,None), 0, None)
        marker = '*' if mae(b_oof) < base_oof else ''
        print(f'{w:>6.2f}  {mae(b_oof):>9.5f}  {mae(b_oof)-base_oof:>+9.6f}  {b_test.mean():>10.3f} {marker}')

# Replace oracle_seq components with xgb_combined
print('\n=== Replace oracle_seq with xgb_combined ===')
if 'xgb_combined' in loaded:
    oo_c, ot_c = loaded['xgb_combined']
    # Try using xgb_combined instead of xgb_o
    for wxgb_c in [0.12, 0.16, 0.20, 0.24, 0.28]:
        w_rest = 0.36 - wxgb_c
        wlv2 = w_rest * 0.16/0.24
        wrem = w_rest * 0.08/0.24
        b_oof = np.clip(0.64*fixed_rw_oof + wxgb_c*np.clip(oo_c,0,None) + wlv2*lv2_o + wrem*rem_o, 0, None)
        b_test = np.clip(0.64*fixed_rw_test + wxgb_c*np.clip(ot_c,0,None) + wlv2*lv2_t + wrem*rem_t, 0, None)
        marker = '*' if mae(b_oof) < base_oof else ''
        print(f'  wxgb_c={wxgb_c:.2f} wlv2={wlv2:.3f} wrem={wrem:.3f}: OOF={mae(b_oof):.5f}  delta={mae(b_oof)-base_oof:+.6f}  test={b_test.mean():.3f} {marker}')

# Best combination: replace xgb_o with xgb_combined
print('\n=== Full 3D sweep: replace xgb_o with xgb_combined, also try v31 ===')
if 'xgb_combined' in loaded:
    oo_c, ot_c = loaded['xgb_combined']
    best_val = base_oof
    best_cfg = None
    for wf in [0.60, 0.62, 0.64, 0.66, 0.68]:
        w_seq = 1.0 - wf
        for xc_frac in [0.30, 0.35, 0.40, 0.45, 0.50]:
            for lv2_frac in [0.35, 0.40, 0.45]:
                rem_frac = 1.0 - xc_frac - lv2_frac
                if rem_frac < 0.05 or rem_frac > 0.30:
                    continue
                wx = w_seq * xc_frac
                wl = w_seq * lv2_frac
                wr = w_seq * rem_frac
                b_oof = np.clip(wf*fixed_rw_oof + wx*np.clip(oo_c,0,None) + wl*lv2_o + wr*rem_o, 0, None)
                b_test = np.clip(wf*fixed_rw_test + wx*np.clip(ot_c,0,None) + wl*lv2_t + wr*rem_t, 0, None)
                if mae(b_oof) < best_val:
                    best_val = mae(b_oof)
                    best_cfg = (wf, wx, wl, wr, mae(b_oof), b_test.mean())

    if best_cfg:
        wf, wx, wl, wr, bv, tm = best_cfg
        print(f'Best: wf={wf:.2f} wxgb_c={wx:.3f} wlv2={wl:.3f} wrem={wr:.3f}: OOF={bv:.5f}  delta={bv-base_oof:+.6f}  test={tm:.3f}')
        # Generate submission
        b_oof = np.clip(wf*fixed_rw_oof + wx*np.clip(oo_c,0,None) + wl*lv2_o + wr*rem_o, 0, None)
        b_test = np.clip(wf*fixed_rw_test + wx*np.clip(ot_c,0,None) + wl*lv2_t + wr*rem_t, 0, None)
        sub = pd.read_csv('sample_submission.csv')
        sub['avg_delay_minutes_next_30m'] = b_test
        fname = f'submission_rw_xgbc_OOF{mae(b_oof):.5f}.csv'
        sub.to_csv(fname, index=False)
        print(f'Saved: {fname}  test_mean={b_test.mean():.3f}')
    else:
        print('No improvement found')

# Fold-level analysis for best config
print('\n=== Fold-level check for xgb_combined blend (w=0.10 with rw) ===')
if 'xgb_combined' in loaded:
    oo_c, ot_c = loaded['xgb_combined']
    b_oof = np.clip(0.90*rw_oof + 0.10*np.clip(oo_c,0,None), 0, None)
    b_test = np.clip(0.90*rw_test + 0.10*np.clip(ot_c,0,None), 0, None)
    print(f'Global: OOF={mae(b_oof):.5f}  delta={mae(b_oof)-base_oof:+.6f}  test={b_test.mean():.3f}')
    groups = train_raw['layout_id'].values
    gkf = GroupKFold(n_splits=5)
    for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        fo = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs],0,None))
        fb = mean_absolute_error(y_true[vs], b_oof[vs])
        print(f'  Fold {fi+1}: oracle={fo:.5f}  xgbc_blend={fb:.5f}  delta={fb-fo:+.6f}')

# Fold check for xgb_v31 blend (w=0.10)
print('\n=== Fold-level check for xgb_v31 blend (w=0.10 with rw) ===')
if 'xgb_v31' in loaded:
    oo_v31, ot_v31 = loaded['xgb_v31']
    b_oof_v31 = np.clip(0.90*rw_oof + 0.10*np.clip(oo_v31,0,None), 0, None)
    b_test_v31 = np.clip(0.90*rw_test + 0.10*np.clip(ot_v31,0,None), 0, None)
    print(f'Global: OOF={mae(b_oof_v31):.5f}  delta={mae(b_oof_v31)-base_oof:+.6f}  test={b_test_v31.mean():.3f}')
    groups = train_raw['layout_id'].values
    gkf = GroupKFold(n_splits=5)
    for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        fo = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs],0,None))
        fb = mean_absolute_error(y_true[vs], b_oof_v31[vs])
        print(f'  Fold {fi+1}: oracle={fo:.5f}  xgbv31_blend={fb:.5f}  delta={fb-fo:+.6f}')

# Compare: does xgb_combined have lower test_mean (oracle_seq bias)?
print('\n=== Directional bias analysis on high/low inflow ===')
test_raw['oracle_pred'] = oracle_test
test_raw['rw_pred'] = rw_test
for name, (oo, ot) in loaded.items():
    test_raw[f'{name}_pred'] = np.clip(ot, 0, None)

# Stratify test by inflow
q50 = test_raw['order_inflow_15m'].quantile(0.5)
q80 = test_raw['order_inflow_15m'].quantile(0.8)
print(f'Inflow quantiles: p50={q50:.1f}  p80={q80:.1f}')
for label, mask in [('high_inflow (>p80)', test_raw['order_inflow_15m'] > q80),
                    ('med_inflow (p50-p80)', (test_raw['order_inflow_15m'] > q50) & (test_raw['order_inflow_15m'] <= q80)),
                    ('low_inflow (<p50)', test_raw['order_inflow_15m'] <= q50)]:
    print(f'\n  {label}:')
    print(f'    oracle: {oracle_test[mask].mean():.3f}  rw: {rw_test[mask].mean():.3f}', end='')
    for name, (oo, ot) in loaded.items():
        print(f'  {name[:8]}: {ot[mask].mean():.3f}', end='')
    print()

print('\nDone.')
