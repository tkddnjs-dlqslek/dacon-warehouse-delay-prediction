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

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64):
    mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
    mega_test = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33'] - dr2 - dr3
    w2 = fw['iter_r2'] + dr2
    w3 = fw['iter_r3'] + dr3
    fx_oof  = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2*r2_oof + w3*r3_oof
    fx_test = wm*mega_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0 - wf
    wxgb = 0.12 * w_rem / 0.36
    wlv2 = 0.16 * w_rem / 0.36
    wrem = 0.08 * w_rem / 0.36
    o_oof  = np.clip(wf*fx_oof + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    o_test = np.clip(wf*fx_test + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    return o_oof, o_test

oracle_oof, oracle_test = make_pred(w34=0.0, dr2=0.0, dr3=0.0, wf=0.64)
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

sub = pd.read_csv('sample_submission.csv')
generated = []

configs = [
    # (name, w34, dr2, dr3, wf)
    ('rw_wf064',     0.00, -0.04, -0.02, 0.64),  # fixed_reweighted
    ('rw_wf066',     0.00, -0.04, -0.02, 0.66),
    ('rw_wf068',     0.00, -0.04, -0.02, 0.68),
    ('rw_wf070',     0.00, -0.04, -0.02, 0.70),
    ('rw_wf072',     0.00, -0.04, -0.02, 0.72),
    ('rw_wf075',     0.00, -0.04, -0.02, 0.75),
    ('m34_rw_wf064', 0.25, -0.04, -0.02, 0.64),  # mega34 + rw
    ('m34_rw_wf068', 0.25, -0.04, -0.02, 0.68),
    ('m34_rw_wf070', 0.25, -0.04, -0.02, 0.70),
    ('m34_rw_wf072', 0.25, -0.04, -0.02, 0.72),
    ('m34_wf064',    0.25,  0.00,  0.00, 0.64),  # mega34 only
    ('m34_wf068',    0.25,  0.00,  0.00, 0.68),
    ('orig_wf068',   0.00,  0.00,  0.00, 0.68),  # original weights
    ('orig_wf070',   0.00,  0.00,  0.00, 0.70),
    ('orig_wf072',   0.00,  0.00,  0.00, 0.72),
]

print(f'\n{"config":25s}  {"OOF":>9}  {"delta_OOF":>10}  {"test_mean":>10}  {"delta_test":>10}')
for name, w34, dr2, dr3, wf in configs:
    o_oof, o_test = make_pred(w34=w34, dr2=dr2, dr3=dr3, wf=wf)
    d_oof = mae(o_oof) - base_oof
    d_test = o_test.mean() - oracle_test.mean()
    marker = '*' if d_oof < 0 else ''
    print(f'{name:25s}  {mae(o_oof):>9.5f}  {d_oof:>+10.6f}  {o_test.mean():>10.3f}  {d_test:>+10.4f} {marker}')
    generated.append((name, mae(o_oof), o_test.mean(), o_oof, o_test))

# Save key submissions
print('\n--- Saving key submissions ---')
save_list = [
    'rw_wf068', 'rw_wf072', 'm34_rw_wf064', 'm34_rw_wf068', 'm34_rw_wf070',
    'm34_wf064', 'orig_wf068', 'orig_wf072'
]
for name, oof_val, test_mean, o_oof, o_test in generated:
    if name in save_list:
        sub['avg_delay_minutes_next_30m'] = o_test
        fname = f'submission_{name}_OOF{oof_val:.5f}.csv'
        sub.to_csv(fname, index=False)
        print(f'  Saved: {fname}  (OOF={oof_val:.5f}  test_mean={test_mean:.3f})')

# Final recommendation summary
print('\n=== SUBMISSION RECOMMENDATION ===')
print(f'Baseline (oracle_NEW, LB=9.7527): OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print()
print('Hypothesis: Higher test_mean → better LB for unseen high-load layouts')
print('(OOF-LB inversion: seen-layout overfit improves OOF but hurts LB)')
print()
print('Tier 1 — FIXED weight improvements (genuine, low risk):')
for name, oof_val, test_mean, _, _ in generated:
    if 'rw_wf' in name and name in ['rw_wf064','rw_wf068','rw_wf070']:
        print(f'  {name}: OOF={oof_val:.5f} ({oof_val-base_oof:+.6f})  test_mean={test_mean:.3f} ({test_mean-oracle_test.mean():+.3f})')
print()
print('Tier 2 — mega34 + FIXED rw (OOF improves but test unchanged):')
for name, oof_val, test_mean, _, _ in generated:
    if 'm34_rw' in name:
        print(f'  {name}: OOF={oof_val:.5f} ({oof_val-base_oof:+.6f})  test_mean={test_mean:.3f} ({test_mean-oracle_test.mean():+.3f})')
print()
print('Tier 3 — original weights + higher wf (no FIXED rw):')
for name, oof_val, test_mean, _, _ in generated:
    if 'orig_wf' in name:
        print(f'  {name}: OOF={oof_val:.5f} ({oof_val-base_oof:+.6f})  test_mean={test_mean:.3f} ({test_mean-oracle_test.mean():+.3f})')

# Pack-corrected version of the best fixed_rw
print('\n--- Pack-corrected variants (OOF unchanged, speculative) ---')
# Use the wh217 reference from analyze_extreme_scenarios
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
rw_oof_64, rw_test_64 = make_pred(0.0, -0.04, -0.02, 0.64)
rw_oof_70, rw_test_70 = make_pred(0.0, -0.04, -0.02, 0.70)
train_raw['_pred'] = rw_oof_64
wh217 = train_raw[train_raw['layout_id']=='WH_217'].copy()
sc217 = wh217.groupby('scenario_id').agg(
    pack_mean=('pack_utilization','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    pred_mean=('_pred','mean')
).reset_index()
sc217_hi = sc217[sc217['pack_mean'] >= 0.99]
residual = sc217_hi['y_mean'].mean() - sc217_hi['pred_mean'].mean()

for base_name, base_test in [('rw_wf064', rw_test_64), ('rw_wf070', rw_test_70)]:
    for scale in [0.3, 0.5, 0.7]:
        preds = base_test.copy()
        for wh_id in ['WH_201', 'WH_246', 'WH_283']:
            df = test_raw[test_raw['layout_id']==wh_id].copy()
            sc_stats = df.groupby('scenario_id')['pack_utilization'].mean()
            n_sc = len(sc_stats)
            n_hi_sc = (sc_stats >= 0.99).sum()
            correction = scale * (n_hi_sc / n_sc) * residual
            mask = test_raw['layout_id'] == wh_id
            preds[mask] = np.clip(base_test[mask] + correction, 0, None)
        print(f'  {base_name} + pack_s{scale:.1f}: test_mean={preds.mean():.3f}  delta={preds.mean()-oracle_test.mean():+.3f}')

print('\nDone.')
