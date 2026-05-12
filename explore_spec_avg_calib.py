import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

spec_avg_t = np.clip(np.load('results/cascade/spec_avg_test.npy')[te_id2], 0, None)
spec_avg_o = np.clip(np.load('results/cascade/spec_avg_oof.npy')[id2], 0, None)
h10_t = np.clip(np.load('results/oracle_seq/test_C_huber10.npy'), 0, None)
h10_o = np.clip(np.load('results/oracle_seq/oof_seqC_huber10.npy'), 0, None)

# Rebuild oracle_NEW OOF (fw4_oo)
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)

sub_tmpl = pd.read_csv('sample_submission.csv')

# ============================================================
# Per-bucket OOF: spec_avg vs oracle_NEW on TRAINING data
# ============================================================
print("="*70)
print("Per-bucket OOF calibration: oracle_NEW vs spec_avg vs h10")
print("(bucketed by oracle_NEW prediction on training data)")
print("="*70)
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
y = y_true
p_oN = fw4_oo
p_spec = spec_avg_o
p_h10 = h10_o

print(f"\n  {'bucket':12s} {'n':>7} {'y_mean':>8} {'oN_resid':>9} {'spec_resid':>11} {'h10_resid':>10} {'spec_win':>8}")
spec_wins = 0
h10_wins = 0
total_n = 0
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN >= lo) & (p_oN < hi)
    if mask.sum() > 0:
        ym = y[mask].mean()
        resid_oN = p_oN[mask].mean() - ym
        resid_spec = p_spec[mask].mean() - ym
        resid_h10 = p_h10[mask].mean() - ym
        spec_better = abs(resid_spec) < abs(resid_oN)
        h10_better = abs(resid_h10) < abs(resid_oN)
        total_n += mask.sum()
        if spec_better: spec_wins += mask.sum()
        if h10_better: h10_wins += mask.sum()
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y={ym:8.3f}  oN={resid_oN:+9.3f}  spec={resid_spec:+11.3f}  h10={resid_h10:+10.3f}  spec>oN:{spec_better}")
print(f"\n  spec_avg better than oracle_NEW for {spec_wins}/{total_n} ({100*spec_wins/total_n:.1f}%) rows")
print(f"  h10 better than oracle_NEW for {h10_wins}/{total_n} ({100*h10_wins/total_n:.1f}%) rows")

# ============================================================
# Training layout-level spec_avg analysis
# ============================================================
print("\n" + "="*70)
print("Training: per-layout spec_avg correction vs oracle_NEW")
print("(top unseen-equivalent layouts by prediction mean)")
print("="*70)
layout_ids = train_raw['layout_id'].values
# Compute per-layout stats
from collections import defaultdict
lay_stats = defaultdict(lambda: {'n':0, 'y_sum':0, 'oN_sum':0, 'spec_sum':0, 'h10_sum':0})
for i, (lid, yi, poi, spi, hi) in enumerate(zip(layout_ids, y, p_oN, p_spec, p_h10)):
    lay_stats[lid]['n'] += 1
    lay_stats[lid]['y_sum'] += yi
    lay_stats[lid]['oN_sum'] += poi
    lay_stats[lid]['spec_sum'] += spi
    lay_stats[lid]['h10_sum'] += hi

lay_df = []
for lid, s in lay_stats.items():
    n = s['n']
    ym = s['y_sum']/n
    oN_m = s['oN_sum']/n
    spec_m = s['spec_sum']/n
    h10_m = s['h10_sum']/n
    lay_df.append({'layout_id':lid, 'n':n, 'y_mean':ym, 'oN_mean':oN_m, 'spec_mean':spec_m, 'h10_mean':h10_m,
                   'oN_resid':oN_m-ym, 'spec_resid':spec_m-ym, 'h10_resid':h10_m-ym})
lay_df = pd.DataFrame(lay_df).sort_values('y_mean', ascending=False)
print(f"\n  {'layout_id':12s} {'n':>6} {'y_mean':>8} {'oN_resid':>9} {'spec_resid':>11} {'h10_resid':>10}")
for _, row in lay_df.head(20).iterrows():
    print(f"  {row['layout_id']:12s} {row['n']:6d}  {row['y_mean']:8.3f}  {row['oN_resid']:+9.3f}  {row['spec_resid']:+11.3f}  {row['h10_resid']:+10.3f}")

# ============================================================
# KEY: spec_avg per-bucket OOF MAE breakdown
# ============================================================
print("\n" + "="*70)
print("Per-bucket MAE: oracle_NEW vs spec_avg vs h10 (training)")
print("="*70)
print(f"\n  {'bucket':12s} {'n':>7} {'oN_mae':>8} {'spec_mae':>9} {'h10_mae':>8} {'spec_win':>9}")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_oN >= lo) & (p_oN < hi)
    if mask.sum() > 0:
        mae_oN = np.mean(np.abs(p_oN[mask]-y[mask]))
        mae_spec = np.mean(np.abs(p_spec[mask]-y[mask]))
        mae_h10 = np.mean(np.abs(p_h10[mask]-y[mask]))
        spec_better = mae_spec < mae_oN
        h10_better = mae_h10 < mae_oN
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  oN={mae_oN:8.3f}  spec={mae_spec:9.3f}  h10={mae_h10:8.3f}  spec>{h10_better}")

# ============================================================
# Save optimized spec_avg-based hybrid candidates
# ============================================================
print("\n" + "="*70)
print("Saving optimized spec_avg hybrid candidates")
print("="*70)

below40_mask = unseen_mask & (oracle_new_t < 40)
above40_mask = unseen_mask & (oracle_new_t >= 40)

# Hybrid: h10 for below40, spec_avg for above40
for w_lo, w_hi in [(0.3,0.1), (0.5,0.1), (0.3,0.2), (0.5,0.2)]:
    ct = oracle_new_t.copy()
    ct[below40_mask] = (1-w_lo)*oracle_new_t[below40_mask] + w_lo*h10_t[below40_mask]
    ct[above40_mask] = (1-w_hi)*oracle_new_t[above40_mask] + w_hi*spec_avg_t[above40_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    label = f"hybh10sp_{int(w_lo*10)}{int(w_hi*10)}"
    fname = f"FINAL_NEW_oN_{label}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

# spec_avg unseen-only at 5%
for w in [0.05, 0.08]:
    ct = oracle_new_t.copy()
    ct[unseen_mask] = (1-w)*oracle_new_t[unseen_mask] + w*spec_avg_t[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    fname = f"FINAL_NEW_oN_specAvg_u{int(w*100):02d}_OOF8.3825.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}  seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}  Δ={du:+.3f}")

print("\nDone.")
