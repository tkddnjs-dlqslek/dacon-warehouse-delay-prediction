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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# Load lag_target
lag_oof = np.load('results/lag_target/lag_oof.npy')
lag_test = np.load('results/lag_target/lag_test.npy')
lag_oof_c = np.clip(lag_oof, 0, None)
lag_test_c = np.clip(lag_test, 0, None)

print(f"lag_target: OOF={mae_fn(lag_oof_c):.5f}  seen={lag_test_c[seen_mask].mean():.3f}  unseen={lag_test_c[unseen_mask].mean():.3f}")
print(f"  r(oracle_NEW) = {pearsonr(lag_test_c, oracle_new_t)[0]:.4f}")

# Check oracle_NEW OOF for blending
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o_raw=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o_raw+0.10*xgbc_o+0.08*mono_o,0,None)

print(f"\noracle_NEW OOF: {mae_fn(fw4_oo):.5f}")

# Blend oracle_NEW with lag_target at small weights
print("\n" + "="*70)
print("oracle_NEW × (1-w) + lag_target × w blend")
print("="*70)
for w in [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]:
    blend_oof = np.clip((1-w)*fw4_oo + w*lag_oof_c, 0, None)
    blend_test = np.clip((1-w)*oracle_new_t + w*lag_test_c, 0, None)
    oof_mae = mae_fn(blend_oof)
    print(f"  w={w:.2f}: OOF={oof_mae:.5f}  seen={blend_test[seen_mask].mean():.3f}  unseen={blend_test[unseen_mask].mean():.3f}  ΔOOF={oof_mae-8.37624:+.5f}")

# lag_target per-bucket analysis
print("\n" + "="*70)
print("lag_target per-bucket analysis")
print("="*70)
p_lag = lag_oof_c
y = y_true
bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 200]
print(f"\n  {'pred_bucket':20s} {'n':>7} {'y_mean':>8} {'p_mean':>8} {'resid':>8}")
for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    mask = (p_lag >= lo) & (p_lag < hi)
    if mask.sum() > 0:
        ym = y[mask].mean()
        pm = p_lag[mask].mean()
        resid = pm - ym
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():7d}  y={ym:8.3f}  p={pm:8.3f}  resid={resid:+8.3f}")

# Check lag_target test distribution for unseen
p_unseen_lag = lag_test_c[unseen_mask]
print(f"\nlag_target unseen test prediction distribution:")
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (p_unseen_lag >= lo) & (p_unseen_lag < hi)
    if mask.sum() > 0:
        print(f"  [{lo:3d},{hi:3d}): n={mask.sum():6d}  mean={p_unseen_lag[mask].mean():.3f}")

# Key insight: does lag_target predict HIGHER for unseen than oracle_NEW?
diff = lag_test_c[unseen_mask] - oracle_new_t[unseen_mask]
print(f"\nlag_test_c - oracle_new_t for unseen:")
print(f"  mean={diff.mean():.3f}  std={diff.std():.3f}")
print(f"  pct5={np.percentile(diff,5):.3f}  median={np.median(diff):.3f}  pct95={np.percentile(diff,95):.3f}")

print("\nDone.")
