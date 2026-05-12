import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, glob, os, pickle
import warnings; warnings.filterwarnings('ignore')
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen_mean = oracle_new_t[unseen_mask].mean()
oracle_seen_mean = oracle_new_t[~unseen_mask].mean()

print(f"oracle_NEW: OOF={np.mean(np.abs(y_true-fw4_oo)):.4f}  seen={oracle_seen_mean:.3f}  unseen={oracle_unseen_mean:.3f}")
print()

# Scan test files
test_files = sorted(glob.glob('results/**/*test*.npy', recursive=True))
rows = []
for fpath in test_files:
    try:
        fname = os.path.basename(fpath)
        oof_fname = fname.replace('test', 'oof')
        oof_path = os.path.join(os.path.dirname(fpath), oof_fname)
        if not os.path.exists(oof_path):
            continue

        t = np.clip(np.load(fpath), 0, None)
        o = np.clip(np.load(oof_path), 0, None)

        if t.ndim > 1: t = t.mean(axis=1)
        if o.ndim > 1: o = o.mean(axis=1)

        if len(t) != len(id_order): continue
        if len(o) == len(id2): o = o[id2]
        if len(o) != len(y_true): continue

        oof_mae = np.mean(np.abs(y_true - o))
        corr = np.corrcoef(fw4_oo, o)[0,1]
        t_seen = t[~unseen_mask].mean()
        t_unseen = t[unseen_mask].mean()
        delta_u = t_unseen - oracle_unseen_mean

        rows.append({
            'file': fpath.replace('results/', '').replace('results\\', ''),
            'oof_mae': oof_mae, 'corr': corr,
            't_seen': t_seen, 't_unseen': t_unseen, 'delta_u': delta_u
        })
    except:
        pass

df = pd.DataFrame(rows)

print("=== Files with test_unseen HIGHER than oracle_NEW (delta_u > 0), OOF < 11 ===")
subset = df[(df['delta_u'] > 0) & (df['oof_mae'] < 11)].sort_values('delta_u', ascending=False)
for _, r in subset.head(25).iterrows():
    print(f"  {r['file']:<55} OOF={r['oof_mae']:7.4f}  corr={r['corr']:.4f}  unseen={r['t_unseen']:7.3f}  Δu={r['delta_u']:+.3f}")

print()
print("=== Files with best OOF (< 9), corr > 0.95, sorted by OOF ===")
subset2 = df[(df['oof_mae'] < 9) & (df['corr'] > 0.95)].sort_values('oof_mae')
for _, r in subset2.head(20).iterrows():
    print(f"  {r['file']:<55} OOF={r['oof_mae']:7.4f}  corr={r['corr']:.4f}  seen={r['t_seen']:6.3f}  unseen={r['t_unseen']:6.3f}  Δu={r['delta_u']:+.3f}")
