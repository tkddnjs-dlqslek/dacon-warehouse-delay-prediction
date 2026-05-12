import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
import warnings; warnings.filterwarnings('ignore')
os.chdir("C:/Users/user/Desktop/데이콘 4월")

# Load y_true in sorted order
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

# Reconstruct oracle_NEW OOF
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
oracle_mae = np.mean(np.abs(y_true - fw4_oo))

print(f"oracle_NEW OOF MAE: {oracle_mae:.4f}")
print(f"{'Component':<60} {'MAE':>8} {'corr':>7} {'blend5%':>9} {'blend10%':>9} {'best_w':>7}")
print("-"*110)

# Scan all _oof.npy files
oof_files = sorted(glob.glob('results/**/*_oof*.npy', recursive=True))
oof_files += sorted(glob.glob('results/**/oof*.npy', recursive=True))
oof_files = sorted(set(oof_files))

results_list = []
for fpath in oof_files:
    try:
        arr = np.load(fpath)
        # Skip if wrong size
        if arr.shape[0] == len(y_true):
            arr_c = np.clip(arr, 0, None)
        elif arr.shape[0] == len(id2):
            arr_c = np.clip(arr[id2] if arr.ndim == 1 else arr, 0, None)
        else:
            continue

        # Handle 2D arrays (take first column or mean)
        if arr_c.ndim > 1:
            arr_c = arr_c.mean(axis=1) if arr_c.shape[1] <= 10 else arr_c[:, 0]

        if len(arr_c) != len(y_true): continue

        mae_solo = np.mean(np.abs(y_true - arr_c))
        corr = np.corrcoef(fw4_oo, arr_c)[0,1]

        # Best blend weight
        best_w, best_blend_mae = 0, oracle_mae
        for w in np.arange(0.02, 0.51, 0.02):
            bl = np.clip((1-w)*fw4_oo + w*arr_c, 0, None)
            m = np.mean(np.abs(y_true - bl))
            if m < best_blend_mae:
                best_blend_mae = m; best_w = w

        blend5 = np.mean(np.abs(y_true - np.clip(0.95*fw4_oo + 0.05*arr_c, 0, None)))
        blend10 = np.mean(np.abs(y_true - np.clip(0.90*fw4_oo + 0.10*arr_c, 0, None)))

        results_list.append({
            'file': fpath.replace('results/', ''),
            'mae': mae_solo, 'corr': corr,
            'blend5': blend5, 'blend10': blend10,
            'best_w': best_w, 'best_mae': best_blend_mae
        })
    except Exception as e:
        pass

df = pd.DataFrame(results_list)
df = df.sort_values('best_mae')

# Show top 30 by best blend MAE that improve on oracle
improved = df[df['best_mae'] < oracle_mae - 0.0001].head(30)
print(f"\n=== Components that IMPROVE oracle_NEW (best_mae < {oracle_mae:.4f}) ===")
for _, row in improved.iterrows():
    delta = row['best_mae'] - oracle_mae
    print(f"  {row['file']:<55} MAE={row['mae']:7.4f}  corr={row['corr']:.4f}  "
          f"blend5={row['blend5']:.4f}  blend10={row['blend10']:.4f}  "
          f"best_w={row['best_w']:.2f}  best={row['best_mae']:.4f} ({delta:+.4f})")

print(f"\nTotal files scanned: {len(df)}. Files that improve: {len(improved)}")

# Extra: check round4 specifically
print("\n=== Specific checks ===")
r4 = np.load('results/iter_pseudo/round4_oof.npy')
r4_id = r4[id2] if r4.shape[0] != len(y_true) else r4
r4_c = np.clip(r4_id, 0, None)
print(f"round4_oof: MAE={np.mean(np.abs(y_true-r4_c)):.4f}  corr={np.corrcoef(fw4_oo,r4_c)[0,1]:.4f}")
for w in [0.01, 0.02, 0.03, 0.05, 0.08]:
    bl = np.clip((1-w)*fw4_oo + w*r4_c, 0, None)
    print(f"  oracle(1-{w})+r4({w}): {np.mean(np.abs(y_true-bl)):.4f}")

rank_ens = np.load('results/ranking_variants/rank_ens_oof.npy')
re = np.clip(rank_ens, 0, None)
print(f"\nrank_ens_oof: MAE={np.mean(np.abs(y_true-re)):.4f}  corr={np.corrcoef(fw4_oo,re)[0,1]:.4f}")
for w in [0.05, 0.10, 0.15]:
    bl = np.clip((1-w)*fw4_oo + w*re, 0, None)
    print(f"  oracle(1-{w})+rank_ens({w}): {np.mean(np.abs(y_true-bl)):.4f}")
