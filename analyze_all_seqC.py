import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob
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
test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof  = (fw['mega33']*d33['meta_avg_oof'][id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_oof = np.clip(0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)

xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW baseline: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

# Scan all seqC OOF files
oof_files = sorted(glob.glob('results/oracle_seq/oof_seqC_*.npy'))
print(f'\nFound {len(oof_files)} seqC OOF files')

results = []
for f in oof_files:
    name = os.path.basename(f).replace('oof_seqC_','').replace('.npy','')
    try:
        p = np.load(f)
        if len(p) != len(y_true):
            continue
        p_clipped = np.clip(p, 0, None)
        p_mae = mae(p_clipped)
        corr_with_oracle = float(np.corrcoef(oracle_oof, p_clipped)[0,1])
        corr_with_fixed  = float(np.corrcoef(fixed_oof,  p_clipped)[0,1])

        # Test correlation (if test file exists)
        test_f = f.replace('oof_seqC_','test_C_')
        test_mean = None
        if os.path.exists(test_f):
            pt = np.load(test_f)
            test_mean = float(np.clip(pt, 0, None).mean())

        results.append({
            'name': name, 'oof': p_mae, 'corr_oracle': corr_with_oracle,
            'corr_fixed': corr_with_fixed, 'test_mean': test_mean
        })
    except Exception as e:
        pass

df_res = pd.DataFrame(results).sort_values('oof')
print(f'\n{"Name":35s}  {"OOF":9s}  {"delta":8s}  {"corr_oracle":12s}  {"test_mean":10s}')
for _, r in df_res.iterrows():
    tm = f'{r["test_mean"]:.3f}' if r["test_mean"] is not None else 'N/A'
    print(f'{r["name"]:35s}  {r["oof"]:9.5f}  {r["oof"]-base_oof:+8.5f}  {r["corr_oracle"]:12.4f}  {tm:>10s}')

# Blend oracle_NEW with each seqC oracle
print(f'\n--- Blend oracle_NEW + seqC oracle at w=0.05,0.10 ---')
print(f'{"Name":35s}  {"w=0.05_delta":14s}  {"w=0.10_delta":14s}  {"corr":8s}')
for _, r in df_res[df_res['corr_oracle'] < 0.98].sort_values('corr_oracle').iterrows():
    f = f'results/oracle_seq/oof_seqC_{r["name"]}.npy'
    try:
        p = np.clip(np.load(f), 0, None)
        b05 = np.clip(0.95*oracle_oof + 0.05*p, 0, None)
        b10 = np.clip(0.90*oracle_oof + 0.10*p, 0, None)
        d05 = mae(b05) - base_oof
        d10 = mae(b10) - base_oof
        marker = ' **' if d05 < 0 or d10 < 0 else ''
        print(f'{r["name"]:35s}  {d05:+14.5f}  {d10:+14.5f}  {r["corr_oracle"]:.4f}{marker}')
    except:
        pass

# Best combinations: 4-way or 5-way blend
print('\n--- Best 4-way: oracle_NEW + top low-corr oracles ---')
low_corr = df_res[df_res['corr_oracle'] < 0.97].sort_values('oof').head(5)
for _, r in low_corr.iterrows():
    f = f'results/oracle_seq/oof_seqC_{r["name"]}.npy'
    try:
        p = np.clip(np.load(f), 0, None)
        for w in [0.03, 0.05, 0.08, 0.10]:
            b = np.clip((1-w)*oracle_oof + w*p, 0, None)
            if mae(b) < base_oof:
                print(f'  {r["name"]} w={w}: OOF={mae(b):.5f}  delta={mae(b)-base_oof:+.5f}')
    except:
        pass

print('\nDone.')
