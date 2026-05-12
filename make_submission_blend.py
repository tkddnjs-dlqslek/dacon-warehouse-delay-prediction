"""
Generate submission for best static blend: xgb=0.12, lv2=0.16, late=0.10 (OOF 8.3828)
Also checks xgb=0.12, lv2=0.20 (OOF 8.3831, LB 9.7558 current best) as reference.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_ls = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id_to_ls = [te_ls_pos[i] for i in test_raw['ID'].values]

mega_oof  = d['meta_avg_oof'][id_to_ls]
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls]
i1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls]
i2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls]
i3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed = fw['mega33']*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*i1_oof+fw['iter_r2']*i2_oof+fw['iter_r3']*i3_oof
y = train_raw['avg_delay_minutes_next_30m'].values

mega_test  = d['meta_avg_test'][te_id_to_ls]
rank_test  = np.load('results/ranking/rank_adj_test.npy')[te_id_to_ls]
i1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id_to_ls]
i2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id_to_ls]
i3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id_to_ls]
fixed_test = fw['mega33']*mega_test+fw['rank_adj']*rank_test+fw['iter_r1']*i1_test+fw['iter_r2']*i2_test+fw['iter_r3']*i3_test

xgb_oof  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
xgb_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test = np.load('results/oracle_seq/test_C_log_v2.npy')

late_oof_raw  = np.load('results/oracle_seq/oof_seqC_lgb_latepos.npy')
late_test_raw = np.load('results/oracle_seq/test_C_lgb_latepos.npy')
late_mask_tr = (train_raw['row_in_sc'].values >= 17)
late_mask_te = (test_raw['row_in_sc'].values >= 17)
late_oof = fixed.copy(); late_oof[late_mask_tr] = late_oof_raw[late_mask_tr]
late_test = fixed_test.copy(); late_test[late_mask_te] = late_test_raw[late_mask_te]

sample_sub = pd.read_csv('sample_submission.csv')

blends = [
    # name, w_xgb, w_lv2, w_late
    ('xgb012_lv2016_late010', 0.12, 0.16, 0.10),
    ('xgb012_lv2020',         0.12, 0.20, 0.00),  # current LB best reference
]

for name, wx, wl, wlt in blends:
    oof_b  = (1-wx-wl-wlt)*fixed     + wx*xgb_oof  + wl*lv2_oof  + wlt*late_oof
    test_b = (1-wx-wl-wlt)*fixed_test + wx*xgb_test + wl*lv2_test + wlt*late_test
    test_b = np.maximum(0, test_b)
    mae = np.mean(np.abs(oof_b - y))
    print(f"{name}: OOF MAE={mae:.4f}", flush=True)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': test_b})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_{name}_OOF{mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"  Saved: {fname}", flush=True)

print("Done.", flush=True)
