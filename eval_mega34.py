"""
Evaluate mega34 vs mega33, find best blend with oracles, save submission if improved.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

oof33 = d33['meta_avg_oof'][id2];  te33 = d33['meta_avg_test'][te_id2]
oof34 = d34['meta_avg_oof'][id2];  te34 = d34['meta_avg_test'][te_id2]

print(f"mega33 OOF: {np.mean(np.abs(oof33-y_true)):.5f}")
print(f"mega34 OOF: {np.mean(np.abs(oof34-y_true)):.5f}")
print(f"corr(33,34): {np.corrcoef(oof33, oof34)[0,1]:.4f}")

# Oracle predictions — OOF and test are in row_id order (no te_id2 needed)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

def make_fixed(oof_m, te_m):
    fo = (fw['mega33']*oof_m
         + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
         + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
         + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
         + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
    ft = (fw['mega33']*te_m
         + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
         + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
         + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
         + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
    return fo, ft

fixed33_oof, fixed33_te = make_fixed(oof33, te33)
fixed34_oof, fixed34_te = make_fixed(oof34, te34)

# Current best: FIXED33×0.64+xgb×0.12+lv2×0.16+rem×0.08
curr_best_oof = 0.64*fixed33_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
curr_best_mae = np.mean(np.abs(curr_best_oof-y_true))
print(f"\nCurrent best (mega33 blend) OOF: {curr_best_mae:.5f}")

# Grid search: blend mega33_fixed + mega34 raw + oracles
# Try replacing mega33 with mega34, also try blending both
print("\n--- Search: replace mega33 with mega34 ---")
best_mae = curr_best_mae; best_cfg = ''; best_oof_pred = curr_best_oof
best_te_pred = 0.64*fixed33_te + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te

for wf in [0.56,0.58,0.60,0.62,0.64,0.66,0.68,0.70]:
    for wx in [0.06,0.08,0.10,0.12,0.14,0.16]:
        for wl in [0.08,0.10,0.12,0.14,0.16,0.18,0.20]:
            for wr in [0.00,0.02,0.04,0.06,0.08,0.10]:
                ws = round(wf+wx+wl+wr, 4)
                if ws != 1.0: continue
                blend = wf*fixed34_oof + wx*xgb_o + wl*lv2_o + wr*rem_o
                mm = np.mean(np.abs(blend-y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'F34={wf:.2f}+xgb={wx:.2f}+lv2={wl:.2f}+rem={wr:.2f}'
                    best_oof_pred = blend
                    best_te_pred  = wf*fixed34_te + wx*xgb_te + wl*lv2_te + wr*rem_te

print(f"Best (mega34-only blend): {best_cfg if best_cfg else 'no improvement'}")
print(f"Best OOF: {best_mae:.5f}  (delta: {best_mae-curr_best_mae:+.5f})")

# Also try 2-component blend: α*mega33 + (1-α)*mega34 instead of FIXED
print("\n--- Search: blend mega33 + mega34 together ---")
for w33 in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
    w34 = round(1.0 - w33, 1)
    blended_oof = w33*fixed33_oof + w34*fixed34_oof
    blended_te  = w33*fixed33_te  + w34*fixed34_te
    for wx in [0.06,0.08,0.10,0.12,0.14,0.16]:
        for wl in [0.08,0.10,0.12,0.14,0.16,0.18,0.20]:
            for wr in [0.00,0.02,0.04,0.06,0.08,0.10]:
                wf = round(1.0 - wx - wl - wr, 4)
                if wf <= 0: continue
                blend = wf*blended_oof + wx*xgb_o + wl*lv2_o + wr*rem_o
                mm = np.mean(np.abs(blend-y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'(33x{w33}+34x{w34})x{wf:.2f}+xgb={wx:.2f}+lv2={wl:.2f}+rem={wr:.2f}'
                    best_oof_pred = blend
                    best_te_pred  = wf*blended_te + wx*xgb_te + wl*lv2_te + wr*rem_te

print(f"\n=== FINAL BEST ===")
print(f"Config: {best_cfg if best_cfg else 'no improvement vs mega33'}")
print(f"OOF MAE: {best_mae:.5f}  (delta vs {curr_best_mae:.5f}: {best_mae-curr_best_mae:+.5f})")

CURRENT_BEST = 8.3825
if best_mae < CURRENT_BEST - 0.0003:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_te_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mega34_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} (vs current {CURRENT_BEST}: {best_mae-CURRENT_BEST:+.5f}) ***")
else:
    print(f"\nOOF {best_mae:.5f} not better enough vs {CURRENT_BEST}. Not saved.")
print("Done.")
