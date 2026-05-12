"""
Evaluate new mega33_v31 vs old mega33, find best blend with oracles, save submission.
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

# Map ls→_row_id order
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f:    d_old = pickle.load(f)
with open('results/mega33_v31_final.pkl','rb') as f: d_new = pickle.load(f)

old_oof  = d_old['meta_avg_oof'][id2];  old_test  = d_old['meta_avg_test'][te_id2]
new_oof  = d_new['meta_avg_oof'][id2];  new_test  = d_new['meta_avg_test'][te_id2]

print(f"mega33_orig OOF: {np.mean(np.abs(old_oof-y_true)):.5f}")
print(f"mega33_v31  OOF: {np.mean(np.abs(new_oof-y_true)):.5f}")
print(f"corr(old,new):   {np.corrcoef(old_oof, new_oof)[0,1]:.4f}")

# Oracle predictions
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')[te_id2]
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')[te_id2]
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')[te_id2]

# FIXED_new: replace mega33 with mega33_v31
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_new_oof = (fw['mega33']*new_oof
               + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
               + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
               + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
               + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_new_te  = (fw['mega33']*new_test
               + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
               + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
               + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
               + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

print(f"\nFIXED_new OOF: {np.mean(np.abs(fixed_new_oof-y_true)):.5f}")

# Best blend search with FIXED_new + oracles
best4_new = 0.64*fixed_new_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
print(f"best4_new (same weights): {np.mean(np.abs(best4_new-y_true)):.5f}")

best_mae = 1e9; best_cfg = ''
best_oof_pred = best4_new
best_te_pred  = 0.64*fixed_new_te + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te
for wf in np.arange(0.55, 0.80, 0.02):
    for wx in np.arange(0.06, 0.22, 0.02):
        for wl in np.arange(0.08, 0.26, 0.02):
            for wr in np.arange(0.0, 0.16, 0.02):
                if abs(wf+wx+wl+wr-1.0) > 0.001: continue
                blend = wf*fixed_new_oof + wx*xgb_o + wl*lv2_o + wr*rem_o
                mm = np.mean(np.abs(blend-y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'F={wf:.2f}+xgb={wx:.2f}+lv2={wl:.2f}+rem={wr:.2f}'
                    best_oof_pred = blend
                    best_te_pred  = wf*fixed_new_te + wx*xgb_te + wl*lv2_te + wr*rem_te

print(f"\nBest blend: {best_cfg}")
print(f"Best OOF MAE: {best_mae:.5f}")

CURRENT_BEST = 8.3825
if best_mae < CURRENT_BEST - 0.0003:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_te_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mega33v31_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} (vs {CURRENT_BEST}: {best_mae-CURRENT_BEST:+.5f}) ***")
else:
    print(f"OOF {best_mae:.5f} not better enough vs {CURRENT_BEST}. Not saved.")
print("Done.")
