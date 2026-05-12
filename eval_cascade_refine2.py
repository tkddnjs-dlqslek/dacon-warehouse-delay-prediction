"""
Refine2: extend dual gate search + try triple gate + broader p thresholds.
Prev best: dual p10_h_w0.02+p20_m_w0.03 → OOF=8.37962
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle

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

with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d['meta_avg_oof'][id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

best_oof  = 0.64*fixed_oof  + 0.12*xgb_o  + 0.16*lv2_o  + 0.08*rem_o
best_test = 0.64*fixed_test + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te
curr_mae = np.mean(np.abs(best_oof - y_true))
print(f"Current base OOF: {curr_mae:.5f}")

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best = 8.37962
best_mae = prev_best
best_cfg = None; best_oof_pred = None; best_test_pred = None

# Precompute masks
masks = {}
masks_te = {}
for p in [0.05, 0.07, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    masks[p]    = (clf_oof  > p).astype(float)
    masks_te[p] = (clf_test > p).astype(float)

print(f"\n[Search1] Extended dual gate (wider p range, finer w)", flush=True)
for p1 in [0.05, 0.07, 0.10, 0.12, 0.15]:
    for p2 in [0.15, 0.20, 0.25, 0.30]:
        if p2 <= p1: continue
        for w1 in np.arange(0.01, 0.08, 0.01):
            # Apply p1/lgb_huber first
            b1    = (1 - masks[p1]*w1)*best_oof  + masks[p1]*w1*lgb_rh_oof
            b1_te = (1 - masks_te[p1]*w1)*best_test + masks_te[p1]*w1*lgb_rh_test
            for w2 in np.arange(0.01, 0.08, 0.01):
                blend = (1 - masks[p2]*w2)*b1 + masks[p2]*w2*lgb_rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'dual p{p1}_h_w{w1:.2f}+p{p2}_m_w{w2:.2f}'
                    best_oof_pred  = blend
                    best_test_pred = (1 - masks_te[p2]*w2)*b1_te + masks_te[p2]*w2*lgb_rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search2] Reverse dual: p2/lgb_mae first, then p1/lgb_huber higher conf", flush=True)
for p1 in [0.20, 0.25, 0.30, 0.40]:
    for p2 in [0.40, 0.50]:
        if p2 <= p1: continue
        for w1 in np.arange(0.01, 0.08, 0.01):
            b1    = (1 - masks[p1]*w1)*best_oof  + masks[p1]*w1*lgb_rm_oof
            b1_te = (1 - masks_te[p1]*w1)*best_test + masks_te[p1]*w1*lgb_rm_test
            for w2 in np.arange(0.01, 0.11, 0.01):
                blend = (1 - masks[p2]*w2)*b1 + masks[p2]*w2*lgb_rh_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'rev dual p{p1}_m_w{w1:.2f}+p{p2}_h_w{w2:.2f}'
                    best_oof_pred  = blend
                    best_test_pred = (1 - masks_te[p2]*w2)*b1_te + masks_te[p2]*w2*lgb_rh_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

if best_cfg:
    print(f"\n=== IMPROVED: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} (vs prev {prev_best:.5f}) ===")
    print(f"Config: {best_cfg}")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_cascade_refined2_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"\nNo improvement over {prev_best:.5f} — {best_cfg} is still best")
print("Done.")
