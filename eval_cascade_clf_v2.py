"""
Eval with recall-boosted classifier v2 (spw=200).
Uses already-trained v2 specialists (lgb_raw_huber, lgb_raw_mae).
Tries larger w values at high-confidence gate.
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

clf_oof  = np.load('results/cascade/clf_v2_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_v2_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

y_high = (y_true > 80).astype(int)
print(f"CLF v2: P(high)_mean={clf_oof.mean():.4f}  P(high|y>80)={clf_oof[y_high==1].mean():.4f}")
for pct in [90, 95, 97, 99]:
    thresh = np.percentile(clf_oof, pct)
    n_above = (clf_oof > thresh).sum()
    recall = ((clf_oof > thresh) & (y_high == 1)).sum() / y_high.sum()
    precision = ((clf_oof > thresh) & (y_high == 1)).sum() / max(n_above, 1)
    print(f"  top {100-pct}% (p>{thresh:.4f}): recall={recall:.3f}  precision={precision:.3f}")

prev_best = 8.37905
best_mae = prev_best
best_cfg = None; best_oof_pred = None; best_test_pred = None

p_vals = sorted(set([np.percentile(clf_oof, p) for p in range(85, 100)]))
masks    = {p: (clf_oof  > p).astype(float) for p in p_vals}
masks_te = {p: (clf_test > p).astype(float) for p in p_vals}

print(f"\n[Search1] Hard gate with clf_v2 (larger w range)", flush=True)
for p_thresh in p_vals:
    for spec_oof, spec_test, sname in [
        (lgb_rh_oof, lgb_rh_test, 'huber'),
        (lgb_rm_oof, lgb_rm_test, 'mae'),
    ]:
        for w in np.arange(0.01, 0.41, 0.01):
            blend = (1 - masks[p_thresh]*w)*best_oof + masks[p_thresh]*w*spec_oof
            mm = np.mean(np.abs(blend - y_true))
            if mm < best_mae:
                best_mae = mm
                best_cfg = f'clf_v2 p>{p_thresh:.4f} {sname} w={w:.2f}'
                best_oof_pred  = blend
                best_test_pred = (1 - masks_te[p_thresh]*w)*best_test + masks_te[p_thresh]*w*spec_test
                print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search2] Dual gate with clf_v2", flush=True)
for p1 in p_vals[:8]:  # bottom half (broader gate)
    for w1 in np.arange(0.01, 0.10, 0.01):
        b1    = (1-masks[p1]*w1)*best_oof  + masks[p1]*w1*lgb_rh_oof
        b1_te = (1-masks_te[p1]*w1)*best_test + masks_te[p1]*w1*lgb_rh_test
        for p2 in p_vals[4:]:  # top half (tighter gate)
            if p2 <= p1: continue
            for w2 in np.arange(0.01, 0.21, 0.01):
                blend = (1-masks[p2]*w2)*b1 + masks[p2]*w2*lgb_rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'clf_v2 dual p{p1:.3f}_h_w{w1:.2f}+p{p2:.3f}_m_w{w2:.2f}'
                    best_oof_pred  = blend
                    best_test_pred = (1-masks_te[p2]*w2)*b1_te + masks_te[p2]*w2*lgb_rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

if best_cfg:
    print(f"\n=== IMPROVED: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} ===")
    print(f"Config: {best_cfg}")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_clf_v2_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"\nNo improvement over {prev_best:.5f}")
print("Done.")
