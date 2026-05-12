"""
Refine3: exhaustive search around best (p0.12_h_w0.03 + p0.25_m_w0.03).
Also tries triple gate.
Prev best: 8.37910
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

prev_best = 8.37910
best_mae = prev_best
best_cfg = None; best_oof_pred = None; best_test_pred = None

# Fine mask grid
p_vals = [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15,
          0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40, 0.50]
masks    = {p: (clf_oof  > p).astype(float) for p in p_vals}
masks_te = {p: (clf_test > p).astype(float) for p in p_vals}

print(f"\n[Search1] Fine exhaustive dual gate (all p combinations)", flush=True)
for p1 in [0.05, 0.07, 0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15]:
    for w1 in np.arange(0.01, 0.08, 0.005):
        b1    = (1 - masks[p1]*w1)*best_oof  + masks[p1]*w1*lgb_rh_oof
        b1_te = (1 - masks_te[p1]*w1)*best_test + masks_te[p1]*w1*lgb_rh_test
        for p2 in [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]:
            if p2 <= p1: continue
            for w2 in np.arange(0.01, 0.08, 0.005):
                blend = (1 - masks[p2]*w2)*b1 + masks[p2]*w2*lgb_rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'dual p{p1}_h_w{w1:.3f}+p{p2}_m_w{w2:.3f}'
                    best_oof_pred  = blend
                    best_test_pred = (1 - masks_te[p2]*w2)*b1_te + masks_te[p2]*w2*lgb_rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search2] Triple gate: p_broad/huber + p_mid/mae + p_narrow/huber_boost", flush=True)
# Start from best dual config, add 3rd gate
if best_oof_pred is not None:
    intermediate_oof  = best_oof_pred
    intermediate_test = best_test_pred
else:
    # Use known best: p0.12_h_w0.03 + p0.25_m_w0.03
    m12    = masks[0.12];    m12_te = masks_te[0.12]
    m25    = masks[0.25];    m25_te = masks_te[0.25]
    b1    = (1-m12*0.03)*best_oof  + m12*0.03*lgb_rh_oof
    b1_te = (1-m12_te*0.03)*best_test + m12_te*0.03*lgb_rh_test
    intermediate_oof  = (1-m25*0.03)*b1  + m25*0.03*lgb_rm_oof
    intermediate_test = (1-m25_te*0.03)*b1_te + m25_te*0.03*lgb_rm_test

for p3 in [0.40, 0.50]:
    for w3 in np.arange(0.01, 0.21, 0.01):
        for spec3_oof, spec3_test, sname in [
            (lgb_rh_oof, lgb_rh_test, 'h'),
            (lgb_rm_oof, lgb_rm_test, 'm')
        ]:
            blend = (1-masks[p3]*w3)*intermediate_oof + masks[p3]*w3*spec3_oof
            mm = np.mean(np.abs(blend - y_true))
            if mm < best_mae:
                best_mae = mm
                best_cfg = f'triple+p{p3}_{sname}_w{w3:.2f}'
                best_oof_pred  = blend
                best_test_pred = (1-masks_te[p3]*w3)*intermediate_test + masks_te[p3]*w3*spec3_test
                print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

if best_cfg:
    print(f"\n=== IMPROVED: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} (vs prev {prev_best:.5f}) ===")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_cascade_refined3_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"\nConverged — no improvement over {prev_best:.5f}")
print("Done.")
