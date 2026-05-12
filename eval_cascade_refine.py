"""
Refined hard gate search: finer grid + dual gate + combined specialists.
Current best: hard p>0.20 v2_lgb_mae w=0.05 → OOF=8.38049
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
print(f"Current best OOF: {curr_mae:.5f}")

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]

lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

prev_best_mae = 8.38049  # from hardgate
best_mae = prev_best_mae
best_cfg = None
best_oof_pred  = None
best_test_pred = None

print(f"\n[Search1] Fine-grained w around best (p>0.20 v2_lgb_mae)", flush=True)
mask20    = (clf_oof  > 0.20).astype(float)
mask20_te = (clf_test > 0.20).astype(float)
for w in np.arange(0.01, 0.16, 0.01):
    blend = (1 - mask20*w)*best_oof + mask20*w*lgb_rm_oof
    mm = np.mean(np.abs(blend - y_true))
    if mm < best_mae:
        best_mae = mm
        best_cfg = f'hard p>0.20 lgb_mae w={w:.2f}'
        best_oof_pred  = blend
        best_test_pred = (1 - mask20_te*w)*best_test + mask20_te*w*lgb_rm_test
        print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search2] Fine-grained w around (p>0.10 v2_lgb_huber)", flush=True)
mask10    = (clf_oof  > 0.10).astype(float)
mask10_te = (clf_test > 0.10).astype(float)
for w in np.arange(0.01, 0.16, 0.01):
    blend = (1 - mask10*w)*best_oof + mask10*w*lgb_rh_oof
    mm = np.mean(np.abs(blend - y_true))
    if mm < best_mae:
        best_mae = mm
        best_cfg = f'hard p>0.10 lgb_huber w={w:.2f}'
        best_oof_pred  = blend
        best_test_pred = (1 - mask10_te*w)*best_test + mask10_te*w*lgb_rh_test
        print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search3] Dual gate: first apply (p>0.10 lgb_huber), then (p>0.20 lgb_mae)", flush=True)
for w1 in np.arange(0.01, 0.11, 0.01):
    base = (1 - mask10*w1)*best_oof + mask10*w1*lgb_rh_oof
    base_te = (1 - mask10_te*w1)*best_test + mask10_te*w1*lgb_rh_test
    for w2 in np.arange(0.01, 0.11, 0.01):
        blend = (1 - mask20*w2)*base + mask20*w2*lgb_rm_oof
        mm = np.mean(np.abs(blend - y_true))
        if mm < best_mae:
            best_mae = mm
            best_cfg = f'dual p10_h_w{w1:.2f}+p20_m_w{w2:.2f}'
            best_oof_pred  = blend
            best_test_pred = (1 - mask20_te*w2)*base_te + mask20_te*w2*lgb_rm_test
            print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

print(f"\n[Search4] Mixed specialist: blend huber+mae first, then gate", flush=True)
for rh in np.arange(0.1, 1.0, 0.1):
    rm = round(1.0 - rh, 1)
    mixed_oof  = rh*lgb_rh_oof  + rm*lgb_rm_oof
    mixed_test = rh*lgb_rh_test + rm*lgb_rm_test
    for p_thresh in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        mask    = (clf_oof  > p_thresh).astype(float)
        mask_te = (clf_test > p_thresh).astype(float)
        for w in np.arange(0.01, 0.11, 0.01):
            blend = (1 - mask*w)*best_oof + mask*w*mixed_oof
            mm = np.mean(np.abs(blend - y_true))
            if mm < best_mae:
                best_mae = mm
                best_cfg = f'mixed rh={rh:.1f} p>{p_thresh:.2f} w={w:.2f}'
                best_oof_pred  = blend
                best_test_pred = (1 - mask_te*w)*best_test + mask_te*w*mixed_test
                print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

if best_cfg:
    print(f"\n=== IMPROVED: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} vs prev_best={prev_best_mae:.5f} ===")
    print(f"Config: {best_cfg}")
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_cascade_refined_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"\nNo improvement over {prev_best_mae:.5f}")
print("Done.")
