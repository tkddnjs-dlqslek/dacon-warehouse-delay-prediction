"""
Hard gate cascade eval.
Issue with soft gate: P(high|y>80)=0.1864 → effective blend <1%.
Hard gate: if clf_oof > P_threshold, apply specialist weight directly.
Also tries: exclude cb_raw (high=98.21 — worse than baseline) from spec avg.
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
print(f"Current best OOF: {curr_mae:.5f}  y>80 MAE: {np.mean(np.abs(best_oof[y_true>80]-y_true[y_true>80])):.3f}")

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]

# Build specialist options
# v1: log1p, 30x (high=79.27)
spec_v1_oof  = np.load('results/cascade/spec_avg_oof.npy')[id2]
spec_v1_test = np.load('results/cascade/spec_avg_test.npy')[te_id2]

# v2 components (raw-y, 100x)
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]   # high=42.08
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]     # high=52.92
cb_raw_oof  = np.load('results/cascade/spec_cb_raw_oof.npy')[id2]          # high=98.21
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]
cb_raw_test = np.load('results/cascade/spec_cb_raw_test.npy')[te_id2]

# Best v2 combo: exclude cb_raw, average lgb_rh + lgb_rm only
spec_v2_best_oof  = 0.5*lgb_rh_oof  + 0.5*lgb_rm_oof
spec_v2_best_test = 0.5*lgb_rh_test + 0.5*lgb_rm_test

print(f"\nSpecialist options (y>80 MAE):")
for name, arr in [('v1_avg(high=79.27)', spec_v1_oof),
                   ('v2_lgb_huber(high=42.08)', lgb_rh_oof),
                   ('v2_lgb_mae(high=52.92)', lgb_rm_oof),
                   ('v2_lgb_avg(excl_cb)', spec_v2_best_oof)]:
    print(f"  {name}: all={np.mean(np.abs(arr-y_true)):.4f}  y>80={np.mean(np.abs(arr[y_true>80]-y_true[y_true>80])):.3f}")

print(f"\nClassifier: P(high)_mean={clf_oof.mean():.4f}  P(high|y>80)_mean={clf_oof[y_true>80].mean():.4f}")
for pct in [50,75,90,95,99]:
    thresh = np.percentile(clf_oof, pct)
    n_above = (clf_oof > thresh).sum()
    n_correct = ((clf_oof > thresh) & (y_true > 80)).sum()
    recall = ((clf_oof > thresh) & (y_true > 80)).sum() / (y_true > 80).sum()
    precision = n_correct / max(n_above, 1)
    print(f"  P>{thresh:.4f} (top {100-pct}%): n={n_above}  recall={recall:.3f}  precision={precision:.3f}")

best_mae = curr_mae
best_cfg = None
best_oof_pred  = best_oof.copy()
best_test_pred = best_test.copy()

print("\n[Search1] HARD GATE: if clf > threshold, blend specialist", flush=True)
for spec_name, spec_o, spec_t in [
    ('v2_lgb_huber', lgb_rh_oof, lgb_rh_test),
    ('v2_lgb_mae',   lgb_rm_oof, lgb_rm_test),
    ('v2_lgb_avg',   spec_v2_best_oof, spec_v2_best_test),
    ('v1_avg',       spec_v1_oof, spec_v1_test),
]:
    for p_thresh in [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20]:
        hard_mask    = (clf_oof  > p_thresh).astype(float)
        hard_mask_te = (clf_test > p_thresh).astype(float)
        for w in np.arange(0.05, 0.81, 0.05):
            blend = (1 - hard_mask*w)*best_oof + hard_mask*w*spec_o
            mm = np.mean(np.abs(blend - y_true))
            if mm < best_mae:
                best_mae = mm
                best_cfg = f'hard p>{p_thresh:.2f} {spec_name} w={w:.2f}'
                best_oof_pred  = blend
                best_test_pred = (1 - hard_mask_te*w)*best_test + hard_mask_te*w*spec_t
                high_mm = np.mean(np.abs(blend[y_true>80]-y_true[y_true>80]))
                print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}  y>80={high_mm:.3f}", flush=True)

print("\n[Search2] SOFT GATE with v2_lgb_avg (best spec)", flush=True)
for alpha in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
    gate    = np.clip(clf_oof, 0, 1)**alpha
    gate_te = np.clip(clf_test, 0, 1)**alpha
    for w in np.arange(0.05, 1.01, 0.05):
        blend = (1 - gate*w)*best_oof + gate*w*spec_v2_best_oof
        mm = np.mean(np.abs(blend - y_true))
        if mm < best_mae:
            best_mae = mm
            best_cfg = f'soft alpha={alpha} v2_lgb_avg w={w:.2f}'
            best_oof_pred  = blend
            best_test_pred = (1 - gate_te*w)*best_test + gate_te*w*spec_v2_best_test
            high_mm = np.mean(np.abs(blend[y_true>80]-y_true[y_true>80]))
            print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}  y>80={high_mm:.3f}", flush=True)

print(f"\n=== FINAL: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} ===")
print(f"Config: {best_cfg}")

CURRENT_BEST_LB = 8.3825
if best_mae < CURRENT_BEST_LB - 0.001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_cascade_hardgate_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} ***")
else:
    print(f"\nOOF {best_mae:.5f} not better enough (threshold: {CURRENT_BEST_LB-0.001:.5f})")
print("Done.")
