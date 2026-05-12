"""
Evaluate cascade blend:
  Final = P_high × specialist + (1-P_high) × current_best
Search optimal alpha (softness of gate) and spec weight.
Auto-saves submission if better than current best.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

# ─── Load alignment ───────────────────────────────────────────────
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

# ─── Current best blend (row_id order) ───────────────────────────
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

# Oracle predictions in row_id order (no te_id2)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

best_oof  = 0.64*fixed_oof  + 0.12*xgb_o  + 0.16*lv2_o  + 0.08*rem_o
best_test = 0.64*fixed_test + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te
curr_mae = np.mean(np.abs(best_oof - y_true))
print(f"Current best OOF: {curr_mae:.5f}", flush=True)
print(f"  y>80 MAE: {np.mean(np.abs(best_oof[y_true>80]-y_true[y_true>80])):.3f}  ({(y_true>80).mean():.3%} of data)", flush=True)

# ─── Cascade components (ls-sorted → row_id via id2/te_id2) ──────
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
spec_oof  = np.load('results/cascade/spec_avg_oof.npy')[id2]
spec_test = np.load('results/cascade/spec_avg_test.npy')[te_id2]

print(f"\nSpecialist avg OOF MAE: {np.mean(np.abs(spec_oof-y_true)):.5f}", flush=True)
print(f"  y>80 MAE: {np.mean(np.abs(spec_oof[y_true>80]-y_true[y_true>80])):.3f}", flush=True)
print(f"Classifier OOF P(high) mean: {clf_oof.mean():.4f}  P(high|y>80): {clf_oof[y_true>80].mean():.4f}", flush=True)

# ─── Grid search: alpha (gate softness) ──────────────────────────
print("\n[Search] P_high^alpha × spec + (1-P_high^alpha) × current_best", flush=True)
print(f"{'alpha':<8} {'w_spec':<8} {'OOF MAE':<12} {'delta':<10} {'y>80 MAE'}")

best_mae = curr_mae
best_cfg = None
best_oof_pred  = best_oof
best_test_pred = best_test

for alpha in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
    gate = np.clip(clf_oof, 0, 1) ** alpha
    gate_te = np.clip(clf_test, 0, 1) ** alpha
    for w_spec in np.arange(0.10, 1.01, 0.10):
        effective_gate = gate * w_spec
        blend = (1 - effective_gate) * best_oof + effective_gate * spec_oof
        mm = np.mean(np.abs(blend - y_true))
        high_mm = np.mean(np.abs(blend[y_true>80] - y_true[y_true>80]))
        if mm < best_mae:
            best_mae = mm
            best_cfg = (alpha, w_spec)
            blend_te = (1 - gate_te*w_spec) * best_test + gate_te*w_spec * spec_test
            best_oof_pred  = blend
            best_test_pred = blend_te
            print(f"★ {alpha:<8.1f} {w_spec:<8.2f} {mm:<12.5f} {mm-curr_mae:<10.5f} {high_mm:.3f}", flush=True)

if best_cfg:
    print(f"\n=== BEST: alpha={best_cfg[0]} w_spec={best_cfg[1]:.2f}  OOF={best_mae:.5f}  delta={best_mae-curr_mae:+.5f} ===")
else:
    print(f"\nNo improvement found. Current best: {curr_mae:.5f}")

# ─── Also try: spec directly in oracle blend ─────────────────────
print("\n[Search2] Direct oracle-style blend: spec as 5th component", flush=True)
for ws in np.arange(0.02, 0.25, 0.02):
    # best_oof = 0.64*fixed + 0.12*xgb + 0.16*lv2 + 0.08*rem
    # try reducing rem or fixed and adding spec
    for base_scale in [0.90, 0.92, 0.94, 0.96, 0.98, 1.0]:
        blend = base_scale * best_oof + ws * (spec_oof - best_oof)
        mm = np.mean(np.abs(blend - y_true))
        if mm < best_mae:
            best_mae = mm
            best_cfg = ('direct', ws, base_scale)
            best_oof_pred = blend
            best_test_pred = base_scale * best_test + ws * (spec_test - best_test)
            print(f"★ direct ws={ws:.2f} scale={base_scale:.2f}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}")

print(f"\n=== FINAL BEST OOF: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} ===")

CURRENT_BEST_LB = 8.3825
if best_mae < CURRENT_BEST_LB - 0.001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_cascade_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname}  (sub stats: mean={sub.mean():.3f} std={sub.std():.3f}) ***")
else:
    print(f"\nOOF {best_mae:.5f} not better enough (threshold: {CURRENT_BEST_LB-0.001:.5f})")
print("Done.")
