"""
w34 sweep: change mega34 weight in oracle_NEW and measure OOF + test delta.
Also computes correct test predictions by adding the mega34 delta to oracle_new_t.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

# OOF in rid order
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
# Test in rid order (mega33/34 are in ls order)
mega33_t = d33['meta_avg_test'][te_rid_to_ls]
mega34_t = d34['meta_avg_test'][te_rid_to_ls]

print(f"mega33 test_unseen: {mega33_t[unseen_mask].mean():.3f}")
print(f"mega34 test_unseen: {mega34_t[unseen_mask].mean():.3f}")
print(f"mega33 OOF MAE: {np.mean(np.abs(y_true - np.clip(mega33_oof,0,None))):.4f}")
print(f"mega34 OOF MAE: {np.mean(np.abs(y_true - np.clip(mega34_oof,0,None))):.4f}")
print()

# Reconstruct base oracle_NEW OOF fully
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen = oracle_new_t[unseen_mask].mean()

# Oracle_NEW OOF at w34=0.25 (baseline)
def compute_oracle_oof(w34_val):
    dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
    mega_oof = (1-w34_val)*mega33_oof + w34_val*mega34_oof
    wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
    fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
    w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
    bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
    bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
    return np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)

fw4_base = compute_oracle_oof(0.25)
oracle_mae = np.mean(np.abs(y_true - fw4_base))
print(f"oracle_NEW: OOF={oracle_mae:.4f}  test_unseen={oracle_unseen:.3f}")
print()

# The oracle_NEW test = f(mega33_t, mega34_t, ...) at w34=0.25.
# When we change w34, the test changes by: delta_t = (new_w34 - 0.25) * (mega34_t - mega33_t)
# This is the linear approximation (exact since mega blend is linear in the blend)
mega_delta_t = mega34_t - mega33_t  # delta if we shift w34 by 1 unit
print(f"mega34-mega33 delta_t: mean={mega_delta_t.mean():.3f}  unseen_mean={mega_delta_t[unseen_mask].mean():.3f}")
print()

print("=== w34 sweep: OOF and test_unseen change ===")
print(f"{'w34':>6} {'OOF':>10} {'ΔOOF':>8} {'t_unseen':>10} {'Δu':>7}")
print("-"*50)

sub_tmpl = pd.read_csv('sample_submission.csv')

for w34_val in [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 1.0]:
    fw4_oo = compute_oracle_oof(w34_val)
    mae = np.mean(np.abs(y_true - fw4_oo))

    # Test: oracle_new_t + linear delta from changing w34
    # oracle_new_t was built at w34=0.25
    # New: oracle_new_t + (w34_val - 0.25) * mega_delta_t_component
    # But need to trace through full oracle_NEW pipeline...
    # Approximate: the mega_oof contribution to test is scaled by (w34_val-0.25)*(mega34_t-mega33_t)
    # Full factor: through fx_o → bb_oo → fw4_t
    # Approximate scaling: wm_effective ≈ 0.763 * 0.72 * 0.74 ≈ 0.407
    # So delta_t_unseen ≈ 0.407 * (w34_val - 0.25) * (mega34_t - mega33_t)[unseen].mean()
    delta_w34 = w34_val - 0.25
    approx_t_unseen = oracle_unseen + delta_w34 * mega_delta_t[unseen_mask].mean() * 0.407

    print(f"  {w34_val:6.2f}  {mae:10.4f}  {mae-oracle_mae:+8.4f}  {approx_t_unseen:10.3f}  {approx_t_unseen-oracle_unseen:+7.3f}")

print()
print("Exact delta_t requires full pipeline reconstruction with correct test components.")
print("Only OOF changes are exact above.")
print()

# Key: w34=0.30 gives -0.00002 OOF improvement
# Save w34=0.30 if it truly improves OOF
w34_best_val = 0.30
fw4_best = compute_oracle_oof(w34_best_val)
mae_best = np.mean(np.abs(y_true - fw4_best))
print(f"w34={w34_best_val}: OOF={mae_best:.4f} vs oracle {oracle_mae:.4f} = {mae_best-oracle_mae:+.5f}")
print()

# Check what components CAN fully reconstruct oracle_NEW test predictions
# by comparing against oracle_new_t
print("=== Verifying if mega33/34 test is in correct order ===")
# oracle_new_t unseen mean = 22.716
# mega33_t unseen = 22.974
# mega34_t unseen = 23.011
# The blend at w34=0.25: 0.75*22.974 + 0.25*23.011 = 22.983
# But final pipeline should give 22.716 (lower due to oracle_seq pulling it down)
print(f"Weighted mega blend (w34=0.25) test_unseen: {(0.75*mega33_t + 0.25*mega34_t)[unseen_mask].mean():.3f}")
print(f"oracle_new_t actual: {oracle_unseen:.3f}")
print(f"Difference: {oracle_unseen - (0.75*mega33_t + 0.25*mega34_t)[unseen_mask].mean():.3f}")
print("(oracle_seq components pull it down by ~0.27)")
