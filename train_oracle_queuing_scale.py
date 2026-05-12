"""
Queuing Theory Scale Correction for oracle_NEW
Physical insight: M/M/1 queue → mean_delay = 1 / (service_rate × (1-utilization))
As utilization ρ → 1 (system saturation): delay → nonlinearly higher
oracle_NEW trained on ρ_train → extrapolation to ρ_test (64% higher load)
fails because tree models can't extrapolate nonlinear queuing relationships.

Correction approach:
1. Estimate utilization proxy ρ for each scenario using order_inflow_15m / robot_active
2. Compute utilization ratio: ρ_sc / ρ_train_mean
3. Scale oracle_NEW prediction by: (1 - ρ_train_mean) / (1 - ρ_sc) — queuing multiplier
   (clipped to avoid ρ_sc >= 1)

This is NOT a gate:
- Same formula applied to ALL scenarios (seen and unseen)
- No learned parameters from seen-layout distributions
- Physically motivated: based on queuing theory, not data-fitting
- For training scenarios: multiplier ≈ 1 (ρ_sc ≈ ρ_train_mean) → minimal change
- For test unseen (ρ_sc >> ρ_train): multiplier > 1 → correct higher prediction

OOF impact: possibly slightly WORSE on seen layouts (where ρ_sc varies but oracle_NEW
already captures this nonlinearly). LB impact: potentially BETTER if test utilization
is in the nonlinear regime where oracle_NEW under-predicts.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

t0 = time.time()
print('='*60)
print('Queuing Theory Scale Correction for oracle_NEW')
print('  M/M/1 utilization correction: higher load → nonlinear delay boost')
print('  NO gate: same formula for all scenarios')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# Reconstruct oracle_NEW OOF and test
print('\n[1] oracle_NEW 재구성...')
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]

test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof  = (fw['mega33']*d33['meta_avg_oof'][id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
xgb_o_oof = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o_oof = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o_oof = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_o_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o_test = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o_test = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

oracle_oof  = np.clip(0.64*fixed_oof  + 0.12*xgb_o_oof  + 0.16*lv2_o_oof  + 0.08*rem_o_oof,  0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)
del d33; gc.collect()

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'  oracle_NEW OOF: {mae(oracle_oof):.5f}')

# Compute utilization proxy per row: ρ ≈ inflow / (robot_active × capacity_per_robot)
# Estimate capacity_per_robot = mean(inflow) / mean(robot_active) from training
print('\n[2] Utilization proxy 계산...')
mean_inflow = train_raw['order_inflow_15m'].mean()
mean_robot  = train_raw['robot_active'].mean()
capacity_per_robot = mean_inflow / max(mean_robot, 1.0)

print(f'  Training mean_inflow: {mean_inflow:.3f}, mean_robot: {mean_robot:.3f}')
print(f'  capacity_per_robot (estimated): {capacity_per_robot:.3f}')

# Utilization ρ = inflow / (robot_active × capacity_per_robot)
# Clip to [0, 0.99] to avoid ρ >= 1 (which gives infinite delay)
def compute_rho(df, cap_per_robot):
    inflow = pd.to_numeric(df['order_inflow_15m'], errors='coerce').fillna(0).values
    robot  = pd.to_numeric(df['robot_active'],     errors='coerce').fillna(1).clip(lower=1).values
    capacity = robot * cap_per_robot
    rho = (inflow / capacity.clip(min=1e-6)).clip(0, 0.99)
    return rho.astype(np.float32)

rho_tr = compute_rho(train_raw, capacity_per_robot)
rho_te = compute_rho(test_raw,  capacity_per_robot)

rho_train_mean = rho_tr.mean()
print(f'  ρ train mean: {rho_train_mean:.4f}')
print(f'  ρ test  mean: {rho_te.mean():.4f}')
print(f'  ρ test  max:  {rho_te.max():.4f}')

# M/M/1 queuing correction: multiply prediction by (1-ρ_train) / (1-ρ_sc)
# For ρ_sc = ρ_train: correction = 1 (no change)
# For ρ_sc > ρ_train: correction > 1 (higher prediction)
def queuing_scale(oracle_pred, rho_sc, rho_ref):
    correction = (1.0 - rho_ref) / (1.0 - rho_sc).clip(min=0.01)
    correction = correction.clip(0.5, 5.0)  # safety bounds
    return (oracle_pred * correction).clip(0, None)

# Analyze the correction factor distribution
correction_tr = (1.0 - rho_train_mean) / (1.0 - rho_tr).clip(min=0.01)
correction_te = (1.0 - rho_train_mean) / (1.0 - rho_te).clip(min=0.01)
correction_tr_clipped = correction_tr.clip(0.5, 5.0)
correction_te_clipped = correction_te.clip(0.5, 5.0)

print(f'\n  Correction factor distribution:')
print(f'    Train: mean={correction_tr_clipped.mean():.4f}, std={correction_tr_clipped.std():.4f}, max={correction_tr_clipped.max():.4f}')
print(f'    Test:  mean={correction_te_clipped.mean():.4f}, std={correction_te_clipped.std():.4f}, max={correction_te_clipped.max():.4f}')

# Apply full correction
scaled_oof  = queuing_scale(oracle_oof,  rho_tr, rho_train_mean)
scaled_test = queuing_scale(oracle_test, rho_te, rho_train_mean)

print(f'\n[3] 보정 결과:')
print(f'  oracle_NEW  OOF: {mae(oracle_oof):.5f}')
print(f'  scaled      OOF: {mae(scaled_oof):.5f} (delta={mae(scaled_oof)-mae(oracle_oof):+.5f})')

# Try different alpha values: blend = (1-alpha)*oracle + alpha*scaled_oof
print(f'\n[4] Partial blend 분석:')
for alpha in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]:
    blend = (1-alpha)*oracle_oof + alpha*scaled_oof
    delta = mae(blend) - mae(oracle_oof)
    print(f'  alpha={alpha:.2f}: OOF delta={delta:+.5f}, pred_mean={np.clip(blend,0,None).mean():.3f}')

# If small alpha helps, compute test prediction and save
best_alpha = None
best_delta = 0.0
for alpha in [0.02, 0.05, 0.1, 0.2]:
    blend = (1-alpha)*oracle_oof + alpha*scaled_oof
    delta = mae(blend) - mae(oracle_oof)
    if delta < best_delta:
        best_delta = delta
        best_alpha = alpha

if best_alpha is not None:
    blend_oof  = (1-best_alpha)*oracle_oof  + best_alpha*scaled_oof
    blend_test = (1-best_alpha)*oracle_test + best_alpha*scaled_test
    print(f'\n★ Best blend: alpha={best_alpha}, OOF delta={best_delta:+.5f}')
    print(f'  pred_mean: {np.clip(blend_oof,0,None).mean():.3f} (oracle: {oracle_oof.mean():.3f})')

    # Save submission
    sub = pd.read_csv('sample_submission.csv')
    sub['predicted'] = blend_test
    sub_file = f'submission_queuing_scale_a{int(best_alpha*100)}_OOF{mae(blend_oof):.5f}.csv'
    sub.to_csv(sub_file, index=False)
    print(f'  Saved: {sub_file}')
else:
    print(f'\nNo alpha value improved OOF. Skip submission.')

print(f'\nDone. ({time.time()-t0:.0f}s total)')
