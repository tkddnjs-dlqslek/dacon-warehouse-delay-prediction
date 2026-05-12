"""
Position-aware blend evaluation.
Different oracle weights for early (pos 0-5) vs late (pos 20-24) positions.
oracle-Remaining is most accurate early; oracle-SC is uniform across positions.
Run after oracle results are available.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
y_true = train_raw['avg_delay_minutes_next_30m'].values
pos = train_raw['row_in_sc'].values

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_to_pos[i] for i in train_raw['ID'].values]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed2 = (fw['mega33']*d['meta_avg_oof'][id2]
         + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
         + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
         + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
         + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_mae = np.mean(np.abs(fixed2 - y_true))

oracle_xgb = np.load('results/oracle_seq/oof_seqC_xgb.npy')
oracle_lv2 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
current_best = 0.68*fixed2 + 0.12*oracle_xgb + 0.20*oracle_lv2
print(f"FIXED MAE: {fixed_mae:.4f}")
print(f"current_best (0.68F+0.12X+0.20L): {np.mean(np.abs(current_best-y_true)):.4f}")

# Analyze MAE per position for each model
print("\n=== MAE by position group ===")
groups = [(range(0,5), 'early[0-4]'), (range(5,15), 'mid[5-14]'), (range(15,25), 'late[15-24]')]
for rng, label in groups:
    m = np.isin(pos, list(rng))
    f_mae = np.mean(np.abs(fixed2[m] - y_true[m]))
    x_mae = np.mean(np.abs(oracle_xgb[m] - y_true[m]))
    l_mae = np.mean(np.abs(oracle_lv2[m] - y_true[m]))
    c_mae = np.mean(np.abs(current_best[m] - y_true[m]))
    print(f"  {label}: n={m.sum()} FIXED={f_mae:.4f} xgb={x_mae:.4f} lv2={l_mae:.4f} best={c_mae:.4f}")

# Also check if remaining oracle is available
if os.path.exists('results/oracle_seq/oof_seqC_xgb_remaining.npy'):
    oracle_rem = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
    r_mae = np.mean(np.abs(oracle_rem - y_true))
    print(f"\noracle-Remaining OOF MAE: {r_mae:.4f}")
    for rng, label in groups:
        m = np.isin(pos, list(rng))
        r_mae_g = np.mean(np.abs(oracle_rem[m] - y_true[m]))
        print(f"  {label}: remaining={r_mae_g:.4f}")

if os.path.exists('results/oracle_seq/oof_seqC_xgb_sc_only.npy'):
    oracle_sc = np.load('results/oracle_seq/oof_seqC_xgb_sc_only.npy')
    sc_mae = np.mean(np.abs(oracle_sc - y_true))
    print(f"\noracle-SC-only OOF MAE: {sc_mae:.4f}")
    for rng, label in groups:
        m = np.isin(pos, list(rng))
        sc_mae_g = np.mean(np.abs(oracle_sc[m] - y_true[m]))
        print(f"  {label}: sc_only={sc_mae_g:.4f}")

# Position-aware blend: use different weights per position group
print("\n=== Position-aware blending ===")
available_oracles = {'xgb': oracle_xgb, 'lv2': oracle_lv2}
if os.path.exists('results/oracle_seq/oof_seqC_xgb_remaining.npy'):
    available_oracles['rem'] = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
if os.path.exists('results/oracle_seq/oof_seqC_xgb_sc_only.npy'):
    available_oracles['sc']  = np.load('results/oracle_seq/oof_seqC_xgb_sc_only.npy')

if len(available_oracles) >= 2:
    best_pos_mae = np.mean(np.abs(current_best - y_true))
    best_pos_cfg = None
    # Simple: optimize 3 position groups independently, 2 oracle weights
    pos_early = np.isin(pos, list(range(0, 8)))
    pos_mid   = np.isin(pos, list(range(8, 18)))
    pos_late  = np.isin(pos, list(range(18, 25)))

    for wx in np.arange(0, 0.25, 0.04):
        for wl in np.arange(0, 0.35, 0.04):
            if wx + wl > 0.5: continue
            blend = (1-wx-wl)*fixed2 + wx*oracle_xgb + wl*oracle_lv2
            mm = np.mean(np.abs(blend - y_true))
            if mm < best_pos_mae:
                best_pos_mae = mm
                best_pos_cfg = (wx, wl)

    if best_pos_cfg:
        wx, wl = best_pos_cfg
        blend = (1-wx-wl)*fixed2 + wx*oracle_xgb + wl*oracle_lv2
        print(f"  Best global: xgb={wx:.2f} lv2={wl:.2f} fixed={1-wx-wl:.2f} MAE={best_pos_mae:.4f}")

print("\nDone.")
