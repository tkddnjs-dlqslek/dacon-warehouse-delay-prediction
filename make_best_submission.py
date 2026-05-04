"""
Generate submission from best oracle blend.
Run after oracle training completes. Finds optimal blend and saves submission.
All test arrays in _row_id order (test_raw sorted by _row_id).
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
test_raw['row_in_sc'] = test_raw.groupby(['layout_id','scenario_id']).cumcount()

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

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

fixed_mae = np.mean(np.abs(fixed_oof - y_true))
print(f"FIXED OOF MAE: {fixed_mae:.4f}")

# Load all available oracle models (OOF + test, both in _row_id order)
ORACLES = {
    'xgb':       ('results/oracle_seq/oof_seqC_xgb.npy',
                  'results/oracle_seq/test_C_xgb.npy'),
    'lv2':       ('results/oracle_seq/oof_seqC_log_v2.npy',
                  'results/oracle_seq/test_C_log_v2.npy'),
    'remaining': ('results/oracle_seq/oof_seqC_xgb_remaining.npy',
                  'results/oracle_seq/test_C_xgb_remaining.npy'),
    'sc_only':   ('results/oracle_seq/oof_seqC_xgb_sc_only.npy',
                  'results/oracle_seq/test_C_xgb_sc_only.npy'),
    'residual':  ('results/oracle_seq/oof_seqC_xgb_residual.npy',
                  'results/oracle_seq/test_C_xgb_residual.npy'),
    'layout':    ('results/oracle_seq/oof_seqC_xgb_layout.npy',
                  'results/oracle_seq/test_C_xgb_layout.npy'),
    'layout_v2': ('results/oracle_seq/oof_seqC_xgb_layout_v2.npy',
                  'results/oracle_seq/test_C_xgb_layout_v2.npy'),
    'lgb_sc':    ('results/oracle_seq/oof_seqC_lgb_sc_only.npy',
                  'results/oracle_seq/test_C_lgb_sc_only.npy'),
    'cumulative':('results/oracle_seq/oof_seqC_xgb_cumulative.npy',
                  'results/oracle_seq/test_C_xgb_cumulative.npy'),
    'stack':     ('results/oracle_seq/oof_seqC_lgb_stack.npy',
                  'results/oracle_seq/test_C_lgb_stack.npy'),
    'dual':      ('results/oracle_seq/oof_seqC_lgb_dual.npy',
                  'results/oracle_seq/test_C_lgb_dual.npy'),
    'lgb_rem':   ('results/oracle_seq/oof_seqC_lgb_remaining.npy',
                  'results/oracle_seq/test_C_lgb_remaining.npy'),
    'lgb_rem_v3':('results/oracle_seq/oof_seqC_lgb_remaining_v3.npy',
                  'results/oracle_seq/test_C_lgb_remaining_v3.npy'),
    'latepos':   ('results/oracle_seq/oof_seqC_lgb_latepos.npy',
                  'results/oracle_seq/test_C_lgb_latepos.npy'),
    'lgb_log1':  ('results/oracle_seq/oof_seqC_lgb_log1.npy',
                  'results/oracle_seq/test_C_lgb_log1.npy'),
    'xgb_lag1':  ('results/oracle_seq/oof_seqC_xgb_lag1.npy',
                  'results/oracle_seq/test_C_xgb_lag1.npy'),
    'combined':  ('results/oracle_seq/oof_seqC_xgb_combined.npy',
                  'results/oracle_seq/test_C_xgb_combined.npy'),
    'bestproxy': ('results/oracle_seq/oof_seqC_xgb_bestproxy.npy',
                  'results/oracle_seq/test_C_xgb_bestproxy.npy'),
    'xgb_v31':   ('results/oracle_seq/oof_seqC_xgb_v31.npy',
                  'results/oracle_seq/test_C_xgb_v31.npy'),
    'cb':        ('results/oracle_seq/oof_seqC_cb.npy',
                  'results/oracle_seq/test_C_cb.npy'),
}

oof_models = {}
test_models = {}
for name, (oof_path, test_path) in ORACLES.items():
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof_arr = np.load(oof_path)
        test_arr = np.load(test_path)
        mae = np.mean(np.abs(oof_arr - y_true))
        corr_fixed = np.corrcoef(oof_arr, fixed_oof)[0,1]
        oof_models[name] = oof_arr
        test_models[name] = test_arr
        print(f"  {name:12s}: OOF={mae:.4f}  corr_fixed={corr_fixed:.4f}")

# Current best: FIXED×0.64 + xgb×0.12 + lv2×0.16 + remaining×0.08 (LB 9.7527)
_rem = oof_models.get('remaining', np.zeros(len(y_true)))
_rem_te = test_models.get('remaining', np.zeros(len(test_raw)))
current_oof  = 0.64*fixed_oof  + 0.12*oof_models.get('xgb',0)  + 0.16*oof_models.get('lv2',0)  + 0.08*_rem
current_test = 0.64*fixed_test + 0.12*test_models.get('xgb',0) + 0.16*test_models.get('lv2',0) + 0.08*_rem_te
current_mae = np.mean(np.abs(current_oof - y_true))
print(f"\nCurrent best (0.64F+0.12X+0.16L+0.08R): OOF={current_mae:.4f}")

# 4-way search: FIXED + xgb + lv2 + one oracle (includes remaining as candidate)
print("\n=== Search 4-way: FIXED + xgb + lv2 + oracle ===")
best_overall_mae = current_mae
best_overall_oof = current_oof
best_overall_test = current_test
best_overall_label = 'current'

for n1, o1 in oof_models.items():
    if n1 in ('xgb', 'lv2'): continue
    for wx in np.arange(0.08, 0.21, 0.04):
        for wl in np.arange(0.08, 0.25, 0.04):
            for wn in np.arange(0.02, 0.17, 0.02):
                if wx+wl+wn > 0.58: continue
                blend = (1-wx-wl-wn)*fixed_oof + wx*oof_models['xgb'] + wl*oof_models['lv2'] + wn*o1
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_overall_mae:
                    best_overall_mae = mm
                    best_overall_label = f'F×{1-wx-wl-wn:.2f}+xgb×{wx:.2f}+lv2×{wl:.2f}+{n1}×{wn:.2f}'
                    best_overall_oof = blend
                    best_overall_test = (1-wx-wl-wn)*fixed_test + wx*test_models['xgb'] + wl*test_models['lv2'] + wn*test_models[n1]

print(f"Best 4-way: {best_overall_label} MAE={best_overall_mae:.4f} delta={best_overall_mae-current_mae:+.4f}")

# 5-way search: FIXED + xgb + lv2 + remaining + new oracle
if 'remaining' in oof_models:
    print("\n=== Search 5-way: FIXED + xgb + lv2 + remaining + new oracle ===")
    for n1, o1 in oof_models.items():
        if n1 in ('xgb', 'lv2', 'remaining'): continue
        for wx in np.arange(0.08, 0.17, 0.04):
            for wl in np.arange(0.08, 0.21, 0.04):
                for wr in np.arange(0.04, 0.13, 0.02):
                    for wn in np.arange(0.02, 0.11, 0.02):
                        if wx+wl+wr+wn > 0.56: continue
                        blend = ((1-wx-wl-wr-wn)*fixed_oof + wx*oof_models['xgb']
                                 + wl*oof_models['lv2'] + wr*_rem + wn*o1)
                        mm = np.mean(np.abs(blend - y_true))
                        if mm < best_overall_mae:
                            best_overall_mae = mm
                            best_overall_label = f'5way:F×{1-wx-wl-wr-wn:.2f}+xgb×{wx:.2f}+lv2×{wl:.2f}+rem×{wr:.2f}+{n1}×{wn:.2f}'
                            best_overall_oof = blend
                            best_overall_test = ((1-wx-wl-wr-wn)*fixed_test + wx*test_models['xgb']
                                                 + wl*test_models['lv2'] + wr*_rem_te + wn*test_models[n1])
    print(f"Best 5-way: {best_overall_label} MAE={best_overall_mae:.4f}")

print(f"\n==> Overall best: {best_overall_label}  OOF={best_overall_mae:.4f} delta={best_overall_mae-current_mae:+.4f}")

CURRENT_BEST_OOF = 8.3825  # threshold: only save if clearly better (LB 9.7527)
if best_overall_mae < CURRENT_BEST_OOF - 0.0005:
    sample_sub = pd.read_csv('sample_submission.csv')
    test_b = np.maximum(0, best_overall_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': test_b})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_oracle_NEW_OOF{best_overall_mae:.4f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} (improvement: {best_overall_mae-CURRENT_BEST_OOF:+.4f}) ***")
else:
    print(f"No meaningful improvement ({best_overall_mae:.4f} vs threshold {CURRENT_BEST_OOF:.4f}). Not saved.")
print("Done.")
