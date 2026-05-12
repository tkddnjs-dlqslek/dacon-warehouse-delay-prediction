"""
Evaluate pack-station bottleneck specialist blend.
Strategy: replace/blend predictions for bottleneck layouts (pack_station_count<=2).
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

# Current best blend
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

# Pack specialist (ls-sorted → row_id via id2)
pack_oof  = np.load('results/pack_spec/pack_avg_oof.npy')[id2]
pack_test = np.load('results/pack_spec/pack_avg_test.npy')[te_id2]
is_bot_tr = np.load('results/pack_spec/is_bottleneck_tr.npy')[id2]  # row_id order
is_bot_te = np.load('results/pack_spec/is_bottleneck_te.npy')[te_id2]

print(f"\nPack specialist OOF MAE: {np.mean(np.abs(pack_oof-y_true)):.5f}")
print(f"  bottleneck MAE: {np.mean(np.abs(pack_oof[is_bot_tr==1]-y_true[is_bot_tr==1])):.4f}  (current: {np.mean(np.abs(best_oof[is_bot_tr==1]-y_true[is_bot_tr==1])):.4f})")
print(f"  non-bottleneck MAE: {np.mean(np.abs(pack_oof[is_bot_tr==0]-y_true[is_bot_tr==0])):.4f}  (current: {np.mean(np.abs(best_oof[is_bot_tr==0]-y_true[is_bot_tr==0])):.4f})")

best_mae = curr_mae
best_cfg = None
best_oof_pred  = best_oof.copy()
best_test_pred = best_test.copy()

print("\n[Search1] Soft blend: w_bot × specialist + (1-w_bot) × current_best (bottleneck only)", flush=True)
for w_bot in np.arange(0.05, 1.01, 0.05):
    blend_oof  = best_oof.copy()
    blend_test = best_test.copy()
    blend_oof[is_bot_tr==1]  = (1-w_bot)*best_oof[is_bot_tr==1]  + w_bot*pack_oof[is_bot_tr==1]
    blend_test[is_bot_te==1] = (1-w_bot)*best_test[is_bot_te==1] + w_bot*pack_test[is_bot_te==1]
    mm = np.mean(np.abs(blend_oof - y_true))
    if mm < best_mae:
        best_mae = mm
        best_cfg = f'bottleneck_only w={w_bot:.2f}'
        best_oof_pred  = blend_oof.copy()
        best_test_pred = blend_test.copy()
        bot_mm = np.mean(np.abs(blend_oof[is_bot_tr==1]-y_true[is_bot_tr==1]))
        print(f"★ w={w_bot:.2f}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}  bot_MAE={bot_mm:.4f}", flush=True)

print("\n[Search2] Global soft blend (all rows)", flush=True)
for w in np.arange(0.02, 0.31, 0.02):
    blend = (1-w)*best_oof + w*pack_oof
    mm = np.mean(np.abs(blend - y_true))
    if mm < best_mae:
        best_mae = mm
        best_cfg = f'global w={w:.2f}'
        best_oof_pred  = blend
        best_test_pred = (1-w)*best_test + w*pack_test
        print(f"★ w={w:.2f}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)

# Try combined with cascade if available
cascade_path = 'results/cascade/clf_oof.npy'
if os.path.exists(cascade_path):
    for spec_path in ['results/cascade/spec_v2_avg_oof.npy', 'results/cascade/spec_avg_oof.npy']:
        if os.path.exists(spec_path):
            print(f"\n[Search3] Pack + Cascade combined ({spec_path})", flush=True)
            clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
            clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
            casc_oof  = np.load(spec_path)[id2]
            casc_test_p = spec_path.replace('oof','test')
            casc_test = np.load(casc_test_p)[te_id2]
            for w_bot in [0.2, 0.3, 0.4, 0.5]:
                for alpha in [0.5, 1.0, 1.5]:
                    gate    = np.clip(clf_oof, 0, 1)**alpha
                    gate_te = np.clip(clf_test, 0, 1)**alpha
                    for w_casc in [0.1, 0.2, 0.3]:
                        # First apply cascade gate, then apply pack blend on bottleneck
                        blend = (1-gate*w_casc)*best_oof + gate*w_casc*casc_oof
                        blend[is_bot_tr==1] = (1-w_bot)*blend[is_bot_tr==1] + w_bot*pack_oof[is_bot_tr==1]
                        mm = np.mean(np.abs(blend - y_true))
                        if mm < best_mae:
                            best_mae = mm
                            best_cfg = f'casc(a={alpha},w={w_casc})+pack(w={w_bot})'
                            te_blend = (1-gate_te*w_casc)*best_test + gate_te*w_casc*casc_test
                            te_blend[is_bot_te==1] = (1-w_bot)*te_blend[is_bot_te==1] + w_bot*pack_test[is_bot_te==1]
                            best_oof_pred  = blend
                            best_test_pred = te_blend
                            print(f"★ {best_cfg}  OOF={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)
            break

print(f"\n=== FINAL: {best_mae:.5f}  delta={best_mae-curr_mae:+.5f} ===")
print(f"Config: {best_cfg}")

CURRENT_BEST_LB = 8.3825
if best_mae < CURRENT_BEST_LB - 0.001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_test_pred)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_pack_spec_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} ***")
else:
    print(f"\nOOF {best_mae:.5f} not better enough (threshold: {CURRENT_BEST_LB-0.001:.5f})")
print("Done.")
