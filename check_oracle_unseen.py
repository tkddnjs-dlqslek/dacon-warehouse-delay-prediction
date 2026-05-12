import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, glob

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof=np.clip(d33['meta_oofs']['cb'][id2],0,None); cb_test=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]; rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]
xgbc_o2=xgbc_o; mono_o2=mono_o

# ============================================================
print("="*70)
print("Oracle model predictions for seen vs unseen")
print("="*70)

oracle_models = {
    'xgb':  (xgb_o, xgb_t),
    'lv2':  (lv2_o, lv2_t),
    'rem':  (rem_o, rem_t),
    'xgbc': (xgbc_o, xgbc_t),
    'mono': (mono_o, mono_t),
}
print(f"\n{'Model':8s}  {'OOF':>9}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
print("-"*55)
for name, (oof, t) in oracle_models.items():
    oof_c = np.clip(oof, 0, None)
    t_c   = np.clip(t, 0, None)
    print(f"  {name:8s}  {mae_fn(oof_c):9.5f}  {t_c.mean():8.3f}  {t_c[seen_mask].mean():8.3f}  {t_c[unseen_mask].mean():8.3f}")

# Cascade models
print(f"\n  rh:       --------  {np.clip(rh_t,0,None).mean():8.3f}  {np.clip(rh_t,0,None)[seen_mask].mean():8.3f}  {np.clip(rh_t,0,None)[unseen_mask].mean():8.3f}")
print(f"  slhm:     --------  {np.clip(slhm_t,0,None).mean():8.3f}  {np.clip(slhm_t,0,None)[seen_mask].mean():8.3f}  {np.clip(slhm_t,0,None)[unseen_mask].mean():8.3f}")

# Look at oracle blends
print(f"\n{'Blend':30s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
print("-"*60)
oracle_blends = {
    'xgb_only':             np.clip(xgb_t, 0, None),
    'lv2_only':             np.clip(lv2_t, 0, None),
    'xgb+lv2 (equal)':     np.clip(0.5*xgb_t+0.5*lv2_t, 0, None),
    'xgb+lv2+rem (equal)': np.clip((xgb_t+lv2_t+rem_t)/3, 0, None),
    'all5 equal':           np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None),
    'xgbc_only':            np.clip(xgbc_t, 0, None),
    'mono_only':            np.clip(mono_t, 0, None),
}
for name, ct in oracle_blends.items():
    print(f"  {name:30s}  {ct.mean():8.3f}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("What oracle_NEW might be — examining existing saved submission files")
print("="*70)
# Check all saved CSV files in current directory
csv_files = sorted(glob.glob('*.csv'))
final_csvs = [f for f in csv_files if f.startswith('FINAL') or 'oracle' in f.lower()]
print(f"\nSaved submission CSVs ({len(final_csvs)} files):")

sample_sub = pd.read_csv('sample_submission.csv')
id_order = test_raw['ID'].values

final_stats = []
for fname in final_csvs:
    try:
        df = pd.read_csv(fname)
        # align to test_raw order
        df = df.set_index('ID').reindex(id_order).reset_index()
        preds = df['avg_delay_minutes_next_30m'].values
        final_stats.append((fname, preds.mean(), preds[seen_mask].mean(), preds[unseen_mask].mean()))
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")

final_stats.sort(key=lambda x: x[3])  # sort by unseen mean
print(f"\n{'Filename':65s}  {'test':>8}  {'seen':>8}  {'unseen':>8}")
for fname, tm, sm, um in final_stats:
    print(f"  {fname[:63]:63s}  {tm:8.3f}  {sm:8.3f}  {um:8.3f}")

# ============================================================
print("\n" + "="*70)
print("Key oracle models OOF performance")
print("="*70)

# Full rebuild to get proper OOF for mega blend
fw2 = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega_oof=(1-0.25)*mega33_oof+0.25*mega34_oof
wm2=fw2['mega33']-(-0.04)-(-0.02); w2_=fw2['iter_r2']+(-0.04); w3_=fw2['iter_r3']+(-0.02)
fx_o=wm2*mega_oof+fw2['rank_adj']*rank_oof+fw2['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
slh_o2=slh_o; rh_o2=rh_o; slhm_o2=slhm_o
w_rem=1-0.72; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
bb_oo=np.clip(0.72*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem*rem_o,0,None)
bb_oo=np.clip((1-0.12)*bb_oo+0.12*cb_oof,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o2+0.10*xgbc_o2+0.08*mono_o2,0,None)

# oracle_NEW style: no gate, pure blend with oracle models at different weights
print(f"\nPure oracle-blend OOF tests:")
for wxgb2, wlv2_2, wrem2, wname in [
    (0.0, 0.0, 0.0, 'no_oracle'),
    (0.10, 0.0, 0.0, 'xgb_0.10'),
    (0.0, 0.10, 0.0, 'lv2_0.10'),
    (0.05, 0.05, 0.0, 'xgb+lv2_0.05each'),
    (0.12*0.28/0.36, 0.16*0.28/0.36, 0.08*0.28/0.36, 'current_split'),
]:
    bb = np.clip(0.72*fx_o + wxgb2*xgb_o + wlv2_2*lv2_o + wrem2*rem_o, 0, None)
    bb = np.clip((1-0.12)*bb + 0.12*cb_oof, 0, None)
    blend = np.clip(0.74*bb + 0.08*slh_o2 + 0.10*xgbc_o2 + 0.08*mono_o2, 0, None)
    print(f"  {wname:25s}: OOF={mae_fn(blend):.5f}")

print("\nDone.")
