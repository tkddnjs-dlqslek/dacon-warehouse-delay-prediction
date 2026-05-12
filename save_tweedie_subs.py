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

# ls → rid mapping for test
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

# Reconstruct oracle_NEW OOF
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values

# Load tw11 (ls order) → convert to rid order
tw11_oof_ls = np.load('results/tweedie/tw11_oof.npy')
tw11_test_ls = np.load('results/tweedie/tw11_test.npy')
tw11_oof = np.clip(tw11_oof_ls[id2], 0, None)
tw11_test = np.clip(tw11_test_ls[te_rid_to_ls], 0, None)

print(f"oracle_NEW: OOF={oracle_mae:.4f}  seen={oracle_new_t[~unseen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print(f"tw11:       OOF={np.mean(np.abs(y_true-tw11_oof)):.4f}  seen={tw11_test[~unseen_mask].mean():.3f}  unseen={tw11_test[unseen_mask].mean():.3f}")
print()

sub_tmpl = pd.read_csv('sample_submission.csv')
saved = []

for w in [0.02, 0.03, 0.05, 0.08]:
    oof_bl = np.clip((1-w)*fw4_oo + w*tw11_oof, 0, None)
    test_bl = np.clip((1-w)*oracle_new_t + w*tw11_test, 0, None)
    bl_mae = np.mean(np.abs(y_true - oof_bl))
    bl_unseen = test_bl[unseen_mask].mean()
    bl_seen = test_bl[~unseen_mask].mean()
    delta_oof = bl_mae - oracle_mae
    delta_u = bl_unseen - oracle_new_t[unseen_mask].mean()

    sub = sub_tmpl.copy()
    sub['avg_delay_minutes_next_30m'] = test_bl
    fname = f"FINAL_oN_tw11_w{int(w*100):02d}_OOF{bl_mae:.4f}.csv"
    sub.to_csv(fname, index=False)
    saved.append((fname, bl_mae, delta_oof, bl_seen, bl_unseen, delta_u))
    print(f"Saved: {fname}")
    print(f"  OOF={bl_mae:.4f} ({delta_oof:+.4f})  seen={bl_seen:.3f}  unseen={bl_unseen:.3f} ({delta_u:+.3f})")

print()
print("=== Summary ===")
print(f"oracle_NEW: OOF={oracle_mae:.4f}, unseen={oracle_new_t[unseen_mask].mean():.3f} (LB=9.7527)")
print()
print("Submission candidates (ascending OOF cost, ascending risk):")
for fname, mae, doof, seen, unseen, du in saved:
    print(f"  {fname}")
    print(f"    OOF={mae:.4f} ({doof:+.4f} vs oracle)  unseen={unseen:.3f} ({du:+.3f})")
    print(f"    Hypothesis: if true unseen > 22.716, this corrects under-prediction")
print()
print("Recommended first submission: w=0.03 (moderate unseen correction, small OOF cost)")
