import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, glob, os
import warnings; warnings.filterwarnings('ignore')
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

# oracle_NEW OOF
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
oracle_unseen = oracle_new_t[unseen_mask].mean()
print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

# Check specific promising folders
check_dirs = ['framework_g', 'task_c', 'task_2_joint', 'soft_gate', 'pack_spec',
              'residual_ranking', 'residual_multi', 'base_v31', 'eda_v31',
              'mega34', 'independent', 'cluster_spec', 'paradigm_3',
              'bucket_specialist', 'feat_reinterp']

print()
print("=== Checking specific folders for OOF-improving components ===")
print(f"{'File':<55} {'OOF':>8} {'corr':>7} {'gain':>8} {'t_unseen':>9}")
print("-"*95)

found_any = False
for d in check_dirs:
    oof_files = sorted(glob.glob(f'results/{d}/*oof*.npy') + glob.glob(f'results/{d}/*_oof.npy'))
    for fpath in oof_files:
        try:
            arr = np.clip(np.load(fpath), 0, None)
            if arr.ndim > 1: arr = arr.mean(axis=1)
            if len(arr) == len(id2): o = arr[id2]
            elif len(arr) == len(y_true):
                # Check both orders
                mae_d = np.mean(np.abs(y_true - arr))
                mae_i = np.mean(np.abs(y_true - arr[id2]))
                o = arr if mae_d <= mae_i else arr[id2]
            else: continue

            solo_mae = np.mean(np.abs(y_true - o))
            if solo_mae > 10: continue

            corr = np.corrcoef(fw4_oo, o)[0,1]
            best_w, best_bl = 0, oracle_mae
            for w in np.arange(0.01, 0.51, 0.01):
                bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
                m = np.mean(np.abs(y_true - bl))
                if m < best_bl: best_bl, best_w = m, w

            gain = oracle_mae - best_bl
            fname = os.path.relpath(fpath, 'results')

            # Check test file
            test_path = fpath.replace('oof_', 'test_').replace('_oof.npy', '_test.npy')
            t_unseen_str = "n/a"
            if os.path.exists(test_path):
                t = np.clip(np.load(test_path), 0, None)
                if t.ndim > 1: t = t.mean(axis=1)
                if len(t) == len(id_order):
                    rid_u = t[unseen_mask].mean()
                    ls_u = t[te_rid_to_ls][unseen_mask].mean()
                    true_u = ls_u if abs(rid_u-ls_u) > 0.5 else rid_u
                    test_bl = np.clip((1-best_w)*oracle_new_t + best_w*(t[te_rid_to_ls] if abs(rid_u-ls_u)>0.5 else t), 0, None)
                    tu = test_bl[unseen_mask].mean()
                    t_unseen_str = f"{tu:.3f}({tu-oracle_unseen:+.3f})"

            if gain > 0.0005 or solo_mae < 8.5:
                found_any = True
                print(f"  {fname:<55} {solo_mae:8.4f}  {corr:7.4f}  {gain:+8.4f}  {t_unseen_str}")
        except:
            pass

if not found_any:
    print("  No OOF-improving or high-quality components found in these folders.")

print()
print("=== Summary: What can we still try? ===")
print(f"  oracle_NEW: OOF={oracle_mae:.4f}  LB=9.7527")
print(f"  Best prepared submissions:")
for fname in sorted(glob.glob('FINAL_*.csv'))[-10:]:
    import os
    print(f"    {os.path.basename(fname)}")
