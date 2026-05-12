import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, glob
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

# ls ↔ rid mappings
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

# Reconstruct oracle_NEW OOF
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
print()

# Determine if oracle_seq OOF files are in ls or rid order
# (check xgb vs conversion: if len=len(y_true) and no id2 needed, might be rid order)
oseq_xgb = np.load('results/oracle_seq/oof_seqC_xgb.npy')
print(f"oof_seqC_xgb length: {len(oseq_xgb)}")
print(f"  direct MAE: {np.mean(np.abs(y_true - np.clip(oseq_xgb,0,None))):.4f}")
print(f"  via id2 MAE: {np.mean(np.abs(y_true - np.clip(oseq_xgb[id2],0,None))):.4f}")
print("  (lower MAE = correct order)")
print()

# Decide ordering based on which gives lower MAE
use_id2_for_oseq = np.mean(np.abs(y_true - np.clip(oseq_xgb[id2],0,None))) < np.mean(np.abs(y_true - np.clip(oseq_xgb,0,None)))
print(f"oracle_seq files are in {'ls' if use_id2_for_oseq else '_row_id'} order")
print()

def load_oof(path, use_id2=False):
    arr = np.clip(np.load(path), 0, None)
    if arr.ndim > 1: arr = arr.mean(axis=1)
    if use_id2 and len(arr) == len(id2): return arr[id2]
    if len(arr) == len(y_true): return arr
    return None

def load_test(path, is_ls=False):
    arr = np.clip(np.load(path), 0, None)
    if arr.ndim > 1: arr = arr.mean(axis=1)
    if is_ls and len(arr) == len(id_order): return arr[te_rid_to_ls]
    if len(arr) == len(id_order): return arr
    return None

# Scan ALL oracle_seq OOF files
print("=== Oracle_seq variants scan ===")
print(f"{'Name':<40} {'OOF_solo':>9} {'corr':>7} {'blend_best':>11} {'gain':>7}")
print("-"*85)

oseq_oofs = sorted(glob.glob('results/oracle_seq/oof_*.npy'))
results = []
for fpath in oseq_oofs:
    fname = os.path.basename(fpath).replace('oof_', '').replace('.npy', '')
    try:
        o = load_oof(fpath, use_id2=use_id2_for_oseq)
        if o is None or len(o) != len(y_true): continue

        solo_mae = np.mean(np.abs(y_true - o))
        corr = np.corrcoef(fw4_oo, o)[0,1]

        best_w, best_bl = 0, oracle_mae
        for w in np.arange(0.01, 0.51, 0.01):
            bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
            m = np.mean(np.abs(y_true - bl))
            if m < best_bl: best_bl, best_w = m, w

        gain = oracle_mae - best_bl
        print(f"  {fname:<38} {solo_mae:9.4f}  {corr:7.4f}  {best_bl:11.4f}  {gain:+7.4f}")
        results.append((gain, best_bl, best_w, fname, fpath, o))
    except Exception as e:
        pass

print()
print("=== Top gainers from oracle_seq ===")
for gain, best_bl, best_w, fname, fpath, o in sorted(results, reverse=True)[:10]:
    if gain > 0.0002:
        print(f"  {fname}: best_w={best_w:.2f}  blend_OOF={best_bl:.4f}  gain={gain:+.4f}")
        # Check corresponding test file
        test_path = fpath.replace('oof_', 'test_')
        if os.path.exists(test_path):
            t = load_test(test_path, is_ls=use_id2_for_oseq)
            if t is not None and len(t) == len(id_order):
                test_bl = np.clip((1-best_w)*oracle_new_t + best_w*t, 0, None)
                print(f"    test: seen={test_bl[~unseen_mask].mean():.3f}  unseen={test_bl[unseen_mask].mean():.3f}  Δu={test_bl[unseen_mask].mean()-oracle_unseen:+.3f}")

print()
print("=== Also scan broader results folders ===")
extra_paths = (
    sorted(glob.glob('results/eda_v31/*.npy')) +
    sorted(glob.glob('results/base_v31/*.npy')) +
    sorted(glob.glob('results/ranking_variants/*oof*.npy')) +
    sorted(glob.glob('results/meta_exp/*oof*.npy'))
)
extra_results = []
for fpath in extra_paths:
    try:
        # Try both orderings
        arr = np.clip(np.load(fpath), 0, None)
        if arr.ndim > 1: arr = arr.mean(axis=1)
        if len(arr) == len(id2):
            o_direct = arr
            o_id2 = arr[id2]
            mae_direct = np.mean(np.abs(y_true - o_direct))
            mae_id2 = np.mean(np.abs(y_true - o_id2))
            o = o_id2 if mae_id2 < mae_direct else o_direct
        elif len(arr) == len(y_true):
            o = arr
        else:
            continue

        solo_mae = np.mean(np.abs(y_true - o))
        if solo_mae > 15: continue

        corr = np.corrcoef(fw4_oo, o)[0,1]
        best_w, best_bl = 0, oracle_mae
        for w in np.arange(0.01, 0.31, 0.01):
            bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
            m = np.mean(np.abs(y_true - bl))
            if m < best_bl: best_bl, best_w = m, w

        gain = oracle_mae - best_bl
        if gain > 0.001:
            fname = os.path.relpath(fpath, 'results')
            extra_results.append((gain, best_bl, best_w, fname))
            print(f"  {fname:<50} solo={solo_mae:.4f}  corr={corr:.4f}  blend={best_bl:.4f}  gain={gain:+.4f}  w={best_w:.2f}")
    except:
        pass

if not extra_results:
    print("  None found improving oracle_NEW")
