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
id_order = test_raw['ID'].values  # _row_id order

train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values  # in _row_id order

# ls ↔ _row_id mappings
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]  # train: ls_idx for each _row_id

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])  # for each _row_id index → ls index
te_ls_to_rid = np.argsort(te_rid_to_ls)                   # for each ls index → _row_id index

# Reconstruct oracle_NEW OOF (in _row_id order)
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
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values  # in _row_id order

print(f"oracle_NEW: OOF={oracle_mae:.4f}  seen={oracle_new_t[~unseen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print()

# === Analyze Tweedie + cascade (ls order files → convert to _row_id order) ===
# v30_test_fe is in ls order → Tweedie test files are in ls order
# iter_pseudo OOF files are in ls order → convert via id2

def load_ls_test(path):
    """Load test file in ls order → return in _row_id order"""
    t_ls = np.load(path)
    if t_ls.ndim > 1: t_ls = t_ls.mean(axis=1)
    return t_ls[te_rid_to_ls]  # te_rid_to_ls[i] = ls index for _row_id index i

def load_ls_oof(path):
    """Load OOF file in ls order → return in _row_id order via id2"""
    o_ls = np.load(path)
    if o_ls.ndim > 1: o_ls = o_ls.mean(axis=1)
    if len(o_ls) == len(id2):
        return o_ls[id2]  # id2[i] = ls index for _row_id index i
    return o_ls

components = [
    # (name, oof_path, test_path, oof_in_ls, test_in_ls)
    ('tweedie/tw11', 'results/tweedie/tw11_oof.npy', 'results/tweedie/tw11_test.npy', True, True),
    ('tweedie/tw13', 'results/tweedie/tw13_oof.npy', 'results/tweedie/tw13_test.npy', True, True),
    ('tweedie/tw15', 'results/tweedie/tw15_oof.npy', 'results/tweedie/tw15_test.npy', True, True),
    ('tweedie/tw17', 'results/tweedie/tw17_oof.npy', 'results/tweedie/tw17_test.npy', True, True),
    ('tweedie/tw19', 'results/tweedie/tw19_oof.npy', 'results/tweedie/tw19_test.npy', True, True),
    ('cascade/w30_mae', 'results/cascade/spec_lgb_w30_mae_oof.npy', 'results/cascade/spec_lgb_w30_mae_test.npy', False, False),
    ('iter_r5',  'results/iter_pseudo/round5_oof.npy', 'results/iter_pseudo/round5_test.npy', True, True),
]

print(f"{'Name':<22} {'OOF_MAE':>8} {'corr':>7} {'t_seen':>8} {'t_unseen':>9} {'Δu':>7}")
print("-"*70)

best_candidates = []

for name, oof_path, test_path, oof_ls, test_ls_flag in components:
    try:
        if oof_ls:
            o = load_ls_oof(oof_path)
        else:
            o = np.clip(np.load(oof_path), 0, None)
            if len(o) == len(id2): o = o[id2]

        if test_ls_flag:
            t = load_ls_test(test_path)
        else:
            t = np.clip(np.load(test_path), 0, None)

        o = np.clip(o, 0, None)
        t = np.clip(t, 0, None)

        if len(o) != len(y_true) or len(t) != len(id_order):
            print(f"  {name}: shape mismatch o={o.shape} t={t.shape}")
            continue

        oof_mae = np.mean(np.abs(y_true - o))
        corr = np.corrcoef(fw4_oo, o)[0,1]
        t_seen = t[~unseen_mask].mean()
        t_unseen = t[unseen_mask].mean()
        delta_u = t_unseen - oracle_new_t[unseen_mask].mean()

        print(f"  {name:<20} {oof_mae:8.4f}  {corr:7.4f}  {t_seen:8.3f}  {t_unseen:9.3f}  {delta_u:+7.3f}")

        best_candidates.append({
            'name': name, 'oof_mae': oof_mae, 'corr': corr,
            't_seen': t_seen, 't_unseen': t_unseen, 'delta_u': delta_u,
            'oof_arr': o, 'test_arr': t
        })

    except Exception as e:
        print(f"  {name}: ERROR {e}")

print()
print("=== Blend sweep for each component ===")
oracle_unseen = oracle_new_t[unseen_mask].mean()

for b in best_candidates:
    o = b['oof_arr']; t = b['test_arr']
    print(f"\n  --- {b['name']} (solo OOF={b['oof_mae']:.4f}, t_unseen={b['t_unseen']:.3f}, Δu={b['delta_u']:+.3f}) ---")
    print(f"  {'w':>5} {'OOF':>10} {'ΔOOF':>10} {'t_seen':>8} {'t_unseen':>9} {'Δu':>7}")
    for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        oof_bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
        test_bl = np.clip((1-w)*oracle_new_t + w*t, 0, None)
        bl_mae = np.mean(np.abs(y_true - oof_bl))
        bl_unseen = test_bl[unseen_mask].mean()
        bl_seen = test_bl[~unseen_mask].mean()
        du = bl_unseen - oracle_unseen
        mark = " <<" if bl_mae < oracle_mae else ""
        print(f"  {w:5.2f}  {bl_mae:10.4f}  {bl_mae-oracle_mae:+10.4f}  {bl_seen:8.3f}  {bl_unseen:9.3f}  {du:+7.3f}{mark}")

print()
print("=== Decision summary ===")
print(f"oracle_NEW: OOF={oracle_mae:.4f}  t_unseen={oracle_unseen:.3f}")
print()
print("Any blend that improves OOF AND doesn't lose test_unseen >= -0.3:")
found_any = False
for b in best_candidates:
    o = b['oof_arr']; t = b['test_arr']
    for w in np.arange(0.01, 0.31, 0.01):
        oof_bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
        test_bl = np.clip((1-w)*oracle_new_t + w*t, 0, None)
        bl_mae = np.mean(np.abs(y_true - oof_bl))
        bl_unseen = test_bl[unseen_mask].mean()
        if bl_mae < oracle_mae and (bl_unseen >= oracle_unseen - 0.3):
            found_any = True
            print(f"  FOUND: {b['name']} w={w:.2f}: OOF={bl_mae:.4f} ({bl_mae-oracle_mae:+.4f}) unseen={bl_unseen:.3f} ({bl_unseen-oracle_unseen:+.3f})")
if not found_any:
    print("  None found.")
