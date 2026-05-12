import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle, glob, json
from scipy.stats import pearsonr

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
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
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
residuals_train = y_true - fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("Exploring seqD models (layout-stats, meta_minimal, meta_stacking)")
print("="*70)

seqD_files = [
    ('oof_seqD_layout_stats.npy', 'test_D_layout_stats.npy', 'seqD_layout_stats'),
    ('oof_seqD_meta_minimal.npy', 'test_D_meta_minimal.npy', 'seqD_meta_minimal'),
    ('oof_seqD_meta_stacking.npy', 'test_D_meta_stacking.npy', 'seqD_meta_stacking'),
]

for oof_f, te_f, name in seqD_files:
    oof_path = f'results/oracle_seq/{oof_f}'
    te_path  = f'results/oracle_seq/{te_f}'
    print(f"\n--- {name} ---")
    try:
        oof_raw = np.load(oof_path)
        te_raw  = np.load(te_path)
        print(f"  OOF shape: {oof_raw.shape}  Test shape: {te_raw.shape}")
        print(f"  OOF stats: min={oof_raw.min():.3f}  max={oof_raw.max():.3f}  mean={oof_raw.mean():.3f}")
        print(f"  Test stats: min={te_raw.min():.3f}  max={te_raw.max():.3f}  mean={te_raw.mean():.3f}")

        # Try with and without reindex
        if len(oof_raw) == len(y_true):
            oof_di = np.clip(oof_raw, 0, None)
            mae_di = np.mean(np.abs(oof_di - y_true))
            r_di, _ = pearsonr(oof_di, y_true)
            print(f"  Direct (no reindex): OOF MAE={mae_di:.4f}  r={r_di:.4f}")

            oof_ri = np.clip(oof_raw[id2], 0, None)
            mae_ri = np.mean(np.abs(oof_ri - y_true))
            r_ri, _ = pearsonr(oof_ri, y_true)
            print(f"  Reindexed [id2]:     OOF MAE={mae_ri:.4f}  r={r_ri:.4f}")

            # Use better one
            oof_best = oof_ri if mae_ri < mae_di else oof_di
            te_best = np.clip(te_raw[te_id2], 0, None) if mae_ri < mae_di else np.clip(te_raw, 0, None)
            print(f"  Using: {'reindexed' if mae_ri < mae_di else 'direct'}")
        else:
            # Not same length as train, might be layout-level
            print(f"  Length mismatch! len={len(oof_raw)}, y_true len={len(y_true)}")
            print(f"  n_train_layouts={len(train_raw['layout_id'].unique())}")
            continue

        mae_best = min(mae_di, mae_ri)
        oof_best_clip = oof_ri if mae_ri < mae_di else oof_di

        r_fw4, _ = pearsonr(oof_best_clip, fw4_oo)
        print(f"  r(seqD_oof, fw4_oo) = {r_fw4:.4f}")

        # Test predictions
        u_mean = te_best[unseen_mask].mean()
        s_mean = te_best[seen_mask].mean()
        print(f"  Test: seen={s_mean:.3f}  unseen={u_mean:.3f}  Δ vs oracle_NEW unseen={u_mean-oracle_new_t[unseen_mask].mean():+.3f}")

        # Quick correlation with oracle_NEW
        r_te, _ = pearsonr(te_best, oracle_new_t)
        print(f"  r(seqD_test, oracle_NEW_test) = {r_te:.4f}")

    except Exception as e:
        print(f"  ERROR: {e}")

# ============================================================
# Check what's in results/oracle_seq/summary.json
# ============================================================
print("\n" + "="*70)
print("Summary JSON contents")
print("="*70)
try:
    with open('results/oracle_seq/summary.json') as f:
        d = json.load(f)
    for k, v in d.items():
        if isinstance(v, (int, float, str)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in list(v.items())[:5]:
                print(f"    {k2}: {v2}")
except Exception as e:
    print(f"  Error: {e}")

# ============================================================
# Scan remaining oracle_seq files not yet evaluated
# ============================================================
print("\n" + "="*70)
print("All oracle_seq OOF files — full scan")
print("="*70)
oof_files = sorted(glob.glob('results/oracle_seq/oof_*.npy'))
print(f"{'name':40s}  {'MAE_di':>8}  {'r_di':>6}  {'MAE_ri':>8}  {'r_ri':>6}  {'u_mean_ri':>10}  {'notes'}")
for fp in oof_files:
    name = os.path.basename(fp).replace('oof_','').replace('.npy','')
    te_fp = fp.replace('oof_','test_').replace('oracle_seq/oof_', 'oracle_seq/test_')
    # oracle_seq prefix convention
    te_fp2 = fp.replace('/oof_seqD_', '/test_D_').replace('/oof_seqC_', '/test_C_').replace('/oof_seqB', '/test_B').replace('/oof_seqA', '/test_A')
    try:
        oof_raw = np.load(fp)
        if len(oof_raw) != len(y_true):
            print(f"  {name:40s}  len={len(oof_raw)} ≠ {len(y_true)} — skip")
            continue

        oof_di = np.clip(oof_raw, 0, None)
        oof_ri = np.clip(oof_raw[id2], 0, None)
        mae_di = np.mean(np.abs(oof_di - y_true))
        mae_ri = np.mean(np.abs(oof_ri - y_true))
        r_di, _ = pearsonr(oof_di, y_true)
        r_ri, _ = pearsonr(oof_ri, y_true)

        # Load test if available
        u_mean_ri = np.nan
        for te_f in [te_fp, te_fp2]:
            if os.path.exists(te_f):
                te_raw = np.load(te_f)
                if len(te_raw) == len(id_order):
                    te_di = np.clip(te_raw, 0, None)
                    u_mean_ri = te_di[unseen_mask].mean()
                elif len(te_raw) >= len(te_id2):
                    te_ri = np.clip(te_raw[te_id2], 0, None)
                    u_mean_ri = te_ri[unseen_mask].mean()
                break

        note = ''
        if mae_ri < mae_di: note = 'reindex'
        if abs(mae_di - mae_ri) < 0.01: note = '~same'

        print(f"  {name:40s}  {mae_di:8.4f}  {r_di:6.4f}  {mae_ri:8.4f}  {r_ri:6.4f}  {u_mean_ri:10.3f}  {note}")
    except Exception as e:
        print(f"  {name:40s}  ERROR: {e}")

print("\nDone.")
