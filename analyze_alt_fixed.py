import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
rank_oof    = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test   = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o   = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o   = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o   = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t   = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t   = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t   = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

fixed_orig_oof  = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fixed_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
fixed_rw_oof  = wm_best*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
fixed_rw_test = wm_best*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test

oracle_oof  = np.clip(0.64*fixed_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)
rw_oof  = np.clip(0.64*fixed_rw_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
rw_test = np.clip(0.64*fixed_rw_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'fixed_rw:   OOF={mae(rw_oof):.5f}  test_mean={rw_test.mean():.3f}')

# 1. Load mega34 and mega33_v31
print('\n=== Alternative FIXED components ===')
alt_megas = {}
for name, fpath in [('mega34', 'results/mega34_final.pkl'),
                    ('mega33_v31', 'results/mega33_v31_final.pkl'),
                    ('mega37', 'results/mega37_final.pkl')]:
    if os.path.exists(fpath):
        with open(fpath,'rb') as f:
            dm = pickle.load(f)
        oof = dm['meta_avg_oof'][id2]
        test = dm['meta_avg_test'][te_id2]
        alt_megas[name] = (oof, test)
        print(f'  {name}: OOF={mae(oof):.5f}  test_mean={test.mean():.3f}  corr_mega33={np.corrcoef(mega33_oof,oof)[0,1]:.4f}')
    else:
        print(f'  {name}: not found')

# Try replacing mega33 with mega34 in FIXED
print('\n=== mega34 as FIXED component (replacing mega33) ===')
if 'mega34' in alt_megas:
    m34_oof, m34_test = alt_megas['mega34']
    for w34 in [0.3, 0.5, 0.7, 1.0]:
        # Blend mega33 and mega34
        m_blend_oof = (1-w34)*mega33_oof + w34*m34_oof
        m_blend_test = (1-w34)*mega33_test + w34*m34_test
        fx_oof = wm_best*m_blend_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
        fx_test = wm_best*m_blend_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test
        o_oof = np.clip(0.64*fx_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
        o_test = np.clip(0.64*fx_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)
        marker = '*' if mae(o_oof) < base_oof else ''
        print(f'  w34={w34:.1f}: OOF={mae(o_oof):.5f}  delta={mae(o_oof)-base_oof:+.6f}  test={o_test.mean():.3f} {marker}')

# 2. Adversarial domain adaptation files
print('\n=== Adversarial domain adaptation ===')
if os.path.exists('results/adversarial_phase2.pkl'):
    with open('results/adversarial_phase2.pkl','rb') as f:
        dadv = pickle.load(f)
    print(f'Keys: {list(dadv.keys())}')
    for k, v in dadv.items():
        if isinstance(v, dict):
            print(f'  {k}: {list(v.keys())[:3]}')
            for kk, vv in v.items():
                if isinstance(vv, np.ndarray):
                    print(f'    {kk}: shape={vv.shape}  mean={vv.mean():.3f}')
        elif isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')

# 3. Domain phase 2 (domain_shift reweighting)
print('\n=== Domain phase 2 ===')
if os.path.exists('results/domain_phase2.pkl'):
    with open('results/domain_phase2.pkl','rb') as f:
        dom2 = pickle.load(f)
    print(f'Keys: {list(dom2.keys())}')
    for k, v in dom2.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')

# 4. Offset corrections
print('\n=== Offset phase 2 (scenario offsets) ===')
if os.path.exists('results/offset_phase2.pkl'):
    with open('results/offset_phase2.pkl','rb') as f:
        off2 = pickle.load(f)
    print(f'Keys: {list(off2.keys())}')
    for k, v in off2.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')
        elif isinstance(v, (int, float)):
            print(f'  {k}: {v:.4f}')

if os.path.exists('results/offset_phase3.pkl'):
    with open('results/offset_phase3.pkl','rb') as f:
        off3 = pickle.load(f)
    print(f'\nOffset phase 3 keys: {list(off3.keys())}')
    for k, v in off3.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')
        elif isinstance(v, dict):
            print(f'  {k}: dict with keys {list(v.keys())}')

# 5. Check offset phase 2: sc_oof_avg and sc_test_avg
print('\n=== Offset sc predictions ===')
if os.path.exists('results/offset_phase1.pkl'):
    with open('results/offset_phase1.pkl','rb') as f:
        off1 = pickle.load(f)
    print(f'Phase 1 keys: {list(off1.keys())}')
    for k, v in off1.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}  mean={np.clip(v,0,None).mean():.3f}')

    # Try: offset phase 1 sc_oof_avg as a new signal
    if 'sc_oof_avg' in off1:
        sc_oof = off1['sc_oof_avg']
        sc_test = off1['sc_test_avg']
        print(f'\n  sc_oof_avg as new signal: OOF={mae(np.clip(sc_oof,0,None)):.5f}  corr={np.corrcoef(oracle_oof,np.clip(sc_oof,0,None))[0,1]:.4f}')
        for w in [0.02, 0.05, 0.08, 0.10]:
            b = np.clip((1-w)*rw_oof + w*np.clip(sc_oof,0,None), 0, None)
            bt = np.clip((1-w)*rw_test + w*np.clip(sc_test,0,None), 0, None)
            print(f'    w={w:.2f}: OOF={mae(b):.5f}  delta={mae(b)-base_oof:+.6f}  test={bt.mean():.3f}')

# 6. Check stacking final
print('\n=== Stacking final ===')
if os.path.exists('results/stacking_final.pkl'):
    with open('results/stacking_final.pkl','rb') as f:
        stk = pickle.load(f)
    print(f'Keys: {list(stk.keys())}')
    if 'results' in stk:
        res = stk['results']
        if isinstance(res, dict):
            for k, v in res.items():
                if isinstance(v, dict) and 'oof' in v:
                    print(f'  {k}: OOF={mae(np.clip(v["oof"],0,None)):.5f}  test_mean={np.clip(v.get("test",np.zeros(1)),0,None).mean():.3f}')

# 7. What's in the v31 feature OOF (version with best feature engineering)?
print('\n=== v31 feature set OOF (best feature engineering) ===')
if os.path.exists('results/eda_v31/v31_fe_cache.pkl'):
    with open('results/eda_v31/v31_fe_cache.pkl','rb') as f:
        v31 = pickle.load(f)
    print(f'Keys: {list(v31.keys())}')
    for k, v in v31.items():
        if isinstance(v, pd.DataFrame):
            print(f'  {k}: shape={v.shape}  cols={list(v.columns[:5])}')
        elif isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape}')
        elif isinstance(v, list):
            print(f'  {k}: list len={len(v)} example={v[:3]}')

# 8. mega34 stats vs mega33
print('\n=== mega34 vs mega33: where does mega34 differ? ===')
if 'mega34' in alt_megas:
    m34_oof, m34_test = alt_megas['mega34']
    diff_oof = m34_oof - mega33_oof
    diff_test = m34_test - mega33_test
    print(f'mean diff (OOF): {diff_oof.mean():.3f}  std: {diff_oof.std():.3f}')
    print(f'mean diff (test): {diff_test.mean():.3f}  std: {diff_test.std():.3f}')

    # Layout-level diff
    train_raw['m33'] = mega33_oof
    train_raw['m34'] = m34_oof
    ly_diff = train_raw.groupby('layout_id').agg(
        pack=('pack_utilization','mean'),
        m33_mean=('m33','mean'),
        m34_mean=('m34','mean')
    ).reset_index()
    ly_diff['m34_minus_m33'] = ly_diff['m34_mean'] - ly_diff['m33_mean']
    hi_pack = ly_diff[ly_diff['pack'] > 0.75].sort_values('pack', ascending=False)
    print(f'\nHigh-pack layouts m34 - m33:')
    for _, r in hi_pack.head(10).iterrows():
        print(f'  {r["layout_id"]:12s}: pack={r["pack"]:.3f}  m33={r["m33_mean"]:.2f}  m34={r["m34_mean"]:.2f}  diff={r["m34_minus_m33"]:+.2f}')

print('\nDone.')
