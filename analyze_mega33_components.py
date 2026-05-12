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
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
best_dr2, best_dr3 = -0.04, -0.02
wm_best = fw['mega33'] - best_dr2 - best_dr3
w2_best = fw['iter_r2'] + best_dr2
w3_best = fw['iter_r3'] + best_dr3

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
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
oracle_oof  = np.clip(0.64*fixed_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')

# Examine mega33 component models
print('\n=== mega33 meta_oofs structure ===')
m33_oofs = d33['meta_oofs']
m33_tests = d33['meta_tests']
print(f'meta_oofs type: {type(m33_oofs)}')
if isinstance(m33_oofs, dict):
    print(f'Number of models: {len(m33_oofs)}')
    for k, v in list(m33_oofs.items())[:5]:
        print(f'  {k}: type={type(v).__name__}')
        if isinstance(v, np.ndarray):
            print(f'    shape={v.shape}  mean={v.mean():.3f}')
        elif isinstance(v, list):
            print(f'    len={len(v)}  [0] type={type(v[0]).__name__}')
elif isinstance(m33_oofs, list):
    print(f'Number of models: {len(m33_oofs)}')
    for i, v in enumerate(m33_oofs[:3]):
        print(f'  [{i}]: type={type(v).__name__}  shape={v.shape if hasattr(v,"shape") else "N/A"}')

# Analyze individual models in mega33
print('\n=== Individual model analysis in mega33 ===')
component_info = []
if isinstance(m33_oofs, dict):
    for k in m33_oofs.keys():
        oo = m33_oofs[k]
        ot = m33_tests[k]
        if isinstance(oo, np.ndarray) and oo.shape == (len(train_ls),):
            oo_id = oo[id2]
            ot_id = ot[te_id2]
            oof_v = mae(np.clip(oo_id,0,None))
            corr_v = np.corrcoef(oracle_oof, np.clip(oo_id,0,None))[0,1]
            component_info.append((k, oof_v, ot_id.mean(), corr_v))
        elif isinstance(oo, list) and len(oo) > 0:
            avg_oof = np.mean([np.clip(x,0,None) for x in oo], axis=0)
            avg_test = np.mean([np.clip(x,0,None) for x in ot], axis=0)
            oof_v = mae(avg_oof)
            corr_v = np.corrcoef(oracle_oof, avg_oof)[0,1] if len(avg_oof)==len(y_true) else -1
            component_info.append((k, oof_v, avg_test.mean(), corr_v))

if component_info:
    component_info.sort(key=lambda x: x[2], reverse=True)
    print(f'{"model":30s}  {"OOF":>9}  {"test_mean":>10}  {"corr":>8}')
    for name, oof_v, tm, corr_v in component_info[:20]:
        print(f'{name:30s}  {oof_v:>9.5f}  {tm:>10.3f}  {corr_v:>8.4f}')

# Also check mega34 components for any new models
print('\n=== mega34 additional models (vs mega33) ===')
m34_oofs = d34['meta_oofs']
m34_tests = d34['meta_tests']
if isinstance(m34_oofs, dict) and isinstance(m33_oofs, dict):
    new_keys = set(m34_oofs.keys()) - set(m33_oofs.keys())
    print(f'Keys in mega34 but not in mega33: {len(new_keys)}')
    for k in list(new_keys)[:10]:
        oo = m34_oofs[k]
        ot = m34_tests[k]
        if isinstance(oo, np.ndarray) and oo.shape == (len(train_ls),):
            oo_id = oo[id2]
            ot_id = ot[te_id2]
            print(f'  {k}: OOF={mae(np.clip(oo_id,0,None)):.5f}  test_mean={ot_id.mean():.3f}')
        elif isinstance(oo, list) and len(oo) > 0:
            avg_oof = np.mean([np.clip(x,0,None) for x in oo], axis=0)
            avg_test = np.mean([np.clip(x,0,None) for x in ot], axis=0)
            print(f'  {k}: OOF={mae(avg_oof):.5f}  test_mean={avg_test.mean():.3f}')

# Check: can we pick high-test-mean components and blend them with oracle?
print('\n=== Top high-test-mean components blended with oracle_NEW ===')
rw_oof  = np.clip(0.64*(wm_best*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof) + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
rw_test = np.clip(0.64*(wm_best*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test) + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

if component_info:
    # Top 5 by test_mean
    top_tm = sorted(component_info, key=lambda x: x[2], reverse=True)[:5]
    for name, _, tm, _ in top_tm:
        k = name
        oo = m33_oofs.get(k, m34_oofs.get(k))
        ot = m33_tests.get(k, m34_tests.get(k))
        if isinstance(oo, np.ndarray) and oo.shape == (len(train_ls),):
            oo_id = np.clip(oo[id2], 0, None)
            ot_id = np.clip(ot[te_id2], 0, None)
        elif isinstance(oo, list):
            oo_id = np.clip(np.mean([np.clip(x,0,None) for x in oo], axis=0), 0, None)
            ot_id = np.clip(np.mean([np.clip(x,0,None) for x in ot], axis=0), 0, None)
        else:
            continue
        if len(oo_id) != len(y_true):
            continue
        for w in [0.05, 0.10]:
            b_oof = np.clip((1-w)*rw_oof + w*oo_id, 0, None)
            b_test = np.clip((1-w)*rw_test + w*ot_id, 0, None)
            marker = '*' if mae(b_oof) < base_oof else ''
            print(f'  {name[:25]:25s} w={w:.2f}: OOF={mae(b_oof):.5f}  delta={mae(b_oof)-base_oof:+.6f}  test={b_test.mean():.3f} {marker}')

print('\nDone.')
