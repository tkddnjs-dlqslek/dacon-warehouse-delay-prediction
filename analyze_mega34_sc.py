import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

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

# Baseline
fixed_orig_oof  = fw['mega33']*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + fw['iter_r2']*r2_oof + fw['iter_r3']*r3_oof
fixed_orig_test = fw['mega33']*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + fw['iter_r2']*r2_test + fw['iter_r3']*r3_test
oracle_oof  = np.clip(0.64*fixed_orig_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
oracle_test = np.clip(0.64*fixed_orig_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

# rw baseline
fixed_rw_oof  = wm_best*mega33_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_best*r2_oof + w3_best*r3_oof
fixed_rw_test = wm_best*mega33_test + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2_best*r2_test + w3_best*r3_test
rw_oof  = np.clip(0.64*fixed_rw_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o, 0, None)
rw_test = np.clip(0.64*fixed_rw_test + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
base_oof = mae(oracle_oof)
print(f'oracle_NEW: OOF={base_oof:.5f}  test_mean={oracle_test.mean():.3f}')
print(f'fixed_rw:   OOF={mae(rw_oof):.5f}  test_mean={rw_test.mean():.3f}')
print(f'mega34 standalone: OOF={mae(mega34_oof):.5f}  test_mean={mega34_test.mean():.3f}')

# mega34 as part of FIXED blend
def make_oracle_m34(w34, wm, wr, w1, w2, w3, wf=0.64, wxgb=0.12, wlv2=0.16, wrem=0.08):
    m_blend_oof = (1-w34)*mega33_oof + w34*mega34_oof
    m_blend_test = (1-w34)*mega33_test + w34*mega34_test
    fx_oof = wm*m_blend_oof + wr*rank_oof + w1*r1_oof + w2*r2_oof + w3*r3_oof
    fx_test = wm*m_blend_test + wr*rank_test + w1*r1_test + w2*r2_test + w3*r3_test
    o_oof = np.clip(wf*fx_oof + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    o_test = np.clip(wf*fx_test + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    return o_oof, o_test

print('\n=== mega34 blend sweep (w34) with rw FIXED weights ===')
print(f'{"w34":>6}  {"OOF":>9}  {"delta":>9}  {"test_mean":>10}')
best_val = base_oof
best_w34 = None
for w34 in np.arange(0.0, 0.55, 0.05):
    o_oof, o_test = make_oracle_m34(w34, wm_best, fw['rank_adj'], fw['iter_r1'], w2_best, w3_best)
    marker = '*' if mae(o_oof) < base_oof else ''
    if mae(o_oof) < best_val:
        best_val = mae(o_oof)
        best_w34 = w34
    print(f'{w34:>6.2f}  {mae(o_oof):>9.5f}  {mae(o_oof)-base_oof:>+9.6f}  {o_test.mean():>10.3f} {marker}')

print(f'\nBest w34={best_w34:.2f}: OOF={best_val:.5f}  delta={best_val-base_oof:+.6f}')

# Fold-level analysis for best w34
if best_w34 is not None:
    o_oof_best, o_test_best = make_oracle_m34(best_w34, wm_best, fw['rank_adj'], fw['iter_r1'], w2_best, w3_best)
    print(f'\nFold-level for w34={best_w34:.2f}:')
    groups = train_raw['layout_id'].values
    gkf = GroupKFold(n_splits=5)
    for fi, (_, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        fo = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs],0,None))
        fb = mean_absolute_error(y_true[vs], o_oof_best[vs])
        print(f'  Fold {fi+1}: oracle={fo:.5f}  m34_blend={fb:.5f}  delta={fb-fo:+.6f}')

# Layout residuals for mega34 blend
print('\n=== Layout residuals: mega34 blend vs oracle ===')
o_oof_best, o_test_best = make_oracle_m34(0.30, wm_best, fw['rank_adj'], fw['iter_r1'], w2_best, w3_best)
train_raw['oracle'] = oracle_oof
train_raw['m34_blend'] = o_oof_best
ly = train_raw.groupby('layout_id').agg(
    pack=('pack_utilization','mean'),
    y_mean=('avg_delay_minutes_next_30m','mean'),
    oracle_mean=('oracle','mean'),
    m34_mean=('m34_blend','mean')
).reset_index()
ly['oracle_resid'] = ly['y_mean'] - ly['oracle_mean']
ly['m34_resid'] = ly['y_mean'] - ly['m34_mean']
corr_oracle = np.corrcoef(ly['pack'], ly['oracle_resid'])[0,1]
corr_m34 = np.corrcoef(ly['pack'], ly['m34_resid'])[0,1]
print(f'corr(pack, oracle_resid)={corr_oracle:.4f}  corr(pack, m34_resid)={corr_m34:.4f}')
hi = ly[ly['pack']>0.75].sort_values('pack', ascending=False)
for _, r in hi.iterrows():
    print(f'  {r["layout_id"]:12s}: pack={r["pack"]:.3f}  oracle_r={r["oracle_resid"]:+.2f}  m34_r={r["m34_resid"]:+.2f}  diff={r["m34_resid"]-r["oracle_resid"]:+.2f}')

# sc_pred_row analysis
print('\n=== sc_pred_row (scenario-level model) ===')
with open('results/offset_phase1.pkl','rb') as f:
    off1 = pickle.load(f)
sc_pred = off1['sc_pred_row']
sc_pred_t = off1['sc_pred_row_test']
print(f'sc_pred_row: shape={sc_pred.shape}  mean={sc_pred.mean():.3f}  OOF={mae(sc_pred):.5f}')
print(f'sc_pred_row_test: mean={sc_pred_t.mean():.3f}')
print(f'corr(sc, oracle): {np.corrcoef(oracle_oof, np.clip(sc_pred,0,None))[0,1]:.4f}')

# Blend sc_pred_row with oracle/rw
print('\nBlend sc_pred_row with rw:')
for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
    b_oof = np.clip((1-w)*rw_oof + w*np.clip(sc_pred,0,None), 0, None)
    b_test = np.clip((1-w)*rw_test + w*np.clip(sc_pred_t,0,None), 0, None)
    marker = '*' if mae(b_oof) < base_oof else ''
    print(f'  w={w:.2f}: OOF={mae(b_oof):.5f}  delta={mae(b_oof)-base_oof:+.6f}  test={b_test.mean():.3f} {marker}')

# domain_phase2 analysis
print('\n=== domain_phase2 oofs/tests ===')
with open('results/domain_phase2.pkl','rb') as f:
    dom2 = pickle.load(f)
d2_oofs = dom2['oofs']  # list or dict
d2_tests = dom2['tests']
if isinstance(d2_oofs, dict):
    for k, v in d2_oofs.items():
        if isinstance(v, np.ndarray):
            print(f'  {k}: OOF={mae(np.clip(v,0,None)):.5f}  shape={v.shape}')
elif isinstance(d2_oofs, (list, np.ndarray)):
    print(f'oofs type={type(d2_oofs)}  len={len(d2_oofs)}')
    if isinstance(d2_oofs, list):
        for i, v in enumerate(d2_oofs[:3]):
            if isinstance(v, np.ndarray):
                print(f'  [{i}]: shape={v.shape}  mean={np.clip(v,0,None).mean():.3f}  OOF={mae(np.clip(v,0,None)):.5f}')
    else:
        print(f'  shape={d2_oofs.shape}  OOF={mae(np.clip(d2_oofs,0,None)):.5f}')

if isinstance(d2_tests, (list, np.ndarray)):
    if isinstance(d2_tests, list):
        for i, v in enumerate(d2_tests[:3]):
            if isinstance(v, np.ndarray):
                print(f'  test[{i}]: shape={v.shape}  mean={np.clip(v,0,None).mean():.3f}')
    else:
        print(f'  test shape={d2_tests.shape}  mean={d2_tests.mean():.3f}')

# adversarial domain phase 1
print('\n=== adversarial_phase1 ===')
with open('results/adversarial_phase1.pkl','rb') as f:
    adv1 = pickle.load(f)
for k, v in adv1.items():
    if isinstance(v, np.ndarray):
        print(f'  {k}: shape={v.shape}  mean={v.mean():.3f}')
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, np.ndarray):
                print(f'  {k}.{kk}: shape={vv.shape}  mean={vv.mean():.3f}')

# adversarial_phase2 deeper
print('\n=== adversarial_phase2 deeper ===')
with open('results/adversarial_phase2.pkl','rb') as f:
    adv2 = pickle.load(f)
for alpha_key in ['alpha1', 'alpha3', 'alpha5']:
    sub_d = adv2[alpha_key]
    if 'oofs' in sub_d and 'tests' in sub_d:
        oofs = sub_d['oofs']
        tests = sub_d['tests']
        if isinstance(oofs, (list, np.ndarray)) and len(oofs) > 0:
            if isinstance(oofs, list):
                avg_oof = np.mean([np.clip(x,0,None) for x in oofs], axis=0)
                avg_test = np.mean([np.clip(x,0,None) for x in tests], axis=0)
            else:
                avg_oof = np.clip(oofs, 0, None)
                avg_test = np.clip(tests, 0, None)
            print(f'  {alpha_key}: OOF={mae(avg_oof):.5f}  test_mean={avg_test.mean():.3f}  corr={np.corrcoef(oracle_oof,avg_oof)[0,1]:.4f}')

# offset_phase3 models
print('\n=== offset_phase3 models ===')
with open('results/offset_phase3.pkl','rb') as f:
    off3 = pickle.load(f)
for name, d in off3.items():
    oo = np.clip(d['oof'],0,None)
    ot = np.clip(d['test'],0,None)
    print(f'  {name}: OOF={mae(oo):.5f}  test_mean={ot.mean():.3f}  corr={np.corrcoef(oracle_oof,oo)[0,1]:.4f}')
    for w in [0.05, 0.10]:
        b = np.clip((1-w)*rw_oof + w*oo, 0, None)
        bt = np.clip((1-w)*rw_test + w*ot, 0, None)
        if mae(b) < base_oof:
            print(f'    w={w:.2f}: OOF={mae(b):.5f}  delta={mae(b)-base_oof:+.6f}  test={bt.mean():.3f} *')

print('\nDone.')
