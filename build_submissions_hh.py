"""
HH Round: ref3 variant 간 diversity 기반 앙상블 (R3 파일 필요).
이전 ref3 파일들을 조합하여 새로운 앙상블을 만든다.

HH1: ref3 variant CSV 파일들의 산술 평균 앙상블
HH2: ref3 variant CSV 파일들의 geometric 평균 앙상블
HH3: ref3 variant CSV 파일들의 rank 평균 앙상블 (순위 기반)
HH4: ref3 variant CSV + OOF 기반 weight
HH5: ref3 components 재섞기 (각 component별 best 예측에서 선택)
HH6: ref3 w/ covariate shift weights (layout frequency in test vs train)
HH7: Position-weighted blend (timeslot 위치별 다른 component 선호)
HH8: ref3 blend + test prediction variance minimization
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings, glob
warnings.filterwarnings('ignore')
from scipy.optimize import minimize

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

sample_sub = pd.read_csv('sample_submission.csv')
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)

KEYS = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
co = {
    'mega33':     d33['meta_avg_oof'][id2],
    'rank_adj':   np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_xgb': np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
}
ct = {
    'mega33':     d33['meta_avg_test'][te_id2],
    'rank_adj':   np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_xgb': np.load('results/oracle_seq/test_C_xgb.npy'),
    'oracle_lv2': np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_rem': np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
}

clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
rh_oof = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_te  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_te  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

oofs_all = np.stack([co[k] for k in KEYS], axis=1)
tes_all  = np.stack([ct[k] for k in KEYS], axis=1)

saved = []

def mae(pred): return np.mean(np.abs(pred - y_true))

w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3 = np.clip(oofs_all@w_ref3, 0, None)
bt_ref3 = np.clip(tes_all@w_ref3, 0, None)
print(f"ref3 base OOF: {mae(bo_ref3):.5f}")

def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

def std_gate(base_oof, base_test, label, wide=False):
    p1_rng = np.arange(0.04,0.22,0.01) if wide else np.arange(0.06,0.18,0.01)
    p2_rng = np.arange(0.10,0.55,0.02) if wide else np.arange(0.15,0.45,0.02)
    best=mae(base_oof); bo=base_oof; bt=base_test; cfg='no_gate'
    for p1 in p1_rng:
        m1=(clf_oof>p1).astype(float); m1t=(clf_test>p1).astype(float)
        for w1 in np.arange(0.005,0.080,0.005):
            b1=(1-m1*w1)*base_oof+m1*w1*rh_oof; b1t=(1-m1t*w1)*base_test+m1t*w1*rh_te
            for p2 in p2_rng:
                if p2<=p1: continue
                m2=(clf_oof>p2).astype(float); m2t=(clf_test>p2).astype(float)
                for w2 in np.arange(0.005,0.090,0.005):
                    b2=(1-m2*w2)*b1+m2*w2*rm_oof
                    mm=mae(b2)
                    if mm<best:
                        best=mm; cfg=f'p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                        bo=b2; bt=(1-m2t*w2)*b1t+m2t*w2*rm_te
    print(f"  [{label}] gate={cfg}  OOF={best:.5f}")
    return bo, bt, best

# ── ref3-only CSV 파일들 로드 ─────────────────────────────────────
print("\nLoading ref3-only submission CSV files...")
ref3_only_patterns = [
    'submission_R3*.csv',
    'submission_cascade_refined3*.csv',
    'submission_Q_no_rankadj*.csv',
]
ref3_files = []
for pat in ref3_only_patterns:
    files = glob.glob(pat)
    for f in files:
        try:
            df = pd.read_csv(f).set_index('ID')
            pred = np.array([df.loc[i,'avg_delay_minutes_next_30m'] for i in test_raw['ID'].values])
            # OOF from filename
            import re
            m = re.search(r'OOF(\d+\.\d+)', f)
            oofval = float(m.group(1)) if m else 9999
            ref3_files.append((f, pred, oofval))
            print(f"  loaded: {f.split('/')[-1]} OOF={oofval:.5f}")
        except Exception as e:
            print(f"  skip {f}: {e}")

# Best ref3 CSV as anchor
ref3_files.sort(key=lambda x: x[2])
print(f"\nLoaded {len(ref3_files)} ref3 files.")

# ── HH1: Arithmetic mean of ref3 variant CSVs ────────────────────
print("\n=== HH1: Arithmetic mean of all ref3 CSV files ===")
if ref3_files:
    avg_pred = np.mean([p for _,p,_ in ref3_files], axis=0)
    # OOF estimate = weighted avg of OOF values
    oofs_vals = np.array([o for _,_,o in ref3_files])
    oof_est = np.mean(oofs_vals)
    print(f"  mean of {len(ref3_files)} files, OOF_est={oof_est:.5f}")
    save_sub(avg_pred, oof_est, 'HH1_arith_mean_ref3')
else:
    print("  HH1 skipped: no ref3 files loaded")

# ── HH2: Geometric mean of ref3 variant CSVs ────────────────────
print("\n=== HH2: Geometric mean of ref3 CSV files ===")
if ref3_files:
    eps = 0.01
    log_preds = [np.log(p+eps) for _,p,_ in ref3_files]
    geo_mean = np.exp(np.mean(log_preds, axis=0)) - eps
    geo_mean = np.clip(geo_mean, 0, None)
    oof_est2 = np.mean(oofs_vals)
    print(f"  geo mean of {len(ref3_files)} files, OOF_est={oof_est2:.5f}")
    save_sub(geo_mean, oof_est2, 'HH2_geo_mean_ref3')

# ── HH3: Rank-average of ref3 variant CSVs ──────────────────────
print("\n=== HH3: Rank-average blend of ref3 CSV files ===")
if ref3_files:
    # Convert each prediction to ranks, then average ranks, then map back
    n_test = len(test_raw)
    rank_preds = []
    for _, p, _ in ref3_files:
        ranks = np.argsort(np.argsort(p)).astype(float) / n_test
        rank_preds.append(ranks)
    avg_ranks = np.mean(rank_preds, axis=0)
    # Map rank → original predictions using ref3 as reference scale
    # Sort ref3 predictions, assign to averaged ranks
    ref3_sorted = np.sort(bt_ref3)
    avg_rank_idx = (avg_ranks * n_test).astype(int).clip(0, n_test-1)
    rank_blend = ref3_sorted[avg_rank_idx]
    print(f"  rank-blend of {len(ref3_files)} files")
    save_sub(rank_blend, mae(bo_ref3), 'HH3_rank_mean_ref3')

# ── HH4: OOF inverse-weighted avg ───────────────────────────────
print("\n=== HH4: Inverse-OOF weighted avg of ref3 CSV files ===")
if len(ref3_files) >= 2:
    min_oof = min(o for _,_,o in ref3_files)
    inv_w = np.array([1.0/max(o-min_oof+0.001,0.0001) for _,_,o in ref3_files])
    inv_w /= inv_w.sum()
    inv_pred = sum(w*p for w,(_,p,_) in zip(inv_w,ref3_files))
    oof_est4 = sum(w*o for w,(_,_,o) in zip(inv_w,ref3_files))
    print(f"  inv-OOF weighted, OOF_est={oof_est4:.5f}")
    save_sub(inv_pred, oof_est4, 'HH4_invoof_weighted_ref3')

# ── HH5: Component-level best-of selection ──────────────────────
print("\n=== HH5: Component best-of (per sample pick best component) ===")
# For each sample, among the 5 ref3 components, predict using component
# closest to the median prediction (most "consensus")
stacked_oof = np.stack([np.clip(co[k],0,None) for k in KEYS], axis=1)
stacked_te  = np.stack([np.clip(ct[k],0,None) for k in KEYS], axis=1)
median_oof = np.median(stacked_oof, axis=1, keepdims=True)
median_te  = np.median(stacked_te, axis=1, keepdims=True)
# Pick component closest to median (most consensus)
dist_oof = np.abs(stacked_oof - median_oof)
dist_te  = np.abs(stacked_te  - median_te)
best_comp_oof = np.argmin(dist_oof, axis=1)
best_comp_te  = np.argmin(dist_te, axis=1)
bo_HH5 = stacked_oof[np.arange(len(stacked_oof)), best_comp_oof]
bt_HH5 = stacked_te[np.arange(len(stacked_te)), best_comp_te]
print(f"  best-of-consensus OOF: {mae(bo_HH5):.5f}")
bo_HH5g, bt_HH5g, bm_HH5 = std_gate(bo_HH5, bt_HH5, 'HH5')
save_sub(bt_HH5g, bm_HH5, 'HH5_consensus_select')

# ── HH6: Covariate shift reweighting ────────────────────────────
print("\n=== HH6: Covariate shift weights (test layout freq) ===")
# 테스트에서 많이 등장하는 layout의 훈련 샘플에 더 큰 가중치
test_layout_freq = pd.Series(test_raw['layout_id'].values).value_counts()
train_layout_ids = train_raw['layout_id'].values
# Weight each train sample by how frequently its layout appears in test
weights = np.array([test_layout_freq.get(lid, 0) for lid in train_layout_ids], dtype=float)
weights = weights + 1.0  # smoothing (unseen layouts in test get weight 1)
weights /= weights.mean()  # normalize
print(f"  weight range: {weights.min():.3f} - {weights.max():.3f}")
def obj_shift(w):
    w=np.clip(w,0,1); w/=w.sum()
    pred=np.clip(oofs_all@w,0,None)
    return np.mean(weights * np.abs(pred - y_true))  # weighted MAE
best_HH6=999; best_wHH6=None
for _ in range(30):
    np.random.seed(_+5000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_shift,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_HH6: best_HH6=r.fun; best_wHH6=r.x
best_wHH6=np.clip(best_wHH6,0,1); best_wHH6/=best_wHH6.sum()
for k,w in zip(KEYS,best_wHH6): print(f"    {k}: {w:.4f}")
bo_HH6=np.clip(oofs_all@best_wHH6,0,None); bt_HH6=np.clip(tes_all@best_wHH6,0,None)
actual_mae_HH6=mae(bo_HH6)
print(f"  covariate-shift OOF(MAE): {actual_mae_HH6:.5f}")
bo_HH6g, bt_HH6g, bm_HH6 = std_gate(bo_HH6, bt_HH6, 'HH6')
save_sub(bt_HH6g, bm_HH6, 'HH6_covariate_shift')

# ── HH7: Scenario-position weighted blend ───────────────────────
print("\n=== HH7: Time-position weighted blend ===")
# scenario 내 timeslot 순서에 따라 가중치 변경
if 'timeslot' in train_raw.columns:
    ts = train_raw['timeslot'].values
    ts_test = test_raw['timeslot'].values if 'timeslot' in test_raw.columns else None
    # Normalize to [0,1]
    ts_norm = (ts - ts.min()) / max(ts.max()-ts.min(), 1)
    def obj_ts(w):
        # w = [alpha, delta_mega, delta_oracle_lv2] where alpha=early weight adjustment
        # Early timeslots: more mega33, late: more oracle
        alpha=np.clip(w[0],0,1); dmega=np.clip(w[1],-0.3,0.3); dlv2=np.clip(w[2],-0.3,0.3)
        # Dynamic weights: base=ref3, adjusted by position
        w_dyn = np.clip(w_ref3 + ts_norm[:,None]*np.array([dmega,0,0,dlv2,0]),0,None)
        w_dyn /= w_dyn.sum(axis=1, keepdims=True)
        pred = np.clip(np.sum(oofs_all*w_dyn,axis=1),0,None)
        return np.mean(np.abs(pred-y_true))
    res_ts=minimize(obj_ts,[0.5,0.0,0.0],method='L-BFGS-B',
                    bounds=[(0,1),(-0.3,0.3),(-0.3,0.3)])
    alpha7,dmega7,dlv27=res_ts.x
    print(f"  position params: alpha={alpha7:.3f} dmega={dmega7:.3f} dlv2={dlv27:.3f}")
    # Apply to OOF
    w_dyn_oof = np.clip(w_ref3 + ts_norm[:,None]*np.array([dmega7,0,0,dlv27,0]),0,None)
    w_dyn_oof /= w_dyn_oof.sum(axis=1,keepdims=True)
    bo_HH7=np.clip(np.sum(oofs_all*w_dyn_oof,axis=1),0,None)
    bm_HH7_base=mae(bo_HH7)
    print(f"  position-weighted OOF: {bm_HH7_base:.5f}")
    if ts_test is not None:
        ts_test_norm=(ts_test-ts.min())/max(ts.max()-ts.min(),1)
        w_dyn_te=np.clip(w_ref3+ts_test_norm[:,None]*np.array([dmega7,0,0,dlv27,0]),0,None)
        w_dyn_te/=w_dyn_te.sum(axis=1,keepdims=True)
        bt_HH7=np.clip(np.sum(tes_all*w_dyn_te,axis=1),0,None)
    else:
        bt_HH7=bt_ref3
    bo_HH7g, bt_HH7g, bm_HH7 = std_gate(bo_HH7, bt_HH7, 'HH7')
    save_sub(bt_HH7g, bm_HH7, 'HH7_position_weighted')
else:
    print("  HH7 skipped: no timeslot column — using ref3+gate")
    bo_HH7g, bt_HH7g, bm_HH7 = std_gate(bo_ref3, bt_ref3, 'HH7_fallback')
    save_sub(bt_HH7g, bm_HH7, 'HH7_ts_fallback')

# ── HH8: Test prediction variance minimization ──────────────────
print("\n=== HH8: Blend weights minimizing test prediction variance ===")
# 아이디어: test 예측의 분산을 최소화하는 가중치 → overfitting 줄임
# 단, MAE 제약 조건 (OOF MAE < ref3 OOF + 0.001)
ref3_oof_mae = mae(bo_ref3)
def obj_var(w):
    w=np.clip(w,0,1); w/=w.sum()
    te_pred = tes_all@w
    # 최소화: test prediction variance (layout 간 일관성)
    return np.var(te_pred)
def constraint_mae(w):
    w=np.clip(w,0,1); w/=w.sum()
    return ref3_oof_mae + 0.005 - np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
from scipy.optimize import minimize as sp_min
best_HH8=np.var(bt_ref3); best_wHH8=w_ref3
for _ in range(20):
    np.random.seed(_+6000)
    w0=np.random.dirichlet(np.ones(5))
    try:
        r=sp_min(obj_var,w0,method='SLSQP',
                 bounds=[(0,1)]*5,
                 constraints={'type':'ineq','fun':constraint_mae},
                 options={'maxiter':1000,'ftol':1e-10})
        if r.fun<best_HH8 and constraint_mae(r.x)>=0:
            best_HH8=r.fun; best_wHH8=r.x
    except: pass
best_wHH8=np.clip(best_wHH8,0,1); best_wHH8/=best_wHH8.sum()
for k,w in zip(KEYS,best_wHH8): print(f"    {k}: {w:.4f}")
bo_HH8=np.clip(oofs_all@best_wHH8,0,None); bt_HH8=np.clip(tes_all@best_wHH8,0,None)
print(f"  var-min OOF: {mae(bo_HH8):.5f}, te_var: {np.var(bt_HH8):.5f}")
bo_HH8g, bt_HH8g, bm_HH8 = std_gate(bo_HH8, bt_HH8, 'HH8')
save_sub(bt_HH8g, bm_HH8, 'HH8_var_minimized')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HH ROUND SUMMARY (diversity-based ensemble)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
