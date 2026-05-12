"""
EE Round: mega33 heavy vs oracle-only 양극단 실험 (ref3-only components).
가설: unseen layout에서 oracle model이 seen layout에 더 특화되어 있을 수 있음.
      → mega33 비중 높이면 일반화가 나을 수도 있음.

EE1: mega33 단독 (oracle 전혀 없음) + gate
EE2: mega33(0.70) + oracle_lv2(0.30) only (2-comp, mega33 heavy)
EE3: mega33(0.80) + oracle_xgb(0.10) + oracle_lv2(0.10) (3-comp, mega33 dominant)
EE4: oracle 3개만 (oracle_xgb+lv2+rem, mega33 없음)
EE5: oracle 3개만 + gate
EE6: rank_adj 단독 (layout ranking adjustment만) + gate
EE7: mega33(0.50) + rank_adj(0.50) (oracle 전혀 없음)
EE8: ref3 w/ scipy negentropy objective (예측값 분포 다양성 최대화)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
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

def scipy_blend(keys, n_trials=20, bounds=None):
    oofs_ = np.stack([co[k] for k in keys], axis=1)
    tes_  = np.stack([ct[k] for k in keys], axis=1)
    if bounds is None: bounds = [(0,1)]*len(keys)
    def obj(w):
        w=np.clip(w,0,1); w/=w.sum()
        return np.mean(np.abs(np.clip(oofs_@w,0,None)-y_true))
    best=999; best_w=None
    for _ in range(n_trials):
        w0=np.random.dirichlet(np.ones(len(keys)))
        r=minimize(obj,w0,method='L-BFGS-B',bounds=bounds,options={'maxiter':5000,'ftol':1e-12})
        if r.fun<best: best=r.fun; best_w=r.x
    best_w=np.clip(best_w,0,1); best_w/=best_w.sum()
    for k,w in zip(keys,best_w): print(f"    {k}: {w:.4f}")
    return np.clip(oofs_@best_w,0,None), np.clip(tes_@best_w,0,None), best

# ── EE1: mega33 단독 + gate ─────────────────────────────────────
print("=== EE1: mega33 단독 (oracle 없음) + gate ===")
bo_m33 = np.clip(co['mega33'], 0, None)
bt_m33 = np.clip(ct['mega33'], 0, None)
print(f"  mega33-only OOF: {mae(bo_m33):.5f}")
bo_E1, bt_E1, bm_E1 = std_gate(bo_m33, bt_m33, 'EE1', wide=True)
save_sub(bt_E1, bm_E1, 'EE1_mega33only')

# ── EE2: mega33(0.70) + oracle_lv2(0.30) ───────────────────────
print("\n=== EE2: mega33(0.70) + oracle_lv2(0.30) ===")
def obj_E2(w):
    wm=np.clip(w[0],0.55,0.85)
    pred=np.clip(wm*co['mega33']+(1-wm)*co['oracle_lv2'],0,None)
    return np.mean(np.abs(pred-y_true))
res_E2=minimize(obj_E2,[0.70],method='L-BFGS-B',bounds=[(0.55,0.85)])
wm_E2=np.clip(res_E2.x[0],0.55,0.85)
bo_E2=np.clip(wm_E2*co['mega33']+(1-wm_E2)*co['oracle_lv2'],0,None)
bt_E2=np.clip(wm_E2*ct['mega33']+(1-wm_E2)*ct['oracle_lv2'],0,None)
print(f"  w_mega33={wm_E2:.3f}  OOF={mae(bo_E2):.5f}")
bo_E2g, bt_E2g, bm_E2 = std_gate(bo_E2, bt_E2, 'EE2', wide=True)
save_sub(bt_E2g, bm_E2, 'EE2_heavy_mega33_lv2')

# ── EE3: mega33(0.80) + xgb + lv2 (3-comp, mega33 dominant) ────
print("\n=== EE3: mega33(0.80) + oracle_xgb+lv2 (3-comp dominant) ===")
keys_E3=['mega33','oracle_xgb','oracle_lv2']
bo_E3, bt_E3, bm_E3_base = scipy_blend(keys_E3, n_trials=25,
    bounds=[(0.70,0.90),(0,0.20),(0,0.20)])
print(f"  dominant-mega33 3comp OOF: {bm_E3_base:.5f}")
bo_E3g, bt_E3g, bm_E3 = std_gate(bo_E3, bt_E3, 'EE3', wide=True)
save_sub(bt_E3g, bm_E3, 'EE3_dominant_mega33')

# ── EE4: oracle 3개만 (no mega33, no rank_adj) ─────────────────
print("\n=== EE4: oracle 3개만 (xgb+lv2+rem, no mega33) ===")
keys_E4=['oracle_xgb','oracle_lv2','oracle_rem']
bo_E4, bt_E4, bm_E4_base = scipy_blend(keys_E4, n_trials=25)
print(f"  oracle-only 3comp OOF: {bm_E4_base:.5f}")
bo_E4g, bt_E4g, bm_E4 = std_gate(bo_E4, bt_E4, 'EE4')
save_sub(bt_E4g, bm_E4, 'EE4_oracle3only')

# ── EE5: oracle 2개 (xgb+lv2) + gate ──────────────────────────
print("\n=== EE5: oracle 2개 (xgb+lv2) + gate ===")
keys_E5=['oracle_xgb','oracle_lv2']
bo_E5, bt_E5, bm_E5_base = scipy_blend(keys_E5, n_trials=25)
print(f"  oracle xgb+lv2 OOF: {bm_E5_base:.5f}")
bo_E5g, bt_E5g, bm_E5 = std_gate(bo_E5, bt_E5, 'EE5')
save_sub(bt_E5g, bm_E5, 'EE5_oracle_xgb_lv2')

# ── EE6: rank_adj 단독 + gate ───────────────────────────────────
print("\n=== EE6: rank_adj 단독 + gate ===")
bo_rk = np.clip(co['rank_adj'], 0, None)
bt_rk = np.clip(ct['rank_adj'], 0, None)
print(f"  rank_adj-only OOF: {mae(bo_rk):.5f}")
bo_E6, bt_E6, bm_E6 = std_gate(bo_rk, bt_rk, 'EE6', wide=True)
save_sub(bt_E6, bm_E6, 'EE6_rankadj_only')

# ── EE7: mega33 + rank_adj (oracle 전혀 없음) ───────────────────
print("\n=== EE7: mega33+rank_adj (oracle 없음) ===")
def obj_E7(w):
    wm=np.clip(w[0],0.3,0.9)
    pred=np.clip(wm*co['mega33']+(1-wm)*co['rank_adj'],0,None)
    return np.mean(np.abs(pred-y_true))
res_E7=minimize(obj_E7,[0.5],method='L-BFGS-B',bounds=[(0.3,0.9)])
wm_E7=np.clip(res_E7.x[0],0.3,0.9)
bo_E7=np.clip(wm_E7*co['mega33']+(1-wm_E7)*co['rank_adj'],0,None)
bt_E7=np.clip(wm_E7*ct['mega33']+(1-wm_E7)*ct['rank_adj'],0,None)
print(f"  w_mega33={wm_E7:.3f}  OOF={mae(bo_E7):.5f}")
bo_E7g, bt_E7g, bm_E7 = std_gate(bo_E7, bt_E7, 'EE7', wide=True)
save_sub(bt_E7g, bm_E7, 'EE7_mega33_rankadj_nooracle')

# ── EE8: ref3 scipy + diversity regularization ────────────────────
print("\n=== EE8: ref3 scipy w/ diversity penalty (분산 보존) ===")
# 예측값 표준편차를 훈련 타겟 std와 유사하게 유지하도록 페널티 추가
target_std = np.std(y_true)
def obj_E8(w):
    w=np.clip(w,0,1); w/=w.sum()
    pred=np.clip(oofs_all@w,0,None)
    mae_val=np.mean(np.abs(pred-y_true))
    # std 차이 패널티 (너무 과하면 안되니 작게)
    std_penalty = 0.005 * abs(np.std(pred) - target_std)
    return mae_val + std_penalty
best_E8=999; best_wE8=None
for _ in range(30):
    np.random.seed(_+1000)
    w0=np.random.dirichlet(np.ones(5))
    r=minimize(obj_E8,w0,method='L-BFGS-B',bounds=[(0,1)]*5,options={'maxiter':3000})
    if r.fun<best_E8: best_E8=r.fun; best_wE8=r.x
best_wE8=np.clip(best_wE8,0,1); best_wE8/=best_wE8.sum()
for k,w in zip(KEYS,best_wE8): print(f"    {k}: {w:.4f}")
bo_E8=np.clip(oofs_all@best_wE8,0,None); bt_E8=np.clip(tes_all@best_wE8,0,None)
actual_mae=mae(bo_E8)
print(f"  diversity-reg OOF: {actual_mae:.5f}")
bo_E8g, bt_E8g, bm_E8 = std_gate(bo_E8, bt_E8, 'EE8')
save_sub(bt_E8g, bm_E8, 'EE8_diversity_reg')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("EE ROUND SUMMARY (ref3-only, ablation & polar extremes)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
