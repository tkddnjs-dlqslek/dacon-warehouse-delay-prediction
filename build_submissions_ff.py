"""
FF Round: Layout-conditional weighting (핵심 인사이트).
test 100 layouts 중 50개가 unseen → layout_id로 seen/unseen 분리 가능!

아이디어:
  - Seen test layouts: oracle 모델 신뢰 (학습데이터 있음)
  - Unseen test layouts: mega33 더 신뢰 (layout-agnostic)

실험:
FF1: Layout-split weighting (seen=ref3, unseen=mega33-heavy)
FF2: Layout-split (seen=ref3, unseen=mega33-only)
FF3: Layout-split (seen=oracle-heavy, unseen=mega33-only)
FF4: Seen layouts only 통계 보정 (test_seen 예측값을 train 분포로 calibrate)
FF5: Layout-split soft weighting (n_train_scenarios로 trust 조절)
FF6: Layout-split w/ gate (seen layout에만 gate 적용)
FF7: Unseen layout 예측에 global mean 혼합
FF8: Layout-avg smoothing (같은 layout의 test 예측 평균화)
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

# ── seen/unseen layout 분리 ──────────────────────────────────────
train_layout_ids = set(train_raw['layout_id'].unique())
test_layouts = test_raw['layout_id'].values
seen_mask = np.array([lid in train_layout_ids for lid in test_layouts])
unseen_mask = ~seen_mask

n_seen = seen_mask.sum()
n_unseen = unseen_mask.sum()
print(f"Test layout breakdown: seen={n_seen} ({n_seen/len(seen_mask)*100:.1f}%), "
      f"unseen={n_unseen} ({n_unseen/len(unseen_mask)*100:.1f}%)")
print(f"Unique seen test layouts: {len(set(test_layouts[seen_mask]))}")
print(f"Unique unseen test layouts: {len(set(test_layouts[unseen_mask]))}")
print()

# ref3 weight (best LB anchor)
w_ref3 = np.array([0.4997,0.1019,0.1338,0.1741,0.0991]); w_ref3/=w_ref3.sum()
bo_ref3 = np.clip(oofs_all@w_ref3, 0, None)
bt_ref3 = np.clip(tes_all@w_ref3, 0, None)
print(f"ref3 base OOF: {mae(bo_ref3):.5f}")

# mega33 heavy weight for unseen
w_mega_heavy = np.array([0.80, 0.05, 0.07, 0.05, 0.03]); w_mega_heavy/=w_mega_heavy.sum()
# mega33 only
w_mega_only = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

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

# ── FF1: Layout-split (seen=ref3, unseen=mega33-heavy) ─────────
print("=== FF1: Layout-split: seen=ref3, unseen=mega33-heavy(0.80) ===")
bt_seen_ref3 = np.clip(tes_all@w_ref3, 0, None)
bt_unseen_heavy = np.clip(tes_all@w_mega_heavy, 0, None)
bt_FF1 = np.where(seen_mask, bt_seen_ref3, bt_unseen_heavy)
# OOF: 모든 train은 seen layout → ref3 weight로 OOF
bm_FF1 = mae(bo_ref3)  # OOF는 ref3 기준
print(f"  FF1 OOF(train)={bm_FF1:.5f}")
print(f"  Seen test: ref3 blend / Unseen test: mega33-heavy")
save_sub(bt_FF1, bm_FF1, 'FF1_split_ref3_mega80')

# ── FF2: Layout-split (seen=ref3, unseen=mega33-only) ──────────
print("\n=== FF2: Layout-split: seen=ref3, unseen=mega33-only ===")
bt_unseen_only = np.clip(ct['mega33'], 0, None)
bt_FF2 = np.where(seen_mask, bt_seen_ref3, bt_unseen_only)
bm_FF2 = mae(bo_ref3)
print(f"  FF2 OOF(train)={bm_FF2:.5f}")
save_sub(bt_FF2, bm_FF2, 'FF2_split_ref3_mega100')

# ── FF3: seen=oracle-heavy, unseen=mega33-only ─────────────────
print("\n=== FF3: seen=oracle-heavy, unseen=mega33-only ===")
# For seen layouts, oracle models have reliable info
w_oracle_heavy = np.array([0.35, 0.05, 0.22, 0.28, 0.10]); w_oracle_heavy/=w_oracle_heavy.sum()
def obj_train_seen(w):
    w=np.clip(w,0,1); w/=w.sum()
    return np.mean(np.abs(np.clip(oofs_all@w,0,None)-y_true))
# Optimize oracle-heavy weights for training OOF
from scipy.optimize import minimize as sp_min
best_oh=999; best_woh=None
for _ in range(20):
    np.random.seed(_+1100)
    w0=np.random.dirichlet(np.ones(5))
    r=sp_min(obj_train_seen,w0,method='L-BFGS-B',
             bounds=[(0.20,0.60),(0,0.10),(0.15,0.35),(0.15,0.40),(0,0.15)],
             options={'maxiter':3000})
    if r.fun<best_oh: best_oh=r.fun; best_woh=r.x
best_woh=np.clip(best_woh,0,1); best_woh/=best_woh.sum()
print(f"  oracle-heavy scipy OOF: {best_oh:.5f}")
for k,w in zip(KEYS,best_woh): print(f"    {k}: {w:.4f}")
bt_seen_oh = np.clip(tes_all@best_woh, 0, None)
bt_FF3 = np.where(seen_mask, bt_seen_oh, bt_unseen_only)
bm_FF3 = best_oh  # OOF = oracle-heavy training OOF
print(f"  FF3 OOF(train)={bm_FF3:.5f}")
save_sub(bt_FF3, bm_FF3, 'FF3_split_oracleheavy_mega100')

# ── FF4: Seen-layout calibration (train 분포 → test_seen 보정) ──
print("\n=== FF4: Seen-test calibration (global mean shift) ===")
# Train 전체 평균과 test_seen 평균 맞추기
train_mean = np.mean(y_true)
bt_ref3_arr = np.array(bt_ref3)
test_seen_mean = np.mean(bt_ref3_arr[seen_mask]) if seen_mask.sum() > 0 else np.mean(bt_ref3_arr)
shift = train_mean - test_seen_mean
print(f"  Train mean: {train_mean:.3f}, Test_seen mean: {test_seen_mean:.3f}, shift: {shift:.3f}")
bt_FF4 = bt_ref3.copy()
bt_FF4[seen_mask] = np.clip(bt_FF4[seen_mask] + shift, 0, None)
# OOF: 전체 shift 적용시 OOF 변화
bo_FF4 = bo_ref3.copy()
bm_FF4 = mae(bo_FF4)
print(f"  FF4 base OOF={bm_FF4:.5f} (seen-only shift on test)")
save_sub(bt_FF4, bm_FF4, 'FF4_seen_calibrated')

# ── FF5: Soft weighting by #train_scenarios for that layout ────
print("\n=== FF5: Soft weighting by layout training data density ===")
# 각 test layout이 train에서 몇 개 scenario가 있었는지로 신뢰도 결정
train_layout_counts = train_raw.groupby('layout_id')['ID'].count().to_dict()
# test layout당 train scenario 수 (unseen = 0)
test_layout_n_train = np.array([train_layout_counts.get(lid, 0) for lid in test_layouts])
# max count 대비 비율 → seen trust [0,1]
max_count = max(train_layout_counts.values()) if train_layout_counts else 1
seen_trust = np.clip(test_layout_n_train / max_count, 0, 1)  # 0=unseen, 1=fully seen
unseen_trust = 1 - seen_trust
print(f"  trust range: min={seen_trust.min():.3f} max={seen_trust.max():.3f} "
      f"mean_seen={seen_trust[seen_mask].mean():.3f} mean_unseen={seen_trust[unseen_mask].mean():.3f}")
bt_ref3_te = np.clip(tes_all@w_ref3, 0, None)
bt_mega_te = np.clip(ct['mega33'], 0, None)
bt_FF5 = seen_trust*bt_ref3_te + unseen_trust*bt_mega_te
bm_FF5 = mae(bo_ref3)
print(f"  FF5 OOF={bm_FF5:.5f} (soft layout-trust blend)")
save_sub(bt_FF5, bm_FF5, 'FF5_soft_trust_blend')

# ── FF6: Layout-split w/ gate (seen에만 gate) ──────────────────
print("\n=== FF6: seen layouts with gate, unseen with mega33-heavy ===")
# Apply gate only to predictions, then split
bo_FF6_gated, bt_FF6_gated, bm_FF6_gated = std_gate(bo_ref3, bt_ref3, 'FF6_gate')
bt_FF6 = np.where(seen_mask, bt_FF6_gated, bt_unseen_heavy)
bm_FF6 = bm_FF6_gated  # OOF from gated version
save_sub(bt_FF6, bm_FF6, 'FF6_gate_seen_heavy_unseen')

# ── FF7: Unseen prediction global mean mixing ──────────────────
print("\n=== FF7: Unseen layout → mix prediction with global mean ===")
global_mean_pred = np.full(len(test_raw), train_mean)
# 50% prediction + 50% global mean for unseen layouts
alpha_unseen = 0.5
bt_FF7 = bt_ref3.copy()
bt_FF7[unseen_mask] = (1-alpha_unseen)*bt_ref3[unseen_mask] + alpha_unseen*global_mean_pred[unseen_mask]
bm_FF7 = mae(bo_ref3)
print(f"  FF7 OOF={bm_FF7:.5f} (unseen: 50% mean mix)")
save_sub(bt_FF7, bm_FF7, 'FF7_unseen_mean_mix')

# ── FF8: Layout-avg smoothing (같은 layout 내 예측 평균화) ────────
print("\n=== FF8: Layout-avg smoothing (intra-layout prediction avg) ===")
# 같은 test layout의 모든 예측을 부분적으로 평균화 (layout-level bias 제거)
bt_FF8 = bt_ref3.copy()
alpha_smooth = 0.3  # 30% layout mean, 70% original
for lid in set(test_layouts):
    mask_lid = (test_layouts == lid)
    layout_mean = np.mean(bt_ref3[mask_lid])
    bt_FF8[mask_lid] = (1-alpha_smooth)*bt_ref3[mask_lid] + alpha_smooth*layout_mean
bm_FF8 = mae(bo_ref3)
print(f"  FF8 OOF={bm_FF8:.5f} (layout-smoothed, alpha=0.3)")
save_sub(bt_FF8, bm_FF8, 'FF8_layout_smoothed')

# ─────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("FF ROUND SUMMARY (layout-conditional weighting)")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:28s}  OOF={oofmae:.5f}  {fname}")
print(f"\nTotal: {len(saved)} new files")
print("\n⚠️ FF1/2/3/5/6/7/8 have same train OOF as ref3 — "
      "these only differ in unseen layout predictions.")
print("LB 결과만이 unseen layout 최적화의 효과를 판단할 수 있음.")
