"""
KK Round: oracle_NEW 기반 변형 실험 (과적합 없음)

핵심 발견:
  - oracle_NEW ≈ cascade_ref3 base (corr 0.9999)
  - cascade_ref3의 gate가 과적합 원인
  - oracle_NEW (no gate)가 LB 더 좋음

전략: oracle_NEW를 기반으로 layout-conditional 및 weight 변형
과적합 방지: scipy/gate 탐색 없음, 고정 가중치만

KK1: oracle_NEW 정확 복제 (submission_oracle_NEW_OOF8.3825 재생성)
KK2: oracle_NEW, unseen=mega33 only
KK3: oracle_NEW, unseen=mega33-heavy(0.80)
KK4: oracle_NEW, unseen=70% oracle_NEW + 30% mega33
KK5: oracle_NEW, unseen=50% oracle_NEW + 50% mega33
KK6: FIXED-mega33only (iter 제거) + oracle weights
KK7: oracle_NEW + iter_r4 at fixed w=0.02
KK8: oracle_NEW + xgb_v31 at fixed w=0.03 (tiny, no optimization)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')

print("="*60)
print("KK Round: oracle_NEW variants (anti-overfit)")
print("="*60)

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

# ID → ls-sorted 인덱스 (mega33/rank_adj/iter 용)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

sample_sub = pd.read_csv('sample_submission.csv')
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)

# FIXED 구성 (oracle_NEW 기반)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
r1_o = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_o = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_o = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r4_o = np.load('results/iter_pseudo/round4_oof.npy')[id2]
r1_t = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_t = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_t = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
r4_t = np.load('results/iter_pseudo/round4_test.npy')[te_id2]
rk_o = np.load('results/ranking/rank_adj_oof.npy')[id2]
rk_t = np.load('results/ranking/rank_adj_test.npy')[te_id2]
m33_o = d33['meta_avg_oof'][id2]
m33_t = d33['meta_avg_test'][te_id2]

fixed_oof = (fw['mega33']*m33_o + fw['rank_adj']*rk_o +
             fw['iter_r1']*r1_o + fw['iter_r2']*r2_o + fw['iter_r3']*r3_o)
fixed_te  = (fw['mega33']*m33_t + fw['rank_adj']*rk_t +
             fw['iter_r1']*r1_t + fw['iter_r2']*r2_t + fw['iter_r3']*r3_t)

# Oracle 컴포넌트 (row_id order, te_id2 불필요)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
v31_o  = np.load('results/oracle_seq/oof_seqC_xgb_v31.npy')
xgb_t  = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t  = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t  = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
v31_t  = np.load('results/oracle_seq/test_C_xgb_v31.npy')

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))

# oracle_NEW 기준 (KK1)
oracle_new_oof = 0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
oracle_new_te  = 0.64*fixed_te  + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t
oof_kk_base = mae(oracle_new_oof)
print(f"\noracle_NEW base OOF: {oof_kk_base:.5f}")

# seen/unseen layout 분리
train_layout_ids = set(train_raw['layout_id'].unique())
test_layouts = test_raw['layout_id'].values
seen_mask  = np.array([lid in train_layout_ids for lid in test_layouts])
unseen_mask = ~seen_mask
print(f"Test: seen={seen_mask.sum()}, unseen={unseen_mask.sum()}")

# mega33 단독 test 예측
mega33_te_only = np.clip(m33_t, 0, None)

saved = []
def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

# ── KK1: oracle_NEW 정확 복제 ──────────────────────────────────
print("\n=== KK1: oracle_NEW 정확 복제 (0.64F+0.12X+0.16L+0.08R) ===")
save_sub(oracle_new_te, oof_kk_base, 'KK1_oracle_NEW_exact')

# ── KK2: unseen=mega33 only ────────────────────────────────────
print("\n=== KK2: seen=oracle_NEW, unseen=mega33-only ===")
bt_KK2 = oracle_new_te.copy()
bt_KK2[unseen_mask] = mega33_te_only[unseen_mask]
print(f"  OOF={oof_kk_base:.5f} [unseen: mega33 단독]")
save_sub(bt_KK2, oof_kk_base, 'KK2_oracle_NEW_unseen_mega100')

# ── KK3: unseen=mega33-heavy(0.80) ────────────────────────────
print("\n=== KK3: seen=oracle_NEW, unseen=mega33-heavy(0.80) ===")
bt_KK3 = oracle_new_te.copy()
bt_KK3[unseen_mask] = 0.80*mega33_te_only[unseen_mask] + 0.20*oracle_new_te[unseen_mask]
print(f"  OOF={oof_kk_base:.5f} [unseen: 80% mega33 + 20% oracle_NEW]")
save_sub(bt_KK3, oof_kk_base, 'KK3_oracle_NEW_unseen_mega80')

# ── KK4: unseen=70% oracle_NEW + 30% mega33 ───────────────────
print("\n=== KK4: unseen=70% oracle_NEW + 30% mega33 ===")
bt_KK4 = oracle_new_te.copy()
bt_KK4[unseen_mask] = 0.70*oracle_new_te[unseen_mask] + 0.30*mega33_te_only[unseen_mask]
print(f"  OOF={oof_kk_base:.5f} [unseen: 70% oracle + 30% mega33]")
save_sub(bt_KK4, oof_kk_base, 'KK4_oracle_NEW_unseen_blend70_30')

# ── KK5: unseen=50% oracle_NEW + 50% mega33 ───────────────────
print("\n=== KK5: unseen=50% oracle_NEW + 50% mega33 ===")
bt_KK5 = oracle_new_te.copy()
bt_KK5[unseen_mask] = 0.50*oracle_new_te[unseen_mask] + 0.50*mega33_te_only[unseen_mask]
print(f"  OOF={oof_kk_base:.5f} [unseen: 50% oracle + 50% mega33]")
save_sub(bt_KK5, oof_kk_base, 'KK5_oracle_NEW_unseen_blend50_50')

# ── KK6: FIXED-mega33only (iter 제거) ─────────────────────────
print("\n=== KK6: FIXED = mega33 only (iter 없음, 강한 mega33) ===")
# iter rounds 제거, mega33 가중치만 FIXED로
fixed_no_iter_oof = m33_o + 0.15*rk_o  # 간단 mega33+rank
fixed_no_iter_te  = m33_t + 0.15*rk_t
# 정규화
fixed_ni_oof = fixed_no_iter_oof / 1.15
fixed_ni_te  = fixed_no_iter_te  / 1.15
oracle_ni_oof = 0.64*fixed_ni_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
oracle_ni_te  = 0.64*fixed_ni_te  + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t
oof_kk6 = mae(oracle_ni_oof)
print(f"  OOF={oof_kk6:.5f} [FIXED=mega33+rank only, no iter]")
save_sub(oracle_ni_te, oof_kk6, 'KK6_oracle_no_iter')

# ── KK7: oracle_NEW + iter_r4 (w=0.02, 고정) ──────────────────
print("\n=== KK7: oracle_NEW + iter_r4 (fixed w=0.02) ===")
oracle_r4_oof = (0.98*oracle_new_oof + 0.02*r4_o)
oracle_r4_te  = (0.98*oracle_new_te  + 0.02*r4_t)
oof_kk7 = mae(oracle_r4_oof)
print(f"  OOF={oof_kk7:.5f} [oracle_NEW + 2% iter_r4]")
save_sub(oracle_r4_te, oof_kk7, 'KK7_oracle_NEW_plus_r4')

# ── KK8: oracle_NEW + xgb_v31 (fixed w=0.03) ─────────────────
print("\n=== KK8: oracle_NEW + xgb_v31 (fixed w=0.03) ===")
oracle_v31_oof = 0.97*oracle_new_oof + 0.03*v31_o
oracle_v31_te  = 0.97*oracle_new_te  + 0.03*v31_t
oof_kk8 = mae(oracle_v31_oof)
print(f"  OOF={oof_kk8:.5f} [oracle_NEW + 3% xgb_v31]")
save_sub(oracle_v31_te, oof_kk8, 'KK8_oracle_NEW_plus_xgbv31')

# ── KK9: oracle lv2 heavy (0.24) ──────────────────────────────
print("\n=== KK9: oracle lv2 heavy (lv2 0.24, rem 0.08, xgb 0.04) ===")
# lv2가 개별 OOF 8.4409, 가장 다양한 oracle 중 하나
o9_oof = 0.64*fixed_oof + 0.04*xgb_o + 0.24*lv2_o + 0.08*rem_o
o9_te  = 0.64*fixed_te  + 0.04*xgb_t + 0.24*lv2_t + 0.08*rem_t
oof_kk9 = mae(o9_oof)
print(f"  OOF={oof_kk9:.5f} [lv2-heavy oracle]")
save_sub(o9_te, oof_kk9, 'KK9_oracle_lv2heavy')

# ── KK10: oracle_NEW, unseen layout avg smoothing ─────────────
print("\n=== KK10: Layout-avg smoothing for unseen layouts ===")
bt_KK10 = oracle_new_te.copy()
alpha_smooth = 0.3
for lid in set(test_layouts[unseen_mask]):
    mask_lid = (test_layouts == lid)
    layout_mean = np.mean(oracle_new_te[mask_lid])
    bt_KK10[mask_lid] = (1-alpha_smooth)*oracle_new_te[mask_lid] + alpha_smooth*layout_mean
print(f"  OOF={oof_kk_base:.5f} [unseen: 30% layout-mean smoothing]")
save_sub(bt_KK10, oof_kk_base, 'KK10_oracle_NEW_smooth_unseen')

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("KK ROUND SUMMARY")
print("oracle_NEW base → layout-conditional & weight variants")
print()
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    marker = ' ← best' if oofmae == min(o for _,o,_ in saved) else ''
    print(f"  {label:<42}  OOF={oofmae:.5f}{marker}")
print(f"\nTotal: {len(saved)} files")
print("\n⚠️ 모든 파일: OOF는 참고용 (LB만이 실제 판단 기준)")
print("   gate 없음 = 과적합 없음")
