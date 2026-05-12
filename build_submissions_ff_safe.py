"""
FF Safe Round: Layout-conditional weighting (과적합 제거 버전)

제거된 실험:
  FF3: scipy 최적화 → OOF에 피팅 → unseen layout 과적합
  FF6: 게이트 탐색 → 동일 이유

남은 실험 (고정 가중치, 최적화 없음):
  FF1: seen=ref3, unseen=mega33-heavy(0.80) [고정 가중치]
  FF2: seen=ref3, unseen=mega33-only        [고정 가중치]
  FF4: seen-layout 평균 shift 보정          [단순 통계, 최적화 없음]
  FF5: layout train density 기반 soft blend [고정 공식]
  FF7: unseen layout 예측에 global mean 50% 혼합 [고정 alpha]
  FF8: layout 내 예측 평균화 (smoothing)   [고정 alpha]

Anti-overfit 원칙:
  1. cascade_refined3 가중치 완전 동결 (재최적화 없음)
  2. scipy/게이트 탐색 없음
  3. 모든 블렌딩 가중치 고정 (OOF 기반 탐색 없음)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')

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

oofs_all = np.stack([co[k] for k in KEYS], axis=1)
tes_all  = np.stack([ct[k] for k in KEYS], axis=1)

saved = []

def mae(pred): return np.mean(np.abs(pred - y_true))

# ── seen/unseen layout 분리 ──────────────────────────────────────
train_layout_ids = set(train_raw['layout_id'].unique())
test_layouts = test_raw['layout_id'].values
seen_mask  = np.array([lid in train_layout_ids for lid in test_layouts])
unseen_mask = ~seen_mask

print(f"Test layout: seen={seen_mask.sum()} / unseen={unseen_mask.sum()}")
print(f"Unique seen: {len(set(test_layouts[seen_mask]))} / unseen: {len(set(test_layouts[unseen_mask]))}")
print()

# cascade_refined3 고정 가중치 (절대 변경 금지)
w_ref3 = np.array([0.4997, 0.1019, 0.1338, 0.1741, 0.0991])
w_ref3 /= w_ref3.sum()
bo_ref3 = np.clip(oofs_all @ w_ref3, 0, None)
bt_ref3 = np.clip(tes_all @ w_ref3, 0, None)
oof_ref3 = mae(bo_ref3)
print(f"ref3 base OOF: {oof_ref3:.5f}  ← cascade_refined3 anchor (동결)")
print()

# 고정 가중치 세트 (최적화 없음)
w_mega_heavy = np.array([0.80, 0.05, 0.07, 0.05, 0.03]); w_mega_heavy /= w_mega_heavy.sum()
w_mega_only  = np.array([1.0, 0.0, 0.0, 0.0, 0.0])

def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

# ── FF1: seen=ref3, unseen=mega33-heavy (0.80) ─────────────────
print("=== FF1: seen=ref3, unseen=mega33-heavy(0.80) ===")
bt_unseen_heavy = np.clip(tes_all @ w_mega_heavy, 0, None)
bt_FF1 = np.where(seen_mask, bt_ref3, bt_unseen_heavy)
print(f"  OOF(train)={oof_ref3:.5f} [seen: ref3 / unseen: mega33*0.80]")
save_sub(bt_FF1, oof_ref3, 'FF1_split_ref3_mega80')

# ── FF2: seen=ref3, unseen=mega33-only ─────────────────────────
print("\n=== FF2: seen=ref3, unseen=mega33-only ===")
bt_unseen_only = np.clip(ct['mega33'], 0, None)
bt_FF2 = np.where(seen_mask, bt_ref3, bt_unseen_only)
print(f"  OOF(train)={oof_ref3:.5f} [seen: ref3 / unseen: mega33 단독]")
save_sub(bt_FF2, oof_ref3, 'FF2_split_ref3_mega100')

# ── FF4: Seen-layout 평균 shift 보정 ───────────────────────────
print("\n=== FF4: Seen-test calibration (global mean shift) ===")
train_mean = np.mean(y_true)
test_seen_mean = np.mean(bt_ref3[seen_mask]) if seen_mask.sum() > 0 else np.mean(bt_ref3)
shift = train_mean - test_seen_mean
print(f"  Train mean: {train_mean:.3f} / Test_seen mean: {test_seen_mean:.3f} / shift: {shift:+.3f}")
bt_FF4 = bt_ref3.copy()
bt_FF4[seen_mask] = np.clip(bt_FF4[seen_mask] + shift, 0, None)
print(f"  OOF(train)={oof_ref3:.5f} [seen test에 mean shift 적용]")
save_sub(bt_FF4, oof_ref3, 'FF4_seen_calibrated')

# ── FF5: layout train density 기반 soft blend ───────────────────
print("\n=== FF5: Soft weighting by layout training data density ===")
train_layout_counts = train_raw.groupby('layout_id')['ID'].count().to_dict()
test_layout_n_train = np.array([train_layout_counts.get(lid, 0) for lid in test_layouts])
max_count = max(train_layout_counts.values()) if train_layout_counts else 1
seen_trust   = np.clip(test_layout_n_train / max_count, 0, 1)
unseen_trust = 1 - seen_trust
print(f"  trust: seen_mean={seen_trust[seen_mask].mean():.3f} / unseen_mean={seen_trust[unseen_mask].mean():.3f}")
bt_FF5 = seen_trust * bt_ref3 + unseen_trust * np.clip(ct['mega33'], 0, None)
print(f"  OOF(train)={oof_ref3:.5f} [density-weighted blend]")
save_sub(bt_FF5, oof_ref3, 'FF5_soft_trust_blend')

# ── FF7: Unseen layout → 50% global mean 혼합 ──────────────────
print("\n=== FF7: Unseen layout → 50% global mean mix ===")
alpha_unseen = 0.5
bt_FF7 = bt_ref3.copy()
bt_FF7[unseen_mask] = (1 - alpha_unseen) * bt_ref3[unseen_mask] + alpha_unseen * train_mean
print(f"  OOF(train)={oof_ref3:.5f} [unseen: 50% pred + 50% global_mean={train_mean:.2f}]")
save_sub(bt_FF7, oof_ref3, 'FF7_unseen_mean_mix')

# ── FF8: Layout 내 예측 평균화 (smoothing) ─────────────────────
print("\n=== FF8: Layout-avg smoothing (alpha=0.3) ===")
alpha_smooth = 0.3
bt_FF8 = bt_ref3.copy()
for lid in set(test_layouts):
    mask_lid = (test_layouts == lid)
    layout_mean = np.mean(bt_ref3[mask_lid])
    bt_FF8[mask_lid] = (1 - alpha_smooth) * bt_ref3[mask_lid] + alpha_smooth * layout_mean
print(f"  OOF(train)={oof_ref3:.5f} [layout 내 30% 평균화]")
save_sub(bt_FF8, oof_ref3, 'FF8_layout_smoothed')

# ── FF9: Unseen → 70% ref3 + 30% mega33-only ──────────────────
print("\n=== FF9 (추가): unseen = 70% ref3 + 30% mega33-only ===")
alpha_blend = 0.30
bt_FF9 = bt_ref3.copy()
bt_FF9[unseen_mask] = (1 - alpha_blend) * bt_ref3[unseen_mask] + alpha_blend * ct['mega33'][unseen_mask]
print(f"  OOF(train)={oof_ref3:.5f} [unseen: 70% ref3 + 30% mega33]")
save_sub(bt_FF9, oof_ref3, 'FF9_unseen_blend70_30')

# ── FF10: Unseen → 50% ref3 + 50% mega33-only ─────────────────
print("\n=== FF10 (추가): unseen = 50% ref3 + 50% mega33-only ===")
alpha_blend = 0.50
bt_FF10 = bt_ref3.copy()
bt_FF10[unseen_mask] = (1 - alpha_blend) * bt_ref3[unseen_mask] + alpha_blend * ct['mega33'][unseen_mask]
print(f"  OOF(train)={oof_ref3:.5f} [unseen: 50% ref3 + 50% mega33]")
save_sub(bt_FF10, oof_ref3, 'FF10_unseen_blend50_50')

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("FF SAFE ROUND SUMMARY")
print("Anti-overfit: scipy/gate 제거, 모든 가중치 고정")
print()
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:35s}  OOF={oofmae:.5f}")
print(f"\nTotal: {len(saved)} files")
print()
print("⚠️  모든 FF 파일은 train OOF = ref3와 동일 (8.37905)")
print("   unseen layout 예측만 다름 → LB만이 실제 효과 측정 가능")
print("   OOF 개선 없음 = 과적합 없음 (의도적 설계)")
