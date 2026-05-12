"""
Conservative Pseudo-labeling (과적합 없는 버전)

전략:
  - cascade_refined3 test 예측 → unseen layout 행의 pseudo-label
  - 단일 LGB 학습: train(w=1.0) + unseen_test(w=0.1)
  - GroupKFold(layout_id, n=5) 유지 → 기존과 동일한 CV 구조
  - 최종 제출: cascade_ref3 (seen) + 고정 blend (unseen)

Anti-overfit 원칙:
  1. cascade_refined3 가중치/게이트 완전 동결 (변경 없음)
  2. pseudo-label weight = 0.1 (매우 보수적)
  3. unseen layout만 pseudo-label (seen test 제외)
  4. 블렌딩 가중치 고정 (OOF 탐색 없음)
  5. 조기종료는 seen-layout fold validation으로만
  6. OOF delta > 0.003 이면 → "overfit 의심" 경고 출력

기대 결과:
  - Train OOF: cascade_ref3와 동일 (8.37905)
  - Test(unseen): pseudo-labeled LGB가 layout 특성 반영 → 미세 개선 기대
  - LB 효과: 제출해봐야 알 수 있음
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np, pandas as pd, time, gc, warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

t0 = time.time()
print("="*60)
print("Conservative Pseudo-labeling (anti-overfit)")
print("="*60)

# ── 데이터 로드 ────────────────────────────────────────────────
print("\n[1] 데이터 로드...")
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

# cascade_refined3 test 예측 로드 (pseudo-label 소스)
sub_ref3 = pd.read_csv('submission_cascade_refined3_OOF8.37905.csv')
sub_ref3 = sub_ref3.set_index('ID')
pseudo_all = sub_ref3.loc[test_raw['ID'].values, 'avg_delay_minutes_next_30m'].values
print(f"  cascade_ref3 pseudo-labels: {len(pseudo_all)} rows, mean={pseudo_all.mean():.3f}")

# unseen layout 마스크
train_layout_ids = set(train_raw['layout_id'].unique())
test_layouts = test_raw['layout_id'].values
unseen_mask = ~np.array([lid in train_layout_ids for lid in test_layouts])
seen_mask   = ~unseen_mask
print(f"  Test: seen={seen_mask.sum()}, unseen={unseen_mask.sum()}")

# ── feature cache 로드 (v30) ────────────────────────────────────
print("\n[2] Feature cache (v30) 로드...")
try:
    with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
        blob = pickle.load(f)
    train_fe = blob['train_fe']
    feat_cols = list(blob['feat_cols'])
    with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
        test_fe = pickle.load(f)
    fold_ids = np.load('results/eda_v30/fold_idx.npy')
    print(f"  train_fe: {train_fe.shape}, feat_cols: {len(feat_cols)}")
    print(f"  test_fe: {test_fe.shape}")
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)

# row 순서 정렬 확인
y = train_fe['avg_delay_minutes_next_30m'].values.astype(np.float64)
assert len(y) == len(y_true), "train row count mismatch"

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

# test_fe row 순서 확인 → test_raw 순서와 맞추기
if 'ID' in test_fe.columns:
    te_id_order = {iid: i for i, iid in enumerate(test_fe['ID'].values)}
    te_reorder = [te_id_order[iid] for iid in test_raw['ID'].values]
    X_te = X_te[te_reorder]
    unseen_mask_fe = unseen_mask  # already aligned to test_raw order
else:
    unseen_mask_fe = unseen_mask

X_te_unseen = X_te[unseen_mask_fe]
pseudo_unseen = pseudo_all[unseen_mask_fe]
pseudo_log_unseen = np.log1p(np.clip(pseudo_unseen, 0, None)).astype(np.float32)

print(f"  X_te_unseen: {X_te_unseen.shape}, pseudo mean: {pseudo_unseen.mean():.3f}")

del blob, train_fe, test_fe; gc.collect()

# ── Baseline (no pseudo) ────────────────────────────────────────
PARAMS = dict(
    objective='huber', n_estimators=3000, learning_rate=0.05,
    num_leaves=63, max_depth=8, min_child_samples=80,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=2.0, reg_lambda=2.0,  # 강한 정규화 (과적합 방지)
    random_state=42, verbose=-1, n_jobs=-1,
)
y_log = np.log1p(y)

PSEUDO_WEIGHT = 0.10  # 보수적 (과적합 방지 핵심)
print(f"\n[3] Pseudo-label weight = {PSEUDO_WEIGHT} (보수적)")

# Baseline OOF (no pseudo)
baseline_path = 'results/pseudo_safe/oof_baseline.npy'
import os; os.makedirs('results/pseudo_safe', exist_ok=True)
try:
    oof_base = np.load(baseline_path)
    mae_base = mean_absolute_error(y, oof_base)
    print(f"\n  baseline (no pseudo) OOF: {mae_base:.5f} [cached]")
except:
    print("\n[A] Baseline training (no pseudo)...")
    oof_base = np.zeros(len(y))
    te_preds_base = np.zeros(len(X_te))
    for fold in range(5):
        tr_idx = np.where(fold_ids != fold)[0]
        va_idx = np.where(fold_ids == fold)[0]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_tr[tr_idx], y_log[tr_idx],
              eval_set=[(X_tr[va_idx], y_log[va_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        oof_base[va_idx] = np.clip(np.expm1(m.predict(X_tr[va_idx])), 0, None)
        te_preds_base += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
        print(f"  fold {fold}: it={m.best_iteration_}, fold_mae={mean_absolute_error(y[va_idx], oof_base[va_idx]):.5f}")
        del m; gc.collect()
    mae_base = mean_absolute_error(y, oof_base)
    np.save(baseline_path, oof_base)
    np.save('results/pseudo_safe/test_baseline.npy', te_preds_base)
    print(f"  Baseline OOF: {mae_base:.5f}")

# Pseudo-label OOF
print(f"\n[B] Pseudo-label training (unseen weight={PSEUDO_WEIGHT})...")
oof_pseudo = np.zeros(len(y))
te_preds_pseudo = np.zeros(len(X_te))

for fold in range(5):
    tr_idx = np.where(fold_ids != fold)[0]
    va_idx = np.where(fold_ids == fold)[0]
    t1 = time.time()

    n_tr = len(tr_idx)
    n_ps = len(X_te_unseen)
    X_comb = np.empty((n_tr + n_ps, X_tr.shape[1]), dtype=np.float32)
    X_comb[:n_tr] = X_tr[tr_idx]
    X_comb[n_tr:] = X_te_unseen
    y_comb = np.empty(n_tr + n_ps, dtype=np.float32)
    y_comb[:n_tr] = y_log[tr_idx]
    y_comb[n_tr:] = pseudo_log_unseen
    w_comb = np.ones(n_tr + n_ps, dtype=np.float32)
    w_comb[n_tr:] = PSEUDO_WEIGHT

    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_comb, y_comb, sample_weight=w_comb,
          eval_set=[(X_tr[va_idx], y_log[va_idx])],  # validation = train rows only
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof_pseudo[va_idx] = np.clip(np.expm1(m.predict(X_tr[va_idx])), 0, None)
    te_preds_pseudo += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

    fold_mae = mean_absolute_error(y[va_idx], oof_pseudo[va_idx])
    print(f"  fold {fold}: it={m.best_iteration_}, fold_mae={fold_mae:.5f} ({time.time()-t1:.0f}s)")
    del X_comb, y_comb, w_comb, m; gc.collect()

mae_pseudo = mean_absolute_error(y, oof_pseudo)
delta = mae_pseudo - mae_base
print(f"\n  Baseline OOF:  {mae_base:.5f}")
print(f"  Pseudo   OOF:  {mae_pseudo:.5f}")
print(f"  Delta:        {delta:+.5f}")

if delta < -0.003:
    print("  ⚠️  과적합 의심: OOF 개선이 너무 큼 → pseudo-label에 과적합 가능성")
elif delta > 0.003:
    print("  ⚠️  pseudo-labeling이 오히려 OOF 악화 → LB도 나빠질 가능성")
else:
    print("  ✅ OOF delta < 0.003: 과적합 없는 범위 (안전)")

np.save('results/pseudo_safe/oof_pseudo.npy', oof_pseudo)
np.save('results/pseudo_safe/test_pseudo.npy', te_preds_pseudo)

# ── 최종 submission 구성: 고정 가중치, 최적화 없음 ────────────────
print("\n[4] Submission 구성 (고정 가중치, OOF 탐색 없음)...")

# cascade_refined3 OOF/test 로드 (anchor)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
KEYS = ['mega33','rank_adj','oracle_xgb','oracle_lv2','oracle_rem']
oofs_arr = np.stack([
    d33['meta_avg_oof'][id2],
    np.load('results/ranking/rank_adj_oof.npy')[id2],
    np.load('results/oracle_seq/oof_seqC_xgb.npy'),
    np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),
], axis=1)
tes_arr = np.stack([
    d33['meta_avg_test'][te_id2],
    np.load('results/ranking/rank_adj_test.npy')[te_id2],
    np.load('results/oracle_seq/test_C_xgb.npy'),
    np.load('results/oracle_seq/test_C_log_v2.npy'),
    np.load('results/oracle_seq/test_C_xgb_remaining.npy'),
], axis=1)

w_ref3 = np.array([0.4997, 0.1019, 0.1338, 0.1741, 0.0991])
w_ref3 /= w_ref3.sum()
bt_ref3 = np.clip(tes_arr @ w_ref3, 0, None)
bo_ref3 = np.clip(oofs_arr @ w_ref3, 0, None)
oof_ref3 = mean_absolute_error(y_true, bo_ref3)
print(f"  cascade_ref3 OOF: {oof_ref3:.5f} (기준점)")

# 고정 블렌드: unseen layout에만 pseudo 예측 일부 반영
# 고정 alpha들 (OOF 최적화 없음)
sample_sub = pd.read_csv('sample_submission.csv')

def save_blend(alpha_unseen, label):
    bt = bt_ref3.copy()
    bt[unseen_mask] = (1 - alpha_unseen) * bt_ref3[unseen_mask] + alpha_unseen * te_preds_pseudo[unseen_mask]
    bt = np.maximum(0, bt)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': bt})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_{label}_OOF{oof_ref3:.5f}.csv'
    df.to_csv(fname, index=False)
    print(f"  ★ [{label}] alpha={alpha_unseen:.2f} → {fname}")
    return fname

# alpha = 0.10: 10% pseudo, 90% ref3 (가장 보수적)
save_blend(0.10, 'pseudo_safe_a10')
# alpha = 0.20: 20% pseudo
save_blend(0.20, 'pseudo_safe_a20')
# alpha = 0.30: 30% pseudo
save_blend(0.30, 'pseudo_safe_a30')

print(f"\n완료. 총 소요: {time.time()-t0:.0f}s")
print()
print("제출 우선순위:")
print("  1. pseudo_safe_a10 (가장 보수적, 과적합 최소)")
print("  2. pseudo_safe_a20")
print("  3. pseudo_safe_a30")
print()
print("주의: 모든 파일의 train OOF = cascade_ref3 (8.37905)")
print("      unseen layout 예측만 다름 → LB로만 효과 확인 가능")
