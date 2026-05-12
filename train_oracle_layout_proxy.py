"""
Oracle-Layout-Proxy: 순수 mega33 proxy 기반 layout oracle (train/val 불일치 없음)

문제 분석:
  layout_v3 실패 원인: train = 실제 y 통계, val = mega33 proxy → 불일치 → fold1=9.03

해결책:
  train/val/test 모두 mega33 OOF/test 예측 기반 layout 통계 사용
  GroupKFold에서 validation layout은 ALL rows가 held-out → proxy도 일관성 있음

핵심:
  GroupKFold(layout_id): val layout의 모든 행이 validation에 있음
  → ly_mean_mega(val) = mean(mega33_OOF for val layout's rows) → 자체 일관성 ✅
  → test unseen layout도 동일 방식: mean(mega33_test for that layout's rows) ✅

추가 특징 (proxy-only):
  - ly_mean_mega: layout 평균 지연 (mega33 기반)
  - ly_std_mega: layout 지연 편차
  - ly_mean_norm: 전체 평균 대비 layout 난이도
  - sc_mean_mega: 같은 시나리오 내 평균 (진행 중인 시나리오 컨텍스트)
  - row_in_sc: 시나리오 내 위치 (0-24)

출력: results/oracle_seq/oof_seqC_layout_proxy.npy
      results/oracle_seq/test_C_layout_proxy.npy
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqC_layout_proxy.npy'
OUT_TEST = 'results/oracle_seq/test_C_layout_proxy.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f"이미 존재: {OUT_OOF}")
    sys.exit(0)

t0 = time.time()
print("="*60)
print("Oracle-Layout-Proxy (proxy-only, no train/val gap)")
print("="*60)

# ── 데이터 로드 ────────────────────────────────────────────────
print("\n[1] 데이터 로드...")
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f"  train: {len(train_raw)}, test: {len(test_raw)}")

# ── v30 feature cache ──────────────────────────────────────────
print("\n[2] v30 feature cache 로드...")
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

_tr = fe_tr['train_fe']
_tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord    = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr  = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()

_te = pd.DataFrame(fe_te)
_te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord    = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_base_te  = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()
print(f"  X_base_tr: {X_base_tr.shape}, X_base_te: {X_base_te.shape}")

# ── mega33 proxy 예측 로드 ──────────────────────────────────────
print("\n[3] mega33 proxy 로드...")
with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
mega_oof = d['meta_avg_oof'][id2]

test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]
mega_test = d['meta_avg_test'][te_id2]
del d; gc.collect()

global_mean = float(np.mean(mega_oof))
global_std  = float(np.std(mega_oof))

# ── Layout/Scenario-level proxy 통계 계산 ─────────────────────
print("\n[4] Proxy 통계 계산 (mega33 기반)...")
train_raw['mega_oof'] = mega_oof
test_raw['mega_test'] = mega_test

# Layout-level (train에서 계산)
tr_ly = train_raw.groupby('layout_id')['mega_oof'].agg(['mean','std']).rename(
    columns={'mean':'tr_ly_mean','std':'tr_ly_std'}).fillna(0)
train_raw = train_raw.merge(tr_ly, on='layout_id', how='left')

# Scenario-level (train)
tr_sc = train_raw.groupby(['layout_id','scenario_id'])['mega_oof'].transform('mean')
train_raw['tr_sc_mean'] = tr_sc
train_raw['tr_sc_std']  = train_raw.groupby(['layout_id','scenario_id'])['mega_oof'].transform('std').fillna(0)

# Test: layout-level (test 자체에서 계산 — unseen layout도 자체 통계 사용)
te_ly = test_raw.groupby('layout_id')['mega_test'].agg(['mean','std']).rename(
    columns={'mean':'te_ly_mean','std':'te_ly_std'}).fillna(0)
test_raw = test_raw.merge(te_ly, on='layout_id', how='left')
te_sc = test_raw.groupby(['layout_id','scenario_id'])['mega_test'].transform('mean')
test_raw['te_sc_mean'] = te_sc
test_raw['te_sc_std']  = test_raw.groupby(['layout_id','scenario_id'])['mega_test'].transform('std').fillna(0)

def build_layout_feats_tr(df):
    """Train features: layout/sc 통계 + position (proxy-only, consistent)"""
    return np.column_stack([
        df['tr_ly_mean'].values,
        df['tr_ly_std'].values,
        df['tr_ly_mean'].values / (global_mean + 1e-6),
        df['tr_sc_mean'].values,
        df['tr_sc_std'].values,
        df['row_in_sc'].values.astype(np.float32) / 25.0,
        df['mega_oof'].values,  # mega33 자체 예측도 특징으로
    ]).astype(np.float32)

def build_layout_feats_te(df):
    """Test features: 동일 방식, test layout 통계 사용"""
    return np.column_stack([
        df['te_ly_mean'].values,
        df['te_ly_std'].values,
        df['te_ly_mean'].values / (global_mean + 1e-6),
        df['te_sc_mean'].values,
        df['te_sc_std'].values,
        df['row_in_sc'].values.astype(np.float32) / 25.0,
        df['mega_test'].values,
    ]).astype(np.float32)

lf_tr = build_layout_feats_tr(train_raw)
lf_te = build_layout_feats_te(test_raw)
X_tr_full = np.hstack([X_base_tr, lf_tr])
X_te_full = np.hstack([X_base_te, lf_te])
print(f"  X_tr_full: {X_tr_full.shape}, X_te_full: {X_te_full.shape}")

# ── LGB 5-fold 학습 ────────────────────────────────────────────
print("\n[5] LGB 5-fold 학습 (GroupKFold layout_id)...")
PARAMS = dict(
    objective='huber', n_estimators=2000, learning_rate=0.05,
    num_leaves=63, max_depth=8, min_child_samples=80,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

fold_ids = np.load('results/eda_v30/fold_idx.npy')
groups   = train_raw['layout_id'].values
gkf      = GroupKFold(n_splits=5)
oof      = np.zeros(len(train_raw))
test_preds = []

# ⚠️ 핵심: val 시 layout 통계를 OOF validation 방식으로 재계산
# GroupKFold에서 val layout은 ALL rows가 val에 있음
# → val fold의 layout 통계는 val rows의 mega33_oof로 계산 (자체 일관성)

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    # Val layout-level 통계 재계산 (val rows의 mega33 OOF 기반)
    val_df = train_raw.iloc[val_idx_sorted].copy()
    val_df['mega_oof_val'] = mega_oof[val_idx_sorted]
    val_ly_stats = val_df.groupby('layout_id')['mega_oof_val'].agg(['mean','std']).rename(
        columns={'mean':'val_ly_mean','std':'val_ly_std'}).fillna(0)
    val_df = val_df.merge(val_ly_stats, on='layout_id', how='left')
    val_sc_mean = val_df.groupby(['layout_id','scenario_id'])['mega_oof_val'].transform('mean')
    val_df['val_sc_mean'] = val_sc_mean
    val_df['val_sc_std']  = val_df.groupby(['layout_id','scenario_id'])['mega_oof_val'].transform('std').fillna(0)

    def build_val_feats(df_v):
        return np.column_stack([
            df_v['val_ly_mean'].values,
            df_v['val_ly_std'].values,
            df_v['val_ly_mean'].values / (global_mean + 1e-6),
            df_v['val_sc_mean'].values,
            df_v['val_sc_std'].values,
            df_v['row_in_sc'].values.astype(np.float32) / 25.0,
            df_v['mega_oof_val'].values,
        ]).astype(np.float32)

    lf_val = build_val_feats(val_df)
    X_val_fold = np.hstack([X_base_tr[val_idx_sorted], lf_val])

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr_full[tr_idx], y_true[tr_idx],
              eval_set=[(X_val_fold, y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    fold_pred = np.clip(model.predict(X_val_fold), 0, None)
    oof[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_pred = np.clip(model.predict(X_te_full), 0, None)
    test_preds.append(test_pred)

    print(f"  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)", flush=True)
    del model, val_df, X_val_fold; gc.collect()

overall_mae = mean_absolute_error(y_true, oof)
test_avg = np.mean(test_preds, axis=0)
print(f"\nOverall OOF: {overall_mae:.5f}  ({time.time()-t0:.0f}s total)")

np.save(OUT_OOF,  oof)
np.save(OUT_TEST, test_avg)
print(f"Saved: {OUT_OOF}, {OUT_TEST}")

# 앙상블 기여도 평가
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2_2 = [ls2[i] for i in train_raw['ID'].values]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d33['meta_avg_oof'][id2_2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2_2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2_2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2_2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2_2])
xgb_o2 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o2 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o2 = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_new_oof2 = 0.64*fixed_oof + 0.12*xgb_o2 + 0.16*lv2_o2 + 0.08*rem_o2
oracle_new_mae = mean_absolute_error(y_true, oracle_new_oof2)
corr = float(np.corrcoef(oracle_new_oof2, oof)[0,1])

print(f"\noracle_NEW OOF: {oracle_new_mae:.5f}")
print(f"layout_proxy OOF: {overall_mae:.5f}")
print(f"corr(oracle_NEW, layout_proxy): {corr:.4f}")
print()
for w in [0.03, 0.05, 0.08, 0.10, 0.15]:
    blend = (1-w)*oracle_new_oof2 + w*oof
    d = mean_absolute_error(y_true, blend) - oracle_new_mae
    print(f"  oracle_NEW*(1-{w})+proxy*{w}: delta={d:+.5f}")
print("\nDone.")
