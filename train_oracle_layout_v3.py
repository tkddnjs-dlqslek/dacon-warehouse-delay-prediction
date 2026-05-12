"""
Oracle-Layout-v3: Layout-normalized oracle (핵심 개선)

핵심 아이디어:
  - 기존 oracle: 절대값 lag 사용 → unseen layout에 전이 어려움
  - v3: layout-level로 정규화한 lag 사용 → layout 간 패턴 전이 가능

Oracle feature (train): 실제 y 기반 lag를 layout 평균으로 나눔
                        y_lag1 / layout_mean_y → "이 layout 평균 대비 얼마?"
Proxy feature (val/test): mega33 예측 기반 lag를 mega33 layout 평균으로 나눔
                          mega33_lag1 / layout_mean_mega33

핵심: 정규화된 lag는 layout-agnostic → unseen layout에도 유효

출력: results/oracle_seq/oof_seqC_layout_v3.npy
      results/oracle_seq/test_C_layout_v3.npy

GroupKFold(layout_id, n=5) 유지
Kill if fold1 MAE > 8.9
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import xgboost as xgb
from sklearn.model_selection import GroupKFold
import gc

KILL_THRESH = 8.9
OUT_OOF  = 'results/oracle_seq/oof_seqC_layout_v3.npy'
OUT_TEST = 'results/oracle_seq/test_C_layout_v3.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print("Already exists. Skipping training.")
    import sys; sys.exit(0)

print("="*60)
print("Oracle-Layout-v3: Layout-normalized oracle")
print("="*60)
t0 = time.time()

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

# v30 feature cache (메모리 효율)
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

# mega33 proxy 예측
print("\n[3] mega33 proxy 로드...")
with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id2]

test_ls     = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos   = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2      = [te_ls_pos[i] for i in test_raw['ID'].values]
mega_test_id= d['meta_avg_test'][te_id2]
del d; gc.collect()

# ── Layout-level statistics ─────────────────────────────────────
print("\n[4] Layout-level 통계 계산...")

# Train: 실제 y 기반 layout 통계
train_raw['y'] = y_true
train_raw['mega_pred'] = mega_oof_id
ly_stats = train_raw.groupby('layout_id').agg(
    ly_mean_y   = ('y', 'mean'),
    ly_std_y    = ('y', 'std'),
    ly_mean_mega= ('mega_pred', 'mean'),
    ly_std_mega = ('mega_pred', 'std'),
).fillna(0)
global_mean_y    = float(y_true.mean())
global_std_y     = float(y_true.std())
global_mean_mega = float(mega_oof_id.mean())

train_raw = train_raw.merge(ly_stats, on='layout_id', how='left')
train_raw['ly_mean_y']    = train_raw['ly_mean_y'].fillna(global_mean_y)
train_raw['ly_std_y']     = train_raw['ly_std_y'].fillna(global_std_y)
train_raw['ly_mean_mega'] = train_raw['ly_mean_mega'].fillna(global_mean_mega)

# Test: mega33 예측 기반 layout 통계 (unseen layout도 test 내에서 계산)
test_raw['mega_pred'] = mega_test_id
te_ly_stats = test_raw.groupby('layout_id').agg(
    te_ly_mean_mega = ('mega_pred', 'mean'),
    te_ly_std_mega  = ('mega_pred', 'std'),
).fillna(0)
test_raw = test_raw.merge(te_ly_stats, on='layout_id', how='left')
test_raw['te_ly_mean_mega'] = test_raw['te_ly_mean_mega'].fillna(global_mean_mega)
test_raw['te_ly_std_mega']  = test_raw['te_ly_std_mega'].fillna(1.0)

# ── Oracle 특징 구성 ─────────────────────────────────────────────
print("\n[5] Oracle 특징 구성 (layout-normalized)...")

def build_oracle_feats_train(df):
    """Train: 실제 y 기반 lag, layout 평균으로 정규화"""
    eps = 1e-6
    ly_mean = df['ly_mean_y'].values
    # lag-based (cumulative within scenario, computed from actual y)
    # 여기선 단순히 layout-level 통계를 특징으로 사용
    feats = np.column_stack([
        df['ly_mean_y'].values,                              # 절대 layout 평균
        df['ly_std_y'].values,                               # layout 표준편차
        df['ly_mean_y'].values / (global_mean_y + eps),      # 정규화된 layout 평균
        df['row_in_sc'].values / 25.0,                       # scenario 내 진행도
        df['sc_mean_proxy'].values,                          # scenario-level proxy
        df['sc_mean_proxy'].values / (df['ly_mean_mega'].values + eps),  # proxy 정규화
    ])
    return feats.astype(np.float32)

def build_proxy_feats_test(df):
    """Test: mega33 proxy 기반, layout 평균으로 정규화"""
    eps = 1e-6
    feats = np.column_stack([
        df['te_ly_mean_mega'].values,                        # mega 기반 layout 평균 (seen/unseen 모두)
        df['te_ly_std_mega'].values,                         # layout std
        df['te_ly_mean_mega'].values / (global_mean_mega + eps),  # 정규화
        df['row_in_sc'].values / 25.0,
        df['sc_mean_proxy_te'].values,
        df['sc_mean_proxy_te'].values / (df['te_ly_mean_mega'].values + eps),
    ])
    return feats.astype(np.float32)

# SC-level proxy for train validation & test
sc_mega = train_raw.groupby(['layout_id','scenario_id'])['mega_pred']
train_raw['sc_mean_proxy'] = sc_mega.transform('mean')
train_raw['sc_std_proxy']  = sc_mega.transform('std').fillna(0)

te_sc_mega = test_raw.groupby(['layout_id','scenario_id'])['mega_pred']
test_raw['sc_mean_proxy_te'] = te_sc_mega.transform('mean')
test_raw['sc_std_proxy_te']  = te_sc_mega.transform('std').fillna(0)

# 학습 특징 행렬 구성 (base + oracle_feats)
oracle_feats_tr = build_oracle_feats_train(train_raw)
X_tr_full = np.hstack([X_base_tr, oracle_feats_tr])

# Test 특징 (proxy 기반)
def build_proxy_feats_test_v2(df):
    eps = 1e-6
    feats = np.column_stack([
        df['te_ly_mean_mega'].values,
        df['te_ly_std_mega'].values,
        df['te_ly_mean_mega'].values / (global_mean_mega + eps),
        df['row_in_sc'].values / 25.0,
        df['sc_mean_proxy_te'].values,
        df['sc_mean_proxy_te'].values / (df['te_ly_mean_mega'].values + eps),
    ])
    return feats.astype(np.float32)

oracle_feats_te = build_proxy_feats_test_v2(test_raw)
X_te_full = np.hstack([X_base_te, oracle_feats_te])

print(f"  X_tr_full: {X_tr_full.shape}, X_te_full: {X_te_full.shape}")

# Proxy 특징 for VALIDATION (use mega33-based proxy, not actual y-based)
def build_proxy_feats_val(df_rows):
    """Validation: proxy 기반 (train rows지만 mega33 proxy 사용)"""
    eps = 1e-6
    feats = np.column_stack([
        df_rows['ly_mean_mega'].values,
        df_rows['ly_std_mega'].values,
        df_rows['ly_mean_mega'].values / (global_mean_mega + eps),
        df_rows['row_in_sc'].values / 25.0,
        df_rows['sc_mean_proxy'].values,
        df_rows['sc_mean_proxy'].values / (df_rows['ly_mean_mega'].values + eps),
    ])
    return feats.astype(np.float32)

X_tr_proxy_full = np.hstack([X_base_tr, build_proxy_feats_val(train_raw)])

# ── XGB 학습 ─────────────────────────────────────────────────────
print("\n[6] XGB 5-fold 학습 (GroupKFold layout_id)...")
XGB_PARAMS = dict(
    objective='reg:absoluteerror', n_estimators=2000, learning_rate=0.05,
    max_depth=7, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.5, reg_lambda=0.5, n_jobs=4, random_state=42, verbosity=0,
    early_stopping_rounds=100, eval_metric='mae'
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof    = np.full(len(train_raw), np.nan)
test_preds = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    # Train: oracle 특징 (실제 y 기반)
    X_tr_fold = X_tr_full[tr_idx]
    # Val: proxy 특징 (mega33 기반) — train/test gap 시뮬레이션
    X_val_fold = X_tr_proxy_full[val_idx_sorted]

    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_tr_fold, y_true[tr_idx],
              eval_set=[(X_val_fold, y_true[val_idx_sorted])],
              verbose=False)

    fold_pred = np.maximum(0, model.predict(X_val_fold))
    oof[val_idx_sorted] = fold_pred
    fold_mae = float(np.mean(np.abs(fold_pred - y_true[val_idx_sorted])))
    print(f"  Fold {fold_i+1}: MAE={fold_mae:.5f}  iter={model.best_iteration}  ({time.time()-t1:.0f}s)", flush=True)

    if fold_i == 0 and fold_mae > KILL_THRESH:
        print(f"  *** KILL: fold1={fold_mae:.4f} > {KILL_THRESH} ***")
        sys.exit(1)

    test_pred = np.maximum(0, model.predict(X_te_full))
    test_preds.append(test_pred)
    del model; gc.collect()

overall_mae = float(np.mean(np.abs(oof - y_true)))
test_avg = np.mean(test_preds, axis=0)
print(f"\nOverall OOF: {overall_mae:.5f}  ({time.time()-t0:.0f}s total)")

np.save(OUT_OOF,  oof)
np.save(OUT_TEST, test_avg)
print(f"Saved: {OUT_OOF}, {OUT_TEST}")

# 빠른 앙상블 평가
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id2_2 = [ls2[i] for i in train_raw['ID'].values]
fixed2_oof = (0.7636*d33['meta_avg_oof'][id2_2] + 0.1589*np.load('results/ranking/rank_adj_oof.npy')[id2_2]
             + 0.0119*np.load('results/iter_pseudo/round1_oof.npy')[id2_2]
             + 0.0346*np.load('results/iter_pseudo/round2_oof.npy')[id2_2]
             + 0.0310*np.load('results/iter_pseudo/round3_oof.npy')[id2_2])
xgb_o2 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o2 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o2 = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
oracle_new_oof2 = 0.64*fixed2_oof + 0.12*xgb_o2 + 0.16*lv2_o2 + 0.08*rem_o2
oracle_new_mae = float(np.mean(np.abs(oracle_new_oof2 - y_true)))

corr = float(np.corrcoef(oracle_new_oof2, oof)[0,1])
print(f"\noracle_NEW OOF: {oracle_new_mae:.5f}  layout_v3 OOF: {overall_mae:.5f}")
print(f"corr(oracle_NEW, layout_v3): {corr:.4f}")
print()
for w in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
    blend = (1-w)*oracle_new_oof2 + w*oof
    d = float(np.mean(np.abs(blend - y_true))) - oracle_new_mae
    print(f"  oracle_NEW*(1-{w})+layout_v3*{w}: delta={d:+.5f}")
print("\nDone.")
