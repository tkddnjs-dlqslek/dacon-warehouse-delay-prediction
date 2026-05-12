"""
Option B: 강한 정규화 oracle
- effective capacity 유지: lr=0.03 × n_est=3300 ≈ lr=0.05 × n_est=2000
- 더 얕은/단순한 트리: num_leaves 63→31, max_depth 8→6, min_child_samples 80→200
- 더 강한 정규화: reg_alpha/lambda 1.0→3.0
- 목적: 덜 layout-specific한 패턴 학습 → unseen layout 일반화
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time, gc, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

OUT_OOF  = 'results/oracle_seq/oof_seqC_robust_b.npy'
OUT_TEST = 'results/oracle_seq/test_C_robust_b.npy'

if os.path.exists(OUT_OOF) and os.path.exists(OUT_TEST):
    print(f'이미 존재: {OUT_OOF}'); import sys; sys.exit(0)

t0 = time.time()
print('='*60)
print('Option B: Stronger-Regularization Oracle')
print('  lr=0.03 x n_est=3300 (same capacity as 0.05x2000)')
print('  num_leaves=31, max_depth=6, min_child_samples=200')
print('  reg_alpha=3.0, reg_lambda=3.0')
print('='*60)

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
print(f'train: {len(train_raw)}, test: {len(test_raw)}')

# v30 features (oracle sequential features 없이 base만)
print('\n[1] v30 feature cache 로드...')
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

_tr = fe_tr['train_fe']
_tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord    = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_tr = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()

_te = pd.DataFrame(fe_te)
_te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord    = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_te = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()
print(f'  X_tr: {X_tr.shape}, X_te: {X_te.shape}')

# mega33 proxy for oracle sequential features
print('\n[2] mega33 proxy 로드...')
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

# oracle sequential features (proxy-consistent)
train_raw['mega_oof'] = mega_oof
test_raw['mega_test'] = mega_test
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

def add_oracle_feats(df, pred_col, is_train=True):
    grp = df.groupby(['layout_id','scenario_id'])
    df['lag1'] = grp[pred_col].shift(1).fillna(df[pred_col].mean())
    df['lag2'] = grp[pred_col].shift(2).fillna(df[pred_col].mean())
    df['sc_mean'] = grp[pred_col].transform('mean')
    return np.column_stack([
        df['lag1'].values, df['lag2'].values, df['sc_mean'].values,
        df['row_in_sc'].values.astype(np.float32)/25.0,
    ]).astype(np.float32)

of_tr = add_oracle_feats(train_raw, 'mega_oof', True)
of_te = add_oracle_feats(test_raw,  'mega_test', False)
X_tr_full = np.hstack([X_tr, of_tr])
X_te_full = np.hstack([X_te, of_te])
print(f'  X_tr_full: {X_tr_full.shape}')

PARAMS = dict(
    objective='huber', alpha=0.9,
    n_estimators=3300, learning_rate=0.03,
    num_leaves=31, max_depth=6,
    min_child_samples=200,
    subsample=0.6, colsample_bytree=0.6,
    reg_alpha=3.0, reg_lambda=3.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

print('\n[3] 5-fold GroupKFold 학습...')
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)
oof    = np.zeros(len(train_raw))
test_preds = []

for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t1 = time.time()
    val_idx_sorted = np.sort(val_idx)

    # val fold: lag features를 val 행의 mega_oof로 재계산
    val_df = train_raw.iloc[val_idx_sorted].copy()
    val_df['mega_oof_val'] = mega_oof[val_idx_sorted]
    of_val = add_oracle_feats(val_df, 'mega_oof_val', False)
    X_val = np.hstack([X_tr[val_idx_sorted], of_val])

    model = lgb.LGBMRegressor(**PARAMS)
    model.fit(X_tr_full[tr_idx], y_true[tr_idx],
              eval_set=[(X_val, y_true[val_idx_sorted])],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])

    fold_pred = np.clip(model.predict(X_val), 0, None)
    oof[val_idx_sorted] = fold_pred
    fold_mae = mean_absolute_error(y_true[val_idx_sorted], fold_pred)
    test_preds.append(np.clip(model.predict(X_te_full), 0, None))
    print(f'  Fold {fold_i+1}: MAE={fold_mae:.5f}  it={model.best_iteration_}  ({time.time()-t1:.0f}s)', flush=True)
    del model, val_df, X_val; gc.collect()

overall_mae = mean_absolute_error(y_true, oof)
test_avg = np.mean(test_preds, axis=0)
print(f'\nOverall OOF: {overall_mae:.5f}  ({time.time()-t0:.0f}s total)')

np.save(OUT_OOF,  oof)
np.save(OUT_TEST, test_avg)
print(f'Saved: {OUT_OOF}, {OUT_TEST}')

# oracle_NEW와 비교
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
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
mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
corr = float(np.corrcoef(oracle_new_oof2, oof)[0,1])
print(f'\noracle_NEW OOF: {mae(oracle_new_oof2):.5f}')
print(f'robust_b  OOF: {overall_mae:.5f}')
print(f'corr: {corr:.4f}')
for w in [0.05, 0.10, 0.15, 0.20]:
    b = (1-w)*oracle_new_oof2 + w*oof
    d_val = mae(b) - mae(oracle_new_oof2)
    print(f'  blend w={w}: delta={d_val:+.5f}')
print('Done.')
