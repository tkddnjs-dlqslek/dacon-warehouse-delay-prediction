"""
Oracle-Stack: 2-level stacker using oracle-xgb + oracle-lv2 OOF as features.
- Training: X = [v30_features, oracle_xgb_oof, oracle_lv2_oof, fixed_oof]
  * oracle OOFs are held-out predictions → no leakage
- Test: X = [v30_features, oracle_xgb_test, oracle_lv2_test, fixed_test]
- Model learns non-linear weights and feature-specific blending
Kill if fold1 > 9.0.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 9.0

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values

with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']
import gc
_tr = fe_tr['train_fe']; _tr_id2idx = {v:i for i,v in enumerate(_tr['ID'].values)}
_tr_ord = np.array([_tr_id2idx[i] for i in train_raw['ID'].values])
X_base_tr = _tr[feat_cols].values[_tr_ord].astype(np.float32)
del _tr, _tr_id2idx, _tr_ord, fe_tr; gc.collect()
_te = pd.DataFrame(fe_te); _te_id2idx = {v:i for i,v in enumerate(_te['ID'].values)}
_te_ord = np.array([_te_id2idx[i] for i in test_raw['ID'].values])
X_base_te = _te.reindex(columns=feat_cols, fill_value=0).values[_te_ord].astype(np.float32)
del _te, _te_id2idx, _te_ord, fe_te; gc.collect()

with open('results/mega33_final.pkl', 'rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

fixed_oof_ls = (fw['mega33']*d['meta_avg_oof']
               + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')
               + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')
               + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')
               + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy'))
fixed_oof = fixed_oof_ls[id_to_lspos]

fixed_test_ls = (fw['mega33']*d['meta_avg_test']
                + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')
                + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')
                + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')
                + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy'))
fixed_test = fixed_test_ls[test_id_to_lspos]

# Oracle OOF predictions (already in _row_id order) — load all available
ORACLE_FILES = {
    'xgb':       ('results/oracle_seq/oof_seqC_xgb.npy',       'results/oracle_seq/test_C_xgb.npy'),
    'lv2':       ('results/oracle_seq/oof_seqC_log_v2.npy',     'results/oracle_seq/test_C_log_v2.npy'),
    'remaining': ('results/oracle_seq/oof_seqC_xgb_remaining.npy','results/oracle_seq/test_C_xgb_remaining.npy'),
    'sc_only':   ('results/oracle_seq/oof_seqC_xgb_sc_only.npy','results/oracle_seq/test_C_xgb_sc_only.npy'),
    'residual':  ('results/oracle_seq/oof_seqC_xgb_residual.npy','results/oracle_seq/test_C_xgb_residual.npy'),
    'layout':    ('results/oracle_seq/oof_seqC_xgb_layout.npy', 'results/oracle_seq/test_C_xgb_layout.npy'),
    'layout_v2': ('results/oracle_seq/oof_seqC_xgb_layout_v2.npy','results/oracle_seq/test_C_xgb_layout_v2.npy'),
    'lgb_sc':    ('results/oracle_seq/oof_seqC_lgb_sc_only.npy','results/oracle_seq/test_C_lgb_sc_only.npy'),
    'cumulative':('results/oracle_seq/oof_seqC_xgb_cumulative.npy','results/oracle_seq/test_C_xgb_cumulative.npy'),
    'lgb_rem':   ('results/oracle_seq/oof_seqC_lgb_remaining.npy','results/oracle_seq/test_C_lgb_remaining.npy'),
    'latepos':   ('results/oracle_seq/oof_seqC_lgb_latepos.npy','results/oracle_seq/test_C_lgb_latepos.npy'),
    'xgb_lag1':  ('results/oracle_seq/oof_seqC_xgb_lag1.npy',  'results/oracle_seq/test_C_xgb_lag1.npy'),
    'lgb_rem_v3':('results/oracle_seq/oof_seqC_lgb_remaining_v3.npy','results/oracle_seq/test_C_lgb_remaining_v3.npy'),
    'dual':      ('results/oracle_seq/oof_seqC_lgb_dual.npy',   'results/oracle_seq/test_C_lgb_dual.npy'),
}
oof_arrays, test_arrays, names_avail = [], [], []
for nm, (op, tp) in ORACLE_FILES.items():
    if os.path.exists(op) and os.path.exists(tp):
        oa = np.load(op); ta = np.load(tp)
        mae = np.mean(np.abs(oa - y_true))
        print(f"  {nm:12s}: OOF={mae:.4f}")
        oof_arrays.append(oa); test_arrays.append(ta); names_avail.append(nm)
print(f"FIXED OOF MAE: {np.mean(np.abs(fixed_oof-y_true)):.4f}")
print(f"Stacking {len(names_avail)} oracle models: {names_avail}")

# Build feature matrix: v30 features + all oracle OOF preds + fixed pred
# X_base_tr and X_base_te already built as float32 above

row_sc_arr = train_raw['row_in_sc'].values

def make_X_tr(idx):
    base  = X_base_tr[idx]
    extra_parts = [oa[idx] for oa in oof_arrays] + [fixed_oof[idx], row_sc_arr[idx]]
    return np.hstack([base, np.column_stack(extra_parts)])

extra_te_parts = test_arrays + [fixed_test, test_raw['row_in_sc'].values]
X_te_stack = np.hstack([X_base_te, np.column_stack(extra_te_parts)])

LGB_PARAMS = dict(
    objective='mae', metric='mae', n_estimators=3000, learning_rate=0.05,
    max_depth=6, num_leaves=63, min_child_samples=30,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1, n_jobs=4, random_state=42, verbosity=-1
)

gkf = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values
oof = np.full(len(train_raw), np.nan)
test_list = []

print("Training oracle-Stack (LGB)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()
    X_tr_f = make_X_tr(tr_idx)
    X_val_f = make_X_tr(val_idx)
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr_f, y_true[tr_idx],
              eval_set=[(X_val_f, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)])

    val_idx_sorted = np.sort(val_idx)
    fold_pred = np.maximum(0, model.predict(make_X_tr(val_idx_sorted)))
    oof[val_idx_sorted] = fold_pred

    test_pred = np.maximum(0, model.predict(X_te_stack))
    test_list.append(test_pred)

    mae = np.mean(np.abs(fold_pred - y_true[val_idx_sorted]))
    print(f"Fold {fold_i+1}: oracle-Stack={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i == 0 and mae > KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True)
        sys.exit(1)

test_avg = np.mean(test_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_stack.npy', oof)
np.save('results/oracle_seq/test_C_lgb_stack.npy', test_avg)
print("Saved.", flush=True)

train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fixed2 = fixed_oof_ls[id2]
oof_mae=np.mean(np.abs(oof-y_true)); fixed_mae=np.mean(np.abs(fixed2-y_true))
print(f"\noracle-Stack OOF: {oof_mae:.4f}  FIXED: {fixed_mae:.4f}")
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
print(f"corr xgb_v30: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}  corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m=np.mean(np.abs(base5-y_true)); best_w=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"base5+Stack: w={best_w:.2f} MAE={best_m:.4f} delta={best_m-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
