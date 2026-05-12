"""
Scheduled Sampling: mix true lag (y_{t-1}) and mega33 lag during training.
- Training row: 50% chance true lag, 50% chance mega33_oof lag (randomized)
- OOF: sequential (model's own predictions as lag) → reduces exposure bias gap
- Test: sequential (same as OOF)

Hypothesis: residual_corr drops from 0.98 (pseudo) toward 0.64 (oracle range).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

RNG = np.random.default_rng(42)
MIX_PROB = 0.5   # probability of using mega33 lag (vs true lag) per row

print("Loading data...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')

train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
global_mean = y_true.mean()

# ── v30 FE ──
print("Loading FE...", flush=True)
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

# ── mega33 OOF/Test (ID 순서) ──
print("Loading mega33...", flush=True)
with open('results/mega33_final.pkl','rb') as f:
    d = pickle.load(f)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id_to_lspos = [ls_to_pos[i] for i in train_raw['ID'].values]
mega_oof_id = d['meta_avg_oof'][id_to_lspos]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos = {row['ID']:i for i,row in test_ls.iterrows()}
test_id_to_lspos = [te_ls_to_pos[i] for i in test_raw['ID'].values]
mega_test_id = d['meta_avg_test'][test_id_to_lspos]

# ── True lag (y_{t-1}) & mega33 lag ──
train_raw['mega33_pred'] = mega_oof_id
train_raw['lag1_true']   = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(1).fillna(global_mean))
train_raw['lag2_true']   = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(2).fillna(global_mean))
train_raw['lag1_mega']   = (train_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(1).fillna(global_mean))
train_raw['lag2_mega']   = (train_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(2).fillna(global_mean))

test_raw['mega33_pred'] = mega_test_id
test_raw['lag1_mega']   = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(1).fillna(global_mean))
test_raw['lag2_mega']   = (test_raw
    .groupby(['layout_id','scenario_id'])['mega33_pred']
    .shift(2).fillna(global_mean))

LAG_COLS   = ['lag1', 'lag2', 'row_in_sc']
X_ALL_COLS = feat_cols + LAG_COLS

X_train_base = fe_train_df[feat_cols].values
lag1_true_arr = train_raw['lag1_true'].values
lag2_true_arr = train_raw['lag2_true'].values
lag1_mega_arr = train_raw['lag1_mega'].values
lag2_mega_arr = train_raw['lag2_mega'].values
row_sc_arr    = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base  = X_test_base[feat_cols].values
test_row_sc  = test_raw['row_in_sc'].values

def make_X(base_arr, lag1, lag2, row_sc):
    extra = np.column_stack([lag1, lag2, row_sc])
    return np.hstack([base_arr, extra])

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_preds     = np.full(len(train_raw), np.nan)
test_pred_list = []

print("Training (scheduled sampling)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    # ── Mixed lag for training ──
    use_mega = RNG.random(len(tr_idx)) < MIX_PROB
    lag1_tr  = np.where(use_mega, lag1_mega_arr[tr_idx], lag1_true_arr[tr_idx])
    lag2_tr  = np.where(use_mega, lag2_mega_arr[tr_idx], lag2_true_arr[tr_idx])

    X_tr = make_X(X_train_base[tr_idx], lag1_tr, lag2_tr, row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]

    # Dummy val for early stopping (using mega33 lag — approximates test dist)
    X_val_dummy = make_X(X_train_base[val_idx],
                         lag1_mega_arr[val_idx],
                         lag2_mega_arr[val_idx],
                         row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_dummy, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(500)])

    # ── Sequential OOF (same structure as train_lag_target.py) ──
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values

    fold_oof = np.zeros(len(val_sorted))

    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]

        if pos == 0:
            lag1_fill = np.full(pos_mask.sum(), global_mean)
            lag2_fill = np.full(pos_mask.sum(), global_mean)
        else:
            lag1_fill = fold_oof[row_in_sc_vals == (pos - 1)]
            lag2_fill = (fold_oof[row_in_sc_vals == (pos - 2)]
                         if pos >= 2 else np.full(pos_mask.sum(), global_mean))

        X_pos = make_X(X_train_base[pos_idx], lag1_fill, lag2_fill,
                       np.full(pos_mask.sum(), pos))
        fold_oof[pos_mask] = np.maximum(0, model.predict(X_pos))

    oof_preds[val_sorted] = fold_oof
    fold_mae = np.mean(np.abs(fold_oof - y_true[val_sorted]))
    elapsed  = time.time() - t0
    print(f"Fold {fold_i+1}: OOF_MAE={fold_mae:.4f} ({elapsed:.0f}s)", flush=True)

    # ── Sequential Test ──
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values

    test_pred_sorted = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]

        if pos == 0:
            l1 = np.full(pos_mask.sum(), float(y_tr.mean()))
            l2 = l1.copy()
        else:
            l1 = test_pred_sorted[test_sorted[test_rsc_vals == (pos-1)]]
            l2 = (test_pred_sorted[test_sorted[test_rsc_vals == (pos-2)]]
                  if pos >= 2 else np.full(pos_mask.sum(), float(y_tr.mean())))

        X_pos = make_X(X_test_base[pos_idx], l1, l2, np.full(pos_mask.sum(), pos))
        test_pred_sorted[pos_idx] = np.maximum(0, model.predict(X_pos))

    test_pred_list.append(test_pred_sorted)

# ── Results ──
oof_mae   = np.mean(np.abs(oof_preds - y_true))
mega_mae  = np.mean(np.abs(mega_oof_id - y_true))
res_corr  = np.corrcoef(oof_preds - y_true, mega_oof_id - y_true)[0,1]

print(f"\n=== RESULTS ===", flush=True)
print(f"mega33 OOF MAE        = {mega_mae:.4f}", flush=True)
print(f"sched_sample OOF MAE  = {oof_mae:.4f}  (delta={oof_mae-mega_mae:.4f})", flush=True)
print(f"residual_corr (sched, mega33) = {res_corr:.4f}", flush=True)

best_mae, best_w = 9999, 0
for w in np.arange(0, 1.01, 0.02):
    m = np.mean(np.abs(w*oof_preds + (1-w)*mega_oof_id - y_true))
    if m < best_mae:
        best_mae, best_w = m, w
print(f"Best blend w={best_w:.2f}: MAE={best_mae:.4f}  delta={best_mae-mega_mae:.4f}", flush=True)

# threshold check
threshold = mega_mae + 1.25 * (1 - res_corr)
print(f"Blend threshold: {threshold:.4f}  → {'✓ MEETS' if oof_mae < threshold else '✗ FAILS'}", flush=True)

os.makedirs('results/sched_sample', exist_ok=True)
np.save('results/sched_sample/sched_oof.npy', oof_preds)
test_avg = np.mean(test_pred_list, axis=0)
np.save('results/sched_sample/sched_test.npy', test_avg)

with open('results/sched_sample/summary.json','w',encoding='utf-8') as f:
    json.dump({
        'mega33_mae': float(mega_mae),
        'sched_mae':  float(oof_mae),
        'delta':      float(oof_mae - mega_mae),
        'residual_corr': float(res_corr),
        'best_blend_w': float(best_w),
        'best_blend_mae': float(best_mae),
        'blend_delta': float(best_mae - mega_mae),
        'mix_prob': MIX_PROB
    }, f, indent=2)
print("Saved to results/sched_sample/", flush=True)
