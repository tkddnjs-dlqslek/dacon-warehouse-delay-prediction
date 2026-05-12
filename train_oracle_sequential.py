"""
Oracle Sequential with Mega33 Seed.

Key hypothesis: oracle model (trained with true lags) + mega33_oof as seed for
position-1 (instead of model's own pos-0 prediction) reduces exposure bias.

Training: same oracle (lag = true y_{t-1})
OOF sequential:
  - pos 0: model's own prediction (lag = global_mean) ← same as before
  - pos 1: lag = mega33_oof[pos_0] ← DIFFERENT: use better-quality seed
  - pos 2-24: lag = model's sequential predictions

Rationale:
  - mega33_oof[pos_0] MAE ~ 8.4  (oracle model's own pos-0 pred MAE ~ 9+)
  - Better seed at pos 1 → better chain quality downstream
  - Model was trained with realistic y values as lag → handles mega33 predictions well
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, json, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

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

print("Loading FE...", flush=True)
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

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

# ── Oracle lag features (true y_{t-1}) ──
train_raw['lag1_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(1).fillna(global_mean))
train_raw['lag2_y'] = (train_raw
    .groupby(['layout_id','scenario_id'])['avg_delay_minutes_next_30m']
    .shift(2).fillna(global_mean))

LAG_COLS   = ['lag1_y', 'lag2_y', 'row_in_sc']
X_ALL_COLS = feat_cols + LAG_COLS

X_train_base = fe_train_df[feat_cols].values
train_lag1   = train_raw['lag1_y'].values
train_lag2   = train_raw['lag2_y'].values
row_sc_arr   = train_raw['row_in_sc'].values
mega_oof_arr = mega_oof_id   # used as seed for position 1

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base   = X_test_base[feat_cols].values
test_row_sc   = test_raw['row_in_sc'].values

def make_X(base, lag1, lag2, row_sc):
    return np.hstack([base, np.column_stack([lag1, lag2, row_sc])])

LGB_PARAMS = dict(
    objective='mae', n_estimators=3000, learning_rate=0.05,
    num_leaves=128, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=4, random_state=42, verbose=-1
)

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

# We'll run 3 variants:
# A: standard sequential (global_mean seed at pos 0, model's own pred as lag)
# B: mega33-seeded (mega33_oof as lag for positions 1+, not model's own pos-0 pred)
# C: pseudo-lag (mega33_oof as lag for ALL positions — sanity check)

oof_seqA   = np.full(len(train_raw), np.nan)   # Standard sequential
oof_seqB   = np.full(len(train_raw), np.nan)   # Mega33-seeded at pos 1
oof_seqC   = np.full(len(train_raw), np.nan)   # Pure mega33 lag (pseudo-lag)
test_predsA = []
test_predsB = []

print("Training oracle model (variants A, B, C)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    # Training with TRUE lags
    X_tr = make_X(X_train_base[tr_idx], train_lag1[tr_idx],
                  train_lag2[tr_idx], row_sc_arr[tr_idx])
    y_tr = y_true[tr_idx]

    X_val_true = make_X(X_train_base[val_idx], train_lag1[val_idx],
                        train_lag2[val_idx], row_sc_arr[val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_val_true, y_true[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False),
                         lgb.log_evaluation(9999)])

    # Setup sorted val for sequential eval
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values

    # ── Variant A: Standard sequential (global_mean at pos 0) ──
    foldA = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = foldA[row_in_sc_vals == (pos-1)]
            l2 = foldA[row_in_sc_vals == (pos-2)] if pos >= 2 else np.full(n_pos, global_mean)
        X_pos = make_X(X_train_base[pos_idx], l1, l2, np.full(n_pos, pos))
        foldA[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_seqA[val_sorted] = foldA

    # ── Variant B: Mega33-seeded at pos 1 ──
    foldB = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        elif pos == 1:
            # KEY DIFFERENCE: use mega33_oof[pos_0] as lag, not foldA[pos_0]
            l1 = mega_oof_arr[val_sorted[row_in_sc_vals == 0]]  # mega33's pos-0 preds
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = foldB[row_in_sc_vals == (pos-1)]
            l2 = (foldB[row_in_sc_vals == (pos-2)] if pos >= 2
                  else mega_oof_arr[val_sorted[row_in_sc_vals == 0]])
        X_pos = make_X(X_train_base[pos_idx], l1, l2, np.full(n_pos, pos))
        foldB[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_seqB[val_sorted] = foldB

    # ── Variant C: Pure mega33 lag (pseudo-lag sanity check) ──
    mega_lag1 = mega_oof_arr[val_sorted]
    # Build shifted arrays for val set
    foldC = np.zeros(len(val_sorted))
    # mega33 lags: shifted by 1 within scenario
    mega_val_sorted = mega_oof_arr[val_sorted]
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        pos_idx  = val_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1 = np.full(n_pos, global_mean)
            l2 = np.full(n_pos, global_mean)
        else:
            l1 = mega_val_sorted[row_in_sc_vals == (pos-1)]
            l2 = (mega_val_sorted[row_in_sc_vals == (pos-2)] if pos >= 2
                  else np.full(n_pos, global_mean))
        X_pos = make_X(X_train_base[pos_idx], l1, l2, np.full(n_pos, pos))
        foldC[pos_mask] = np.maximum(0, model.predict(X_pos))
    oof_seqC[val_sorted] = foldC

    maeA = np.mean(np.abs(foldA - y_true[val_sorted]))
    maeB = np.mean(np.abs(foldB - y_true[val_sorted]))
    maeC = np.mean(np.abs(foldC - y_true[val_sorted]))
    elapsed = time.time() - t0
    print(f"Fold {fold_i+1}: A={maeA:.4f}  B(mega-seeded)={maeB:.4f}  C(pseudo)={maeC:.4f}  ({elapsed:.0f}s)", flush=True)

    # ── Test variant A ──
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values

    testA = np.zeros(len(test_raw))
    testB = np.zeros(len(test_raw))
    mega_test_sorted = mega_test_id[test_sorted]

    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        pos_idx  = test_sorted[pos_mask]
        n_pos    = pos_mask.sum()
        if pos == 0:
            l1A = np.full(n_pos, float(y_tr.mean()))
            l2A = l1A.copy()
            l1B = l1A.copy()
            l2B = l2A.copy()
        elif pos == 1:
            l1A = testA[test_sorted[test_rsc_vals == 0]]
            l2A = np.full(n_pos, float(y_tr.mean()))
            l1B = mega_test_sorted[test_rsc_vals == 0]  # mega33 as seed for B
            l2B = np.full(n_pos, float(y_tr.mean()))
        else:
            l1A = testA[test_sorted[test_rsc_vals == (pos-1)]]
            l1B = testB[test_sorted[test_rsc_vals == (pos-1)]]
            if pos >= 2:
                l2A = testA[test_sorted[test_rsc_vals == (pos-2)]]
                l2B = testB[test_sorted[test_rsc_vals == (pos-2)]]
            else:
                l2A = np.full(n_pos, float(y_tr.mean()))
                l2B = mega_test_sorted[test_rsc_vals == 0]

        testA[pos_idx] = np.maximum(0, model.predict(make_X(X_test_base[pos_idx], l1A, l2A, np.full(n_pos, pos))))
        testB[pos_idx] = np.maximum(0, model.predict(make_X(X_test_base[pos_idx], l1B, l2B, np.full(n_pos, pos))))

    test_predsA.append(testA)
    test_predsB.append(testB)

# ── Summary ──
mega_mae = np.mean(np.abs(mega_oof_id - y_true))

for tag, oof_x in [('A_standard', oof_seqA), ('B_mega_seed', oof_seqB), ('C_pseudo', oof_seqC)]:
    mae  = np.mean(np.abs(oof_x - y_true))
    corr = np.corrcoef(oof_x - y_true, mega_oof_id - y_true)[0,1]
    thr  = mega_mae + 1.25*(1-corr)
    best_mae, best_w = 9999, 0
    for w in np.arange(0, 1.01, 0.02):
        m = np.mean(np.abs(w*oof_x + (1-w)*mega_oof_id - y_true))
        if m < best_mae:
            best_mae, best_w = m, w
    print(f"\n[{tag}] MAE={mae:.4f}  corr={corr:.4f}  threshold={thr:.4f}  "
          f"{'✓' if mae < thr else '✗'}  blend_w={best_w:.2f} delta={best_mae-mega_mae:.4f}", flush=True)

os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqA.npy', oof_seqA)
np.save('results/oracle_seq/oof_seqB.npy', oof_seqB)
np.save('results/oracle_seq/oof_seqC.npy', oof_seqC)
test_avgA = np.mean(test_predsA, axis=0)
test_avgB = np.mean(test_predsB, axis=0)
np.save('results/oracle_seq/test_A.npy', test_avgA)
np.save('results/oracle_seq/test_B.npy', test_avgB)

with open('results/oracle_seq/summary.json','w',encoding='utf-8') as f:
    results = {}
    for tag, oof_x, tp in [('A', oof_seqA, test_avgA), ('B', oof_seqB, test_avgB), ('C', oof_seqC, None)]:
        mae  = float(np.mean(np.abs(oof_x - y_true)))
        corr = float(np.corrcoef(oof_x - y_true, mega_oof_id - y_true)[0,1])
        thr  = float(mega_mae + 1.25*(1-corr))
        results[tag] = {'mae': mae, 'corr': corr, 'threshold': thr, 'meets': mae < thr}
    json.dump({'mega33_mae': float(mega_mae), 'variants': results}, f, indent=2)
print("\nSaved to results/oracle_seq/", flush=True)
