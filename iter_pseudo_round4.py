"""
Iter-Pseudo Round 4: uses best oracle blend as pseudo label (more accurate than mega33).
Pseudo = 0.64*FIXED + 0.12*xgb + 0.16*lv2 + 0.08*remaining (current best test predictions).
Filter: rows where best_blend and mega33 agree (std of [mega33, r1, r2, r3, best_blend] <= 70th pct).
Saves: results/iter_pseudo/round4_oof.npy, round4_test.npy
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np, pandas as pd, time, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

OUT = "results/iter_pseudo"
TARGET = "avg_delay_minutes_next_30m"

print("Loading...", flush=True)
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)
y_log = np.log1p(y)
X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
del blob, train_fe, test_fe; gc.collect()

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof  = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])

# FIXED blend (ls order)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof_ls = (fw['mega33']*mega['meta_avg_oof']
              + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')
              + fw['iter_r1']*np.load(f'{OUT}/round1_oof.npy')
              + fw['iter_r2']*np.load(f'{OUT}/round2_oof.npy')
              + fw['iter_r3']*np.load(f'{OUT}/round3_oof.npy'))
fixed_test_ls = (fw['mega33']*mega['meta_avg_test']
               + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')
               + fw['iter_r1']*np.load(f'{OUT}/round1_test.npy')
               + fw['iter_r2']*np.load(f'{OUT}/round2_test.npy')
               + fw['iter_r3']*np.load(f'{OUT}/round3_test.npy'))

# oracle predictions (ls order = same as iter_pseudo fold_ids order)
xgb_oof_ls  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_oof_ls  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_oof_ls  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_test_ls = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_test_ls = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_test_ls = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

# NOTE: oracle arrays are in _row_id order; iter_pseudo uses layout_scenario order (ls order).
# We need to remap oracle OOF from _row_id order to ls order.
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw_rid = train_raw.sort_values('_row_id').reset_index(drop=True)
rid_to_ls = {row['ID']: i for i, row in train_ls.iterrows()}
ls_to_rid = [rid_to_ls.get(iid, i) for i, iid in enumerate(train_raw_rid['ID'].values)]

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw_rid = test_raw.sort_values('_row_id').reset_index(drop=True)
te_rid_to_ls = {row['ID']: i for i, row in test_ls.iterrows()}
te_ls_to_rid = [te_rid_to_ls.get(iid, i) for i, iid in enumerate(test_raw_rid['ID'].values)]

# Build proxy in ls order
# oracle OOF is in _row_id order → need to map to ls order
# Actually: oracle OOF[i] corresponds to train_raw_rid.iloc[i]
# ls order: train_ls.iloc[j]
# We need: for each ls index j, find _row_id index of same ID
ls_id_to_rid_idx = {row['ID']: i for i, row in train_raw_rid.iterrows()}
ls_order_to_rid = np.array([ls_id_to_rid_idx[row['ID']] for _, row in train_ls.iterrows()])

te_ls_id_to_rid_idx = {row['ID']: i for i, row in test_raw_rid.iterrows()}
te_ls_order_to_rid = np.array([te_ls_id_to_rid_idx[row['ID']] for _, row in test_ls.iterrows()])

# Best blend in ls order
best_blend_oof_ls  = (0.64*fixed_oof_ls  + 0.12*xgb_oof_ls[ls_order_to_rid]
                    + 0.16*lv2_oof_ls[ls_order_to_rid]  + 0.08*rem_oof_ls[ls_order_to_rid])
best_blend_test_ls = (0.64*fixed_test_ls + 0.12*xgb_test_ls[te_ls_order_to_rid]
                    + 0.16*lv2_test_ls[te_ls_order_to_rid] + 0.08*rem_test_ls[te_ls_order_to_rid])

baseline_mae = mean_absolute_error(y, mega33_oof)
blend_mae    = mean_absolute_error(y, best_blend_oof_ls)
print(f"mega33 baseline: {baseline_mae:.5f}")
print(f"best_blend OOF:  {blend_mae:.5f}  (proxy quality check)", flush=True)

# Previous round predictions for filter
test_r1 = np.load(f'{OUT}/round1_test.npy')
test_r2 = np.load(f'{OUT}/round2_test.npy')
test_r3 = np.load(f'{OUT}/round3_test.npy')

# Stability filter: std of [mega33, r1, r2, r3, best_blend] <= 70th percentile
preds_stack = np.stack([mega33_test, test_r1, test_r2, test_r3, best_blend_test_ls], axis=1)
std_preds   = preds_stack.std(axis=1)
thresh4     = np.percentile(std_preds, 70)
stable_mask = std_preds <= thresh4
print(f"\nRound 4 stability filter: std <= {thresh4:.3f}")
print(f"  rows passing: {stable_mask.sum()} ({100*stable_mask.mean():.1f}%)", flush=True)

# Pseudo = best blend test (more accurate than mega33)
pseudo_r4 = best_blend_test_ls

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

print("\n=== Round 4 (best-blend pseudo) ===", flush=True)
X_te_p    = X_te[stable_mask]
pseudo_y  = np.log1p(np.clip(pseudo_r4[stable_mask], 0, None)).astype(np.float32)

oof  = np.zeros(len(y))
test_pred = np.zeros(len(X_te))
t_round = time.time()

for f in range(5):
    tf = time.time()
    tr  = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]

    X_comb = np.vstack([X_tr[tr], X_te_p])
    y_comb = np.concatenate([y_log[tr], pseudo_y])
    w_comb = np.concatenate([np.ones(len(tr)), np.ones(len(X_te_p))])

    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_comb, y_comb, sample_weight=w_comb,
          eval_set=[(X_tr[val], y_log[val])],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])

    oof[val]   = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
    test_pred += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

    fm = mean_absolute_error(y[val], oof[val])
    print(f"  fold {f}: mae={fm:.5f} it={m.best_iteration_} ({time.time()-tf:.0f}s)", flush=True)
    del X_comb, y_comb, w_comb, m; gc.collect()

oof_mae = mean_absolute_error(y, oof)
corr_mega = float(np.corrcoef(y - mega33_oof, y - oof)[0, 1])

# Blend search: add round4 to current FIXED
best_w, best_mae_r4 = 0, blend_mae
for w in np.arange(0.01, 0.21, 0.01):
    mm = mean_absolute_error(y, (1-w)*best_blend_oof_ls + w*oof)
    if mm < best_mae_r4: best_w, best_mae_r4 = w, mm

np.save(f'{OUT}/round4_oof.npy', oof)
np.save(f'{OUT}/round4_test.npy', test_pred)

print(f"\nRound 4 OOF: {oof_mae:.5f}")
print(f"corr residual vs mega33: {corr_mega:.4f}")
print(f"best4+round4: w={best_w:.2f} MAE={best_mae_r4:.5f} delta={best_mae_r4-blend_mae:+.5f}")
print(f"elapsed: {time.time()-t_round:.0f}s")
print("Done.")
