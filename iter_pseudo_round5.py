"""
Iter-Pseudo Round 5: uses oracle_NEW test predictions as pseudo-labels.
oracle_NEW is the best submission (LB 9.7527), so its test predictions are the
most accurate proxy available — especially for unseen layouts (mean=22.716).
Filter: std of [r1, r2, r3, r4, oracle_new] <= 70th percentile (row-level agreement).
Saves: results/iter_pseudo/round5_oof.npy, round5_test.npy
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np, pandas as pd, time, warnings, gc, os
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb

os.chdir("C:/Users/user/Desktop/데이콘 4월")
OUT = "results/iter_pseudo"
TARGET = "avg_delay_minutes_next_30m"

print("Loading features...", flush=True)
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

# Order mapping: ls order ↔ _row_id order
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw_rid = train_raw.sort_values('_row_id').reset_index(drop=True)
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw_rid = test_raw.sort_values('_row_id').reset_index(drop=True)
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)

# Map: ls index → _row_id index (for oracle arrays which are in _row_id order)
te_rid_idx = {row['ID']: i for i, row in test_raw_rid.iterrows()}
te_ls_order_to_rid = np.array([te_rid_idx[row['ID']] for _, row in test_ls.iterrows()])

# Load oracle_NEW test predictions (in _row_id order from submission CSV)
oracle_sub = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
id_order_rid = test_raw_rid['ID'].values
oracle_test_rid = oracle_sub.set_index('ID').reindex(id_order_rid)['avg_delay_minutes_next_30m'].values
oracle_test_ls = oracle_test_rid[te_ls_order_to_rid]  # convert to ls order

print(f"oracle_NEW test: mean={oracle_test_ls.mean():.3f}")
# Quick unseen check
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask_rid = ~test_raw_rid['layout_id'].isin(train_layouts).values
unseen_mask_ls = unseen_mask_rid[te_ls_order_to_rid]
print(f"  seen mean={oracle_test_ls[~unseen_mask_ls].mean():.3f}  unseen mean={oracle_test_ls[unseen_mask_ls].mean():.3f}")

# Previous rounds for stability filter (all in ls order already)
test_r1 = np.load(f'{OUT}/round1_test.npy')
test_r2 = np.load(f'{OUT}/round2_test.npy')
test_r3 = np.load(f'{OUT}/round3_test.npy')
test_r4 = np.load(f'{OUT}/round4_test.npy')

# Stability filter: std of [r1, r2, r3, r4, oracle_new] <= 70th percentile
preds_stack = np.stack([test_r1, test_r2, test_r3, test_r4, oracle_test_ls], axis=1)
std_preds = preds_stack.std(axis=1)
thresh5 = np.percentile(std_preds, 70)
stable_mask = std_preds <= thresh5
print(f"\nRound 5 stability filter: std <= {thresh5:.3f}")
print(f"  rows passing: {stable_mask.sum()} ({100*stable_mask.mean():.1f}%)", flush=True)
print(f"  unseen passing: {(stable_mask & unseen_mask_ls).sum()} / {unseen_mask_ls.sum()}", flush=True)

# Pseudo-label = oracle_NEW (best available test prediction)
pseudo_r5 = oracle_test_ls

PARAMS = dict(
    objective="huber", n_estimators=5000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

print("\n=== Round 5 (oracle_NEW pseudo) ===", flush=True)
X_te_p = X_te[stable_mask]
pseudo_y = np.log1p(np.clip(pseudo_r5[stable_mask], 0, None)).astype(np.float32)

oof = np.zeros(len(y))
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
print(f"\nRound 5 OOF MAE: {oof_mae:.5f}")
print(f"  test_unseen mean: {test_pred[unseen_mask_ls].mean():.3f}")
print(f"  test_seen mean: {test_pred[~unseen_mask_ls].mean():.3f}")

# Need oracle_NEW OOF for blend analysis
# Load oracle OOF in ls order (same as iter_pseudo internal order)
ls_rid_idx = {row['ID']: i for i, row in train_raw_rid.iterrows()}
ls_order_to_rid = np.array([ls_rid_idx[row['ID']] for _, row in train_ls.iterrows()])

with open("results/mega33_final.pkl", "rb") as f: mega = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof_ls = (fw['mega33']*mega['meta_avg_oof']
              + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')
              + fw['iter_r1']*np.load(f'{OUT}/round1_oof.npy')
              + fw['iter_r2']*np.load(f'{OUT}/round2_oof.npy')
              + fw['iter_r3']*np.load(f'{OUT}/round3_oof.npy'))
blend_mae = mean_absolute_error(y, fixed_oof_ls)
print(f"\nBlend analysis (oracle_NEW context: OOF~8.3762 in _row_id order)")
print(f"fixed_oof_ls baseline: {blend_mae:.5f}")

# Blend sweep: oracle_NEW-equivalent in ls order vs round5
best_w5, best_mae5 = 0, blend_mae
for w in np.arange(0.01, 0.21, 0.01):
    mm = mean_absolute_error(y, (1-w)*fixed_oof_ls + w*oof)
    if mm < best_mae5: best_w5, best_mae5 = w, mm
    if w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
        print(f"  fixed(1-{w:.2f})+r5({w:.2f}): {mm:.5f}  delta={mm-blend_mae:+.5f}")

print(f"\nbest_w={best_w5:.2f}  best_mae={best_mae5:.5f}  delta={best_mae5-blend_mae:+.5f}")

np.save(f'{OUT}/round5_oof.npy', oof)
np.save(f'{OUT}/round5_test.npy', test_pred)
print("Saved round5_oof.npy, round5_test.npy")
print(f"Total elapsed: {time.time()-t_round:.0f}s")
print("Done.")
