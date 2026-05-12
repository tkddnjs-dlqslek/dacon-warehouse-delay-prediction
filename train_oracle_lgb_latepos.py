"""
Oracle-LGB-latepos: Train ONLY on positions 17-24 (8 late positions where oracle is most beneficial).
At late positions, 3 log lags are fully available (positions 14-16 for pos 17, up to 21-23 for pos 24).
A model specialized for late positions learns tighter lag→y mapping calibrated to late-position dynamics.
At OOF eval: sequential prediction only for positions 15-24 (enough lag history for pos 17-24).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

LATE_MIN = 17   # positions 17-24 are where oracle helps most
NLAGS    = 3    # 3 log lags (confirmed working)

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
y_log  = np.log1p(y_true)
global_mean_log = y_log.mean()
global_mean     = y_true.mean()

with open('results/eda_v30/v30_fe_cache.pkl','rb') as f:
    fe_tr = pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f:
    fe_te = pickle.load(f)
feat_cols = fe_tr['feat_cols']

fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

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

# Log lags of true y for training
train_raw['log_y'] = y_log
grp = train_raw.groupby(['layout_id','scenario_id'])['log_y']
for k in range(1, NLAGS+1):
    train_raw[f'lag{k}_logy'] = grp.shift(k).fillna(global_mean_log)

# Log lags of mega33 for test
test_raw['log_mega33'] = np.log1p(mega_test_id)
m_grp = test_raw.groupby(['layout_id','scenario_id'])['log_mega33']
for k in range(1, NLAGS+1):
    test_raw[f'lag{k}_logmega'] = m_grp.shift(k).fillna(global_mean_log)

X_train_base = fe_train_df[feat_cols].fillna(0).values
train_lags   = [train_raw[f'lag{k}_logy'].values for k in range(1, NLAGS+1)]
row_sc_arr   = train_raw['row_in_sc'].values

X_test_base = fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in X_test_base.columns:
        X_test_base[c] = 0.0
X_test_base = X_test_base[feat_cols].fillna(0).values
test_lags   = [test_raw[f'lag{k}_logmega'].values for k in range(1, NLAGS+1)]
test_row_sc = test_raw['row_in_sc'].values

def make_X(base, lags_list, row_sc):
    return np.hstack([base, np.column_stack(lags_list + [row_sc])])

LGB_PARAMS = dict(
    objective='mae', n_estimators=4000, learning_rate=0.04,
    num_leaves=256, max_depth=-1,
    min_child_samples=20, min_child_weight=0.001,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    reg_alpha=0.1, reg_lambda=0.1,
    n_jobs=6, random_state=42, verbose=-1,
)
EARLY = 100

# Mask: only late positions for training
late_mask_train = train_raw['row_in_sc'].values >= LATE_MIN

gkf    = GroupKFold(n_splits=5)
groups = train_raw['layout_id'].values

oof_lp   = np.full(len(train_raw), np.nan)
test_lp_list = []

print(f"Training oracle-LGB-latepos (pos>={LATE_MIN})...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    t0 = time.time()

    # Training: only late positions from train split
    late_tr_mask = np.zeros(len(train_raw), dtype=bool)
    late_tr_mask[tr_idx] = True
    late_tr_mask &= late_mask_train
    late_tr_idx = np.where(late_tr_mask)[0]

    X_tr   = make_X(X_train_base[late_tr_idx],
                    [train_lags[k][late_tr_idx] for k in range(NLAGS)],
                    row_sc_arr[late_tr_idx])
    y_tr   = y_log[late_tr_idx]

    # Early stopping with late val rows (oracle condition - true y lags)
    late_val_mask = np.zeros(len(train_raw), dtype=bool)
    late_val_mask[val_idx] = True
    late_val_mask &= late_mask_train
    late_val_idx = np.where(late_val_mask)[0]

    X_vtrue = make_X(X_train_base[late_val_idx],
                     [train_lags[k][late_val_idx] for k in range(NLAGS)],
                     row_sc_arr[late_val_idx])

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr, y_tr,
              eval_set=[(X_vtrue, y_log[late_val_idx])],
              callbacks=[lgb.early_stopping(EARLY, verbose=False), lgb.log_evaluation(-1)])

    # OOF sequential eval: scan all positions 0-24 but record predictions only for late positions
    val_df_tmp = train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df_tmp['_orig'] = val_idx
    val_df_tmp = val_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted     = val_df_tmp['_orig'].values
    row_in_sc_vals = val_df_tmp['row_in_sc'].values
    mega_val_sorted_log = np.log1p(mega_oof_id[val_sorted])

    fold_lp = np.zeros(len(val_sorted))
    for pos in range(25):
        pos_mask = row_in_sc_vals == pos
        if pos < LATE_MIN:
            continue  # skip early positions
        pos_idx = val_sorted[pos_mask]
        n_pos   = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_val_sorted_log[row_in_sc_vals == (pos-k)])
        X_pos = make_X(X_train_base[pos_idx], lags_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        fold_lp[pos_mask] = np.maximum(0, np.expm1(pred_log))
    oof_lp[val_sorted] = fold_lp

    # Test inference: only late positions
    test_df_tmp = test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    test_df_tmp['_orig'] = np.arange(len(test_raw))
    test_df_tmp = test_df_tmp.sort_values(['layout_id','scenario_id','row_in_sc'])
    test_sorted   = test_df_tmp['_orig'].values
    test_rsc_vals = test_df_tmp['row_in_sc'].values
    mega_test_sorted_log = np.log1p(mega_test_id[test_sorted])

    test_lp = np.zeros(len(test_raw))
    for pos in range(25):
        pos_mask = test_rsc_vals == pos
        if pos < LATE_MIN:
            continue
        pos_idx = test_sorted[pos_mask]
        n_pos   = pos_mask.sum()
        lags_pos = []
        for k in range(1, NLAGS+1):
            if pos - k < 0:
                lags_pos.append(np.full(n_pos, global_mean_log))
            else:
                lags_pos.append(mega_test_sorted_log[test_rsc_vals == (pos-k)])
        X_pos = make_X(X_test_base[pos_idx], lags_pos, np.full(n_pos, pos))
        pred_log = model.predict(X_pos)
        test_lp[pos_idx] = np.maximum(0, np.expm1(pred_log))
    test_lp_list.append(test_lp)

    # Compute MAE only on late positions in val
    late_mask_val = row_in_sc_vals >= LATE_MIN
    late_val_sorted = val_sorted[late_mask_val]
    mae_late = np.mean(np.abs(fold_lp[late_mask_val] - y_true[late_val_sorted]))
    elapsed = time.time() - t0
    best_it = model.best_iteration_ if hasattr(model, 'best_iteration_') else -1
    print(f"Fold {fold_i+1}: late-pos MAE={mae_late:.4f}  best_iter={best_it}  ({elapsed:.0f}s)", flush=True)

test_lp_avg = np.mean(test_lp_list, axis=0)
os.makedirs('results/oracle_seq', exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_latepos.npy', oof_lp)
np.save('results/oracle_seq/test_C_lgb_latepos.npy', test_lp_avg)
print("Saved oof_seqC_lgb_latepos.npy, test_C_lgb_latepos.npy", flush=True)

# Quick blend summary (only meaningful for late positions)
with open('results/mega33_final.pkl','rb') as f:
    d2 = pickle.load(f)
train_ls2 = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos2 = {row['ID']:i for i,row in train_ls2.iterrows()}
id_to_ls2 = [ls_pos2[rid] for rid in train_raw['ID'].values]
mega_oof2  = d2['meta_avg_oof'][id_to_ls2]
rank_oof2  = np.load('results/ranking/rank_adj_oof.npy')[id_to_ls2]
iter1_oof2 = np.load('results/iter_pseudo/round1_oof.npy')[id_to_ls2]
iter2_oof2 = np.load('results/iter_pseudo/round2_oof.npy')[id_to_ls2]
iter3_oof2 = np.load('results/iter_pseudo/round3_oof.npy')[id_to_ls2]
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.03456830669223538,
          iter_r3=0.031038826035934514)
fixed2 = (fw['mega33']*mega_oof2 + fw['rank_adj']*rank_oof2 +
          fw['iter_r1']*iter1_oof2 + fw['iter_r2']*iter2_oof2 + fw['iter_r3']*iter3_oof2)
y2 = train_raw['avg_delay_minutes_next_30m'].values
xgb2 = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv22 = np.load('results/oracle_seq/oof_seqC_log_v2.npy')

# Per-position MAE analysis
row_sc_all = train_raw['row_in_sc'].values
print(f"\nPer-position analysis (late-pos oracle):", flush=True)
for pos in range(LATE_MIN, 25):
    mask = row_sc_all == pos
    f_mae = np.mean(np.abs(fixed2[mask] - y2[mask]))
    lp_mae = np.mean(np.abs(oof_lp[mask] - y2[mask]))
    lv2_mae = np.mean(np.abs(lv22[mask] - y2[mask]))
    xgb_mae = np.mean(np.abs(xgb2[mask] - y2[mask]))
    print(f"  pos {pos:2d}: FIXED={f_mae:.3f}  latepos={lp_mae:.3f}  lv2={lv2_mae:.3f}  xgb={xgb_mae:.3f}  delta={lp_mae-f_mae:+.4f}", flush=True)

# Best static blend
fixed_mae = np.mean(np.abs(fixed2 - y2))
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.61, 0.04):
    bl = fixed2.copy()
    for pos in range(LATE_MIN, 25):
        mask = row_sc_all == pos
        bl[mask] = (1-w)*fixed2[mask] + w*oof_lp[mask]
    mm = np.mean(np.abs(bl - y2))
    if mm < best_m: best_m = mm; best_w = w
print(f"\nBest static blend (late-only): w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}", flush=True)
print("Done.", flush=True)
