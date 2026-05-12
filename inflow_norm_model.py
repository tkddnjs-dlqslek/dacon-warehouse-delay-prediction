"""
Inflow-normalized features model.
Unseen layouts have +64% order_inflow. If we normalize key features by inflow,
the model sees "per-order" utilization — layout-invariant signal.
Train with GroupKFold, check if it improves OOF or blend vs oracle_NEW.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
groups = train_raw['layout_id'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

# Reconstruct oracle_NEW OOF
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen = oracle_new_t[unseen_mask].mean()
print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

# Load v30 features (already good baseline)
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    blob = pickle.load(f)
train_fe = blob['train_fe']
feat_cols = list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    test_fe = pickle.load(f)
fold_ids = np.load('results/eda_v30/fold_idx.npy')

# Check inflow stats: seen vs unseen
tr_inflow = train_fe['order_inflow_15m'].values
te_inflow = test_fe['order_inflow_15m'].values
te_unseen_inflow = te_inflow[unseen_mask[te_rid_to_ls]]  # unseen test rows in ls order
print(f"\nTrain inflow mean: {tr_inflow.mean():.2f}")
print(f"Test all inflow mean: {te_inflow.mean():.2f}")
print(f"Test unseen inflow mean: {te_unseen_inflow.mean():.2f} (+{(te_unseen_inflow.mean()/tr_inflow.mean()-1)*100:.0f}% vs train)")

# Key numerical features to normalize by inflow
NORM_FEATS = [
    'pack_utilization', 'conveyor_speed_mps', 'outbound_truck_wait_min',
    'congestion_score', 'robot_utilization', 'avg_trip_distance',
    'charge_queue_length', 'fault_count_15m', 'aisle_traffic_score',
    'loading_dock_util', 'staging_area_util', 'express_lane_util',
    'storage_density_pct'
]
# Only use columns that exist in feat_cols
NORM_FEATS = [f for f in NORM_FEATS if f in train_fe.columns]
print(f"\nNormalizing {len(NORM_FEATS)} features by order_inflow_15m")

def add_inflow_normalized(df):
    eps = 0.1
    inflow = df['order_inflow_15m'].values + eps
    new_cols = {}
    for col in NORM_FEATS:
        if col in df.columns:
            new_cols[f'{col}_per_inflow'] = df[col].values / inflow
    return pd.DataFrame(new_cols, index=df.index)

norm_tr = add_inflow_normalized(train_fe)
norm_te = add_inflow_normalized(test_fe)
norm_cols = list(norm_tr.columns)
print(f"Added {len(norm_cols)} normalized features")

# Combine: original v30 + normalized
X_tr_extra = norm_tr.values.astype(np.float32)
X_te_extra = norm_te.values.astype(np.float32)
X_tr_v30 = train_fe[feat_cols].values.astype(np.float32)
X_te_v30 = test_fe[feat_cols].values.astype(np.float32)

X_tr = np.hstack([X_tr_v30, X_tr_extra])
X_te = np.hstack([X_te_v30, X_te_extra])
print(f"Feature shape: {X_tr.shape}")

y = train_fe['avg_delay_minutes_next_30m'].values.astype(np.float64)
y_log = np.log1p(y)

PARAMS = dict(
    objective="huber", n_estimators=3000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

print("\nTraining with inflow-normalized features (GroupKFold)...")
oof = np.zeros(len(y))
test_pred = np.zeros(len(X_te))

from sklearn.model_selection import GroupKFold
import time
# fold_ids is by ls-order, groups_ls is layout_id in ls order
groups_ls = train_fe['layout_id'].values

for f in range(5):
    tf = time.time()
    val_mask = fold_ids == f
    tr_mask = ~val_mask
    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr_mask], y_log[tr_mask],
          eval_set=[(X_tr[val_mask], y_log[val_mask])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    oof[val_mask] = np.clip(np.expm1(m.predict(X_tr[val_mask])), 0, None)
    test_pred += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5
    fm = np.mean(np.abs(y[val_mask] - oof[val_mask]))
    print(f"  fold {f}: MAE={fm:.4f} it={m.best_iteration_} ({time.time()-tf:.0f}s)", flush=True)

oof = np.clip(oof, 0, None)
oof_mae = np.mean(np.abs(y - oof))
print(f"\nInflow-norm model OOF (ls order): {oof_mae:.4f}")

# Convert to rid order
oof_rid = oof[id2]
test_rid = test_pred[te_rid_to_ls]
oof_rid_mae = np.mean(np.abs(y_true - oof_rid))
corr = np.corrcoef(fw4_oo, oof_rid)[0,1]
print(f"Inflow-norm OOF (rid order): {oof_rid_mae:.4f}  corr={corr:.4f}")
print(f"test_unseen: {test_rid[unseen_mask].mean():.3f}  test_seen: {test_rid[~unseen_mask].mean():.3f}")

print()
print("=== Blend with oracle_NEW ===")
for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15]:
    bl_oof = np.clip((1-w)*fw4_oo + w*oof_rid, 0, None)
    bl_t = np.clip((1-w)*oracle_new_t + w*test_rid, 0, None)
    bl_mae = np.mean(np.abs(y_true - bl_oof))
    bl_unseen = bl_t[unseen_mask].mean()
    print(f"  w={w:.2f}: OOF={bl_mae:.4f} ({bl_mae-oracle_mae:+.4f})  unseen={bl_unseen:.3f} ({bl_unseen-oracle_unseen:+.3f})")

# Save npy
np.save('results/inflow_norm_oof.npy', oof_rid)
np.save('results/inflow_norm_test.npy', test_rid)
print("\nSaved results/inflow_norm_oof.npy, inflow_norm_test.npy")

# Save best blend if it improves OOF
best_w, best_mae = 0, oracle_mae
for w in np.arange(0.01, 0.31, 0.01):
    bl = np.clip((1-w)*fw4_oo + w*oof_rid, 0, None)
    m = np.mean(np.abs(y_true - bl))
    if m < best_mae: best_mae, best_w = m, w

if best_w > 0:
    bl_t = np.clip((1-best_w)*oracle_new_t + best_w*test_rid, 0, None)
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = bl_t
    fname = f"FINAL_oN_inflnorm_w{int(best_w*100):02d}_OOF{best_mae:.4f}.csv"
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}  (OOF gain: {oracle_mae-best_mae:+.4f})")
else:
    print("No improvement from inflow-norm model blend.")
print("Done.")
