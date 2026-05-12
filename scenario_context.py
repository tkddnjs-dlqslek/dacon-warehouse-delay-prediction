import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
groups = train_raw['layout_id'].values

# Load oracle_NEW OOF
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

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
train_raw['_oof'] = fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
id_order = test_raw['ID'].values
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
test_raw['_oof'] = oracle_new_t

# --- Build scenario-level context features ---
# Features to aggregate at (layout_id, scenario_id) level
NUM_FEATS = ['order_inflow_15m','pack_utilization','conveyor_speed_mps',
             'outbound_truck_wait_min','congestion_score','robot_utilization',
             'avg_trip_distance','charge_queue_length','fault_count_15m',
             'aisle_traffic_score','loading_dock_util','staging_area_util',
             'express_lane_util','storage_density_pct','pack_utilization']
NUM_FEATS = list(dict.fromkeys(NUM_FEATS))  # deduplicate

print("Building scenario-level context features...")
def add_scenario_context(df, feats, agg_src=None):
    """Add per-(layout_id,scenario_id) aggregation features.
    agg_src: optional df to compute aggregations from (for leave-one-out style OOF).
    If None, compute from df itself (for test).
    """
    if agg_src is None:
        agg_src = df
    grp = agg_src.groupby(['layout_id','scenario_id'])[feats].agg(['mean','std','max','min'])
    grp.columns = [f'sc_{f}_{a}' for f,a in grp.columns]
    grp = grp.reset_index()
    return df.merge(grp, on=['layout_id','scenario_id'], how='left')

train_ctx = add_scenario_context(train_raw, NUM_FEATS)
test_ctx  = add_scenario_context(test_raw, NUM_FEATS)

ctx_cols = [c for c in train_ctx.columns if c.startswith('sc_')]
print(f"Added {len(ctx_cols)} scenario context features")

# --- Build base row features ---
SKIP = {'ID','layout_id','scenario_id','avg_delay_minutes_next_30m','_row_id','_oof','_resid'}
base_feats = [c for c in train_ctx.columns if c not in SKIP and train_ctx[c].dtype != object]

print(f"Total features: {len(base_feats)} (base) + {len(ctx_cols)} (context)")

# --- GroupKFold MAE comparison ---
gkf = GroupKFold(n_splits=5)

lgb_params = dict(
    objective='mae', metric='mae', verbose=-1,
    n_estimators=500, learning_rate=0.05,
    num_leaves=63, min_child_samples=20,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    reg_alpha=0.1, reg_lambda=0.1, random_state=42
)

# Test 1: oracle_NEW OOF baseline
print(f"\nBaseline oracle_NEW OOF: {np.mean(np.abs(y_true - fw4_oo)):.4f}")

# Test 2: New model with scenario context features (base_feats = original + context)
print("\nRunning GroupKFold with scenario context features...")
X_tr = train_ctx[base_feats].values
oof_ctx = np.zeros(len(y_true))
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_tr, y_true, groups)):
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_tr[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr[val_idx], y_true[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
    oof_ctx[val_idx] = model.predict(X_tr[val_idx])
    mae_f = np.mean(np.abs(y_true[val_idx] - oof_ctx[val_idx]))
    print(f"  Fold {fold+1}: MAE={mae_f:.4f}")

oof_ctx = np.clip(oof_ctx, 0, None)
mae_ctx = np.mean(np.abs(y_true - oof_ctx))
print(f"  Context model OOF MAE: {mae_ctx:.4f}  (baseline: {np.mean(np.abs(y_true-fw4_oo)):.4f})")

# Test 3: Blend oracle_NEW + context model
print("\nBlend sweep: oracle_NEW × α + context_model × (1-α)")
for alpha in [0.9, 0.8, 0.7, 0.6, 0.5]:
    blend = alpha * fw4_oo + (1-alpha) * oof_ctx
    blend = np.clip(blend, 0, None)
    mae = np.mean(np.abs(y_true - blend))
    print(f"  alpha={alpha:.1f}: MAE={mae:.4f}")

# If context model improves, train full model and save
if mae_ctx < np.mean(np.abs(y_true - fw4_oo)):
    print(f"\n★ Context model IMPROVES OOF! Training full model for test predictions...")
    X_te = test_ctx[base_feats].values
    full_model = lgb.LGBMRegressor(**lgb_params)
    full_model.fit(X_tr, y_true,
                   callbacks=[lgb.log_evaluation(-1)])
    test_pred_ctx = np.clip(full_model.predict(X_te), 0, None)

    # Find best blend weight
    best_alpha = 0.9; best_mae = 99
    for alpha in np.arange(0.5, 1.0, 0.05):
        blend_oof = alpha * fw4_oo + (1-alpha) * oof_ctx
        mae = np.mean(np.abs(y_true - np.clip(blend_oof, 0, None)))
        if mae < best_mae:
            best_mae = mae; best_alpha = alpha

    print(f"  Best blend alpha={best_alpha:.2f}, OOF MAE={best_mae:.4f}")
    blend_test = np.clip(best_alpha * oracle_new_t + (1-best_alpha) * test_pred_ctx, 0, None)

    sub_tmpl = pd.read_csv('sample_submission.csv')
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = blend_test
    fname = f'FINAL_NEW_oN_ctx{int(best_alpha*10)}_{int((1-best_alpha)*10)}_OOF{best_mae:.4f}.csv'
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}")
    print(f"  test_mean_seen={blend_test[test_raw['layout_id'].isin(train_raw['layout_id'].unique()).values].mean():.3f}")
    print(f"  test_mean_unseen={blend_test[~test_raw['layout_id'].isin(train_raw['layout_id'].unique()).values].mean():.3f}")
else:
    print(f"\n✗ Context model does not improve OOF. No submission generated.")

print("\nDone.")
