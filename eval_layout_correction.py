"""
Layout-level bias correction layer.
Hypothesis: mega33 has systematic per-layout biases (test layouts behave differently).
Method: train a stacking layer on [mega33_oof, layout_info_features] → target.
This captures layout-specific adjustments beyond the base model.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
te_ls_pos= {row['ID']:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw['ID'].values]
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Current best blend (row_id order)
with open('results/mega33_final.pkl','rb') as f: d = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d['meta_avg_oof'][id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_te = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_te = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_te = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

best_oof  = 0.64*fixed_oof  + 0.12*xgb_o  + 0.16*lv2_o  + 0.08*rem_o
best_test = 0.64*fixed_test + 0.12*xgb_te + 0.16*lv2_te + 0.08*rem_te
curr_mae = np.mean(np.abs(best_oof - y_true))
print(f"Current best OOF: {curr_mae:.5f}")

# Best cascade (dual gate)
# Load refined3 prediction as starting point
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]
lgb_rh_oof  = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
lgb_rm_oof  = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
lgb_rh_test = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
lgb_rm_test = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

# Reproduce refined3 best: dual p0.11_h_w0.030+p0.25_m_w0.030
m11 = (clf_oof  > 0.11).astype(float); m11_te = (clf_test > 0.11).astype(float)
m25 = (clf_oof  > 0.25).astype(float); m25_te = (clf_test > 0.25).astype(float)
cascade_oof  = (1-m11*0.03)*best_oof  + m11*0.03*lgb_rh_oof
cascade_test = (1-m11_te*0.03)*best_test + m11_te*0.03*lgb_rh_test
cascade_oof  = (1-m25*0.03)*cascade_oof  + m25*0.03*lgb_rm_oof
cascade_test = (1-m25_te*0.03)*cascade_test + m25_te*0.03*lgb_rm_test
casc_mae = np.mean(np.abs(cascade_oof - y_true))
print(f"Cascade (refined3) OOF: {casc_mae:.5f}")

# Layout info features
layout_info = pd.read_csv('layout_info.csv')
num_cols = ['aisle_width_avg','intersection_count','one_way_ratio','pack_station_count',
            'charger_count','layout_compactness','zone_dispersion','robot_total',
            'building_age_years','floor_area_sqm']
layout_info = layout_info[['layout_id'] + num_cols].fillna(0)

def add_layout_feats(df_raw):
    merged = df_raw[['layout_id']].merge(layout_info, on='layout_id', how='left')
    return merged[num_cols].values.astype(np.float32)

layout_tr = add_layout_feats(train_raw)
layout_te = add_layout_feats(test_raw)

# Derived layout features
robot_per_pack = (layout_tr[:, num_cols.index('robot_total')] /
                  np.clip(layout_tr[:, num_cols.index('pack_station_count')], 1, None)).reshape(-1,1)
robot_per_pack_te = (layout_te[:, num_cols.index('robot_total')] /
                     np.clip(layout_te[:, num_cols.index('pack_station_count')], 1, None)).reshape(-1,1)

# Build feature matrix: [cascade_pred, layout_feats, robot_per_pack, log_pred]
X_tr_corr = np.hstack([
    cascade_oof.reshape(-1,1),
    np.log1p(np.maximum(0, cascade_oof)).reshape(-1,1),
    layout_tr,
    robot_per_pack,
])
X_te_corr = np.hstack([
    cascade_test.reshape(-1,1),
    np.log1p(np.maximum(0, cascade_test)).reshape(-1,1),
    layout_te,
    robot_per_pack_te,
])

print(f"\nCorrection layer features: {X_tr_corr.shape[1]}", flush=True)

# GroupKFold correction layer (same groups as main model)
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
folds = list(gkf.split(np.arange(len(y_true)), groups=groups))

print("\n[LGB correction layer]", flush=True)
corr_oof  = np.zeros(len(y_true))
corr_test = np.zeros(len(test_raw))

for n_est in [200, 500, 1000]:
    oof = np.zeros(len(y_true)); te = np.zeros(len(test_raw))
    for i, (tr_idx, val_idx) in enumerate(folds):
        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=n_est, learning_rate=0.05,
            num_leaves=31, max_depth=4, min_child_samples=100,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, n_jobs=-1
        )
        m.fit(X_tr_corr[tr_idx], y_true[tr_idx],
              eval_set=[(X_tr_corr[val_idx], y_true[val_idx])],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        oof[val_idx] = m.predict(X_tr_corr[val_idx])
        te += m.predict(X_te_corr) / 5
    oof = np.clip(oof, 0, None); te = np.clip(te, 0, None)
    mm = mean_absolute_error(y_true, oof)
    print(f"  n_est={n_est}: OOF MAE={mm:.5f}  delta={mm-curr_mae:+.5f}", flush=True)
    if mm < mean_absolute_error(y_true, corr_oof) or corr_oof.sum() == 0:
        corr_oof = oof; corr_test = te

corr_mae = mean_absolute_error(y_true, corr_oof)
print(f"\nBest correction OOF: {corr_mae:.5f}  delta={corr_mae-curr_mae:+.5f}")
print(f"vs cascade: {corr_mae-casc_mae:+.5f}")

CURRENT_BEST_LB = 8.3825
if corr_mae < min(casc_mae, CURRENT_BEST_LB) - 0.001:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, corr_test)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_layout_corr_OOF{corr_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"\n*** SAVED: {fname} ***")
else:
    print(f"\nOOF {corr_mae:.5f} not better enough (threshold: {min(casc_mae,CURRENT_BEST_LB)-0.001:.5f})")
print("Done.")
