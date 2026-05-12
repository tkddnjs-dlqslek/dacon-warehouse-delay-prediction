import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os, pickle
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2 = [te_ls_pos[i] for i in test_raw['ID'].values]

# Rebuild fw4_oo
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]
mega34_oof = d34['meta_avg_oof'][id2]
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
residuals_train = y_true - fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
sub_tmpl = pd.read_csv('sample_submission.csv')

print("="*70)
print("K-NN Layout Calibration: find nearest training layouts in feature space")
print("Apply their oracle_NEW residuals as corrections for unseen test layouts")
print("="*70)

feat_cols = [c for c in train_raw.columns
             if c not in ('ID','_row_id','layout_id','scenario_id',
                          'avg_delay_minutes_next_30m','timeslot')]

# Compute layout-level mean features (fill NaN)
train_raw['_resid'] = residuals_train
layout_grp = train_raw.groupby('layout_id')
layout_feat_tr = layout_grp[feat_cols].mean().apply(lambda col: col.fillna(col.median()))
layout_resid_tr = layout_grp['_resid'].mean()
layout_ids_tr_arr = np.array(list(layout_feat_tr.index))

test_layout_feat = test_raw.groupby('layout_id')[feat_cols].mean()
test_layout_feat = test_layout_feat.apply(lambda col: col.fillna(col.median()))

unseen_test_layouts = test_raw[unseen_mask]['layout_id'].unique()
seen_test_layouts   = test_raw[seen_mask]['layout_id'].unique()

# Feature matrix for KNN
sc = StandardScaler()
X_tr_lv = sc.fit_transform(layout_feat_tr.values)  # 250 training layouts
X_u_lv  = sc.transform(test_layout_feat.loc[unseen_test_layouts].values)  # 50 unseen
X_s_lv  = sc.transform(test_layout_feat.loc[seen_test_layouts].values)    # 50 seen

y_tr_resid = layout_resid_tr.values

print(f"\n  Training layouts: {len(X_tr_lv)}")
print(f"  Unseen test layouts: {len(X_u_lv)}")
print(f"  Seen test layouts: {len(X_s_lv)}")

# ============================================================
# K-NN calibration: for each unseen layout, find K nearest
# training layouts and use their mean residual as correction
# ============================================================
from sklearn.neighbors import KNeighborsRegressor

print(f"\n--- K-NN calibration for unseen test layouts ---")
print(f"  {'K':>4}  {'mean_corr_unseen':>16}  {'std_corr':>10}  {'mean_corr_seen':>14}")

knn_preds = {}
for k in [1, 3, 5, 10, 20, 50, 100]:
    knn = KNeighborsRegressor(n_neighbors=min(k, len(X_tr_lv)), weights='distance')
    knn.fit(X_tr_lv, y_tr_resid)
    pred_u = knn.predict(X_u_lv)
    pred_s = knn.predict(X_s_lv)
    print(f"  K={k:4d}: unseen={pred_u.mean():+.4f} ± {pred_u.std():.4f}   seen={pred_s.mean():+.4f}")
    knn_preds[k] = pred_u

# Sanity check: KNN on seen test layouts should give ~+3.5
print(f"\n  Expected seen correction: ~+3.5 (actual training bias for seen layouts)")

# Apply K=5 and K=20 corrections to unseen test
for k in [5, 10, 20]:
    pred_u_k = knn_preds[k]
    ct_knn = oracle_new_t.copy()
    for i, lid in enumerate(unseen_test_layouts):
        te_m = (test_raw['layout_id'] == lid).values
        ct_knn[te_m] = oracle_new_t[te_m] + pred_u_k[i]
    ct_knn = np.clip(ct_knn, 0, None)
    du_knn = ct_knn[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    print(f"\n  K={k}: layout-level correction Δ={du_knn:+.4f}  seen={ct_knn[seen_mask].mean():.3f}  unseen={ct_knn[unseen_mask].mean():.3f}")
    fname_knn = f"FINAL_NEW_oN_knn{k}_OOF8.3825.csv"
    sub_knn = sub_tmpl.copy(); sub_knn['avg_delay_minutes_next_30m'] = ct_knn
    sub_knn.to_csv(fname_knn, index=False)
    print(f"  Saved: {fname_knn}")

# ============================================================
# Leave-one-layout-out validation of KNN approach
# For each training layout, find K nearest OTHER training layouts
# and predict its residual → measure accuracy
# ============================================================
print(f"\n--- Leave-one-layout-out validation of KNN ---")
print(f"  {'K':>4}  {'LOOO_MAE':>10}  {'mean_pred':>10}")
for k in [1, 3, 5, 10, 20, 50]:
    knn = KNeighborsRegressor(n_neighbors=min(k, len(X_tr_lv)-1), weights='distance')
    looo_preds = np.zeros(len(X_tr_lv))
    for i in range(len(X_tr_lv)):
        X_other = np.delete(X_tr_lv, i, axis=0)
        y_other  = np.delete(y_tr_resid, i)
        knn.fit(X_other, y_other)
        looo_preds[i] = knn.predict(X_tr_lv[i:i+1])[0]
    looo_mae = np.mean(np.abs(looo_preds - y_tr_resid))
    print(f"  K={k:4d}: LOOO_MAE={looo_mae:.4f}  mean_pred={looo_preds.mean():+.4f}")

# ============================================================
# Cosine similarity approach: weight by feature similarity
# ============================================================
print(f"\n--- Cosine-weighted correction ---")
cos_sim_u = cosine_similarity(X_u_lv, X_tr_lv)  # 50 x 250
cos_sim_s = cosine_similarity(X_s_lv, X_tr_lv)  # 50 x 250

# Softmax weighting (soft version of KNN)
def softmax_weighted(sim_matrix, y_target, temperature=1.0):
    scores = sim_matrix / temperature
    scores = scores - scores.max(axis=1, keepdims=True)  # numerical stability
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=1, keepdims=True)
    return (weights * y_target[np.newaxis, :]).sum(axis=1)

for temp in [0.1, 0.2, 0.5, 1.0, 2.0]:
    pred_u_cos = softmax_weighted(cos_sim_u, y_tr_resid, temp)
    pred_s_cos = softmax_weighted(cos_sim_s, y_tr_resid, temp)
    print(f"  temp={temp}: unseen={pred_u_cos.mean():+.4f}  seen={pred_s_cos.mean():+.4f}")

# Apply best temperature
temp_best = 0.2
pred_u_cos_best = softmax_weighted(cos_sim_u, y_tr_resid, temp_best)
ct_cos = oracle_new_t.copy()
for i, lid in enumerate(unseen_test_layouts):
    te_m = (test_raw['layout_id'] == lid).values
    ct_cos[te_m] = oracle_new_t[te_m] + pred_u_cos_best[i]
ct_cos = np.clip(ct_cos, 0, None)
du_cos = ct_cos[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
fname_cos = "FINAL_NEW_oN_cosineCorr_OOF8.3825.csv"
sub_cos = sub_tmpl.copy(); sub_cos['avg_delay_minutes_next_30m'] = ct_cos
sub_cos.to_csv(fname_cos, index=False)
print(f"\n  Cosine correction (temp={temp_best}): Δ={du_cos:+.4f}  seen={ct_cos[seen_mask].mean():.3f}  unseen={ct_cos[unseen_mask].mean():.3f}")
print(f"  Saved: {fname_cos}")

print("\nDone.")
