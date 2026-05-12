import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import os
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
residuals_train = y_true - fw4_oo
train_raw['_resid'] = residuals_train
train_raw['_oof'] = fw4_oo

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
test_raw['_oN'] = oracle_new_t
sub_tmpl = pd.read_csv('sample_submission.csv')

tr_lv = train_raw.groupby('layout_id').agg(
    oof_mean=('_oof','mean'), resid_mean=('_resid','mean'),
    pu=('pack_utilization','mean'), tw=('outbound_truck_wait_min','mean'),
    conv=('conveyor_speed_mps','mean'),
).reset_index()
y_lv = tr_lv['resid_mean'].values

unseen_lids = test_raw[unseen_mask]['layout_id'].unique()
te_lv_u = test_raw.groupby('layout_id').agg(
    oN_mean=('_oN','mean'), pu=('pack_utilization','mean'),
    tw=('outbound_truck_wait_min','mean'), conv=('conveyor_speed_mps','mean'),
).reset_index()
te_lv_u = te_lv_u[te_lv_u['layout_id'].isin(unseen_lids)].copy()

feats_tr = tr_lv[['oof_mean','pu','tw','conv']].values
feats_te = te_lv_u[['oN_mean','pu','tw','conv']].values
y_lv = tr_lv['resid_mean'].values

import warnings; warnings.filterwarnings('ignore')

def fit_predict(model_cls, model_kw, X_tr, X_te, y, scale=False):
    preds_looo = []
    for i in range(len(y)):
        Xtr_ = np.delete(X_tr, i, 0); ytr_ = np.delete(y, i)
        if scale:
            sc_ = StandardScaler(); Xtr_s = sc_.fit_transform(Xtr_); Xte_s = sc_.transform(X_tr[i:i+1])
        else:
            Xtr_s = Xtr_; Xte_s = X_tr[i:i+1]
        m = model_cls(**model_kw); m.fit(Xtr_s, ytr_)
        preds_looo.append(m.predict(Xte_s)[0])
    if scale:
        sc_f = StandardScaler(); X_tr_s = sc_f.fit_transform(X_tr); X_te_s = sc_f.transform(X_te)
    else:
        X_tr_s = X_tr; X_te_s = X_te
    m_f = model_cls(**model_kw); m_f.fit(X_tr_s, y)
    test_preds = m_f.predict(X_te_s)
    return np.array(preds_looo), test_preds

print("="*70)
print("GBM vs Ridge: Per-Layout Comparison & Ensemble")
print("="*70)

# Fit the three key models
_, ridge_test = fit_predict(Ridge, {'alpha':100}, feats_tr, feats_te, y_lv, scale=True)
_, gbm_test   = fit_predict(GradientBoostingRegressor,
                             {'max_depth':3,'n_estimators':30,'learning_rate':0.1,'random_state':42},
                             feats_tr, feats_te, y_lv, scale=False)
_, rf_test    = fit_predict(RandomForestRegressor,
                             {'n_estimators':200,'max_depth':3,'random_state':42},
                             feats_tr, feats_te, y_lv, scale=False)

# Load iso corrections
iso_df = pd.read_csv('FINAL_NEW_oN_iso_lvL75_OOF8.3825.csv')
iso_t = iso_df.set_index('ID').reindex(id_order)['avg_delay_minutes_next_30m'].values
iso_by_lid = {lid: (iso_t[test_raw['layout_id']==lid].mean() -
                    oracle_new_t[test_raw['layout_id']==lid].mean())
              for lid in unseen_lids}

te_lv_u['iso'] = te_lv_u['layout_id'].map(iso_by_lid)
te_lv_u['ridge_raw'] = ridge_test
te_lv_u['gbm_raw'] = gbm_test
te_lv_u['rf_raw'] = rf_test

# Scale all to Δ=5.5
r_offset = 5.5 - ridge_test.mean()
g_offset = 5.5 - gbm_test.mean()
f_offset = 5.5 - rf_test.mean()
te_lv_u['ridge5p5'] = ridge_test + r_offset
te_lv_u['gbm5p5']   = gbm_test + g_offset
te_lv_u['rf5p5']    = rf_test + f_offset

print(f"\n  Per-layout corrections (all scaled to mean=5.5):")
print(f"  {'layout_id':10s}  {'oN_mean':7s}  {'pu':5s}  {'iso':6s}  {'ridge':6s}  {'GBM':6s}  {'RF':5s}  {'d_gi':6s}")
for _, row in te_lv_u.sort_values('gbm5p5').iterrows():
    print(f"  {row['layout_id']:10s}  {row['oN_mean']:7.2f}  {row['pu']:.3f}  "
          f"{row['iso']:6.2f}  {row['ridge5p5']:6.2f}  {row['gbm5p5']:6.2f}  "
          f"{row['rf5p5']:5.2f}  {row['iso']-row['gbm5p5']:6.2f}")

# Compute ensemble of GBM + Ridge
print(f"\n{'='*70}")
print("Ensembles of GBM + Ridge (scaled to 5.5)")
print(f"{'='*70}")

def save_blend(lid_corr, label, fname):
    corr_arr = np.zeros(len(test_raw))
    for lid, c in lid_corr.items():
        corr_arr[(test_raw['layout_id'] == lid).values] = c
    ct = oracle_new_t.copy()
    ct[unseen_mask] = oracle_new_t[unseen_mask] + corr_arr[unseen_mask]
    ct = np.clip(ct, 0, None)
    du = ct[unseen_mask].mean() - oracle_new_t[unseen_mask].mean()
    std = corr_arr[unseen_mask].std()
    print(f"  {label:50s}: D={du:+.4f}  std={std:.3f}  min={corr_arr[unseen_mask].min():.2f}  max={corr_arr[unseen_mask].max():.2f}")
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)

lids = te_lv_u['layout_id'].values
for wg in [0.3, 0.4, 0.5, 0.6, 0.7]:
    wr = 1 - wg
    ens_raw = wg * gbm_test + wr * ridge_test
    ens_scaled = ens_raw + (5.5 - ens_raw.mean())
    lid_to_c = dict(zip(lids, ens_scaled))
    save_blend(lid_to_c, f'GBM({wg:.1f}) + Ridge_a100({wr:.1f}) scaled5.5',
               f'FINAL_NEW_oN_gbmRidge_w{int(wg*10)}_OOF8.3825.csv')

# RF + Ridge
for wf2 in [0.3, 0.5, 0.7]:
    wr = 1 - wf2
    ens_raw = wf2 * rf_test + wr * ridge_test
    ens_scaled = ens_raw + (5.5 - ens_raw.mean())
    lid_to_c = dict(zip(lids, ens_scaled))
    save_blend(lid_to_c, f'RF({wf2:.1f}) + Ridge_a100({wr:.1f}) scaled5.5',
               f'FINAL_NEW_oN_rfRidge_w{int(wf2*10)}_OOF8.3825.csv')

# GBM + Ridge + flat
print(f"\n  3-way GBM + Ridge + flat:")
flat_arr_u = np.full(len(lids), 5.5)
for (wg, wr, wf2) in [(0.33,0.33,0.34),(0.50,0.30,0.20),(0.40,0.40,0.20),(0.25,0.50,0.25)]:
    ens_raw = wg * gbm_test + wr * ridge_test + wf2 * flat_arr_u
    ens_scaled = ens_raw + (5.5 - ens_raw.mean())
    lid_to_c = dict(zip(lids, ens_scaled))
    save_blend(lid_to_c, f'GBM({wg:.2f})+Ridge({wr:.2f})+flat({wf2:.2f}) 5.5',
               f'FINAL_NEW_oN_gbmRidgeFlat_{int(wg*100):02d}_{int(wr*100):02d}_OOF8.3825.csv')

# LOOO scores for all ensembles
print(f"\n{'='*70}")
print("LOOO MAE for GBM+Ridge ensembles (on training layouts)")
print(f"{'='*70}")
looo_ridge, _ = fit_predict(Ridge, {'alpha':100}, feats_tr, feats_tr, y_lv, scale=True)
looo_gbm,   _ = fit_predict(GradientBoostingRegressor,
                              {'max_depth':3,'n_estimators':30,'learning_rate':0.1,'random_state':42},
                              feats_tr, feats_tr, y_lv, scale=False)
looo_rf,    _ = fit_predict(RandomForestRegressor,
                              {'n_estimators':200,'max_depth':3,'random_state':42},
                              feats_tr, feats_tr, y_lv, scale=False)

print(f"  Ridge LOOO: {np.mean(np.abs(y_lv - looo_ridge)):.4f}")
print(f"  GBM LOOO:   {np.mean(np.abs(y_lv - looo_gbm)):.4f}")
print(f"  RF LOOO:    {np.mean(np.abs(y_lv - looo_rf)):.4f}")
for wg in [0.3, 0.5, 0.7]:
    wr = 1 - wg
    ens_looo = wg * looo_gbm + wr * looo_ridge
    mae = np.mean(np.abs(y_lv - ens_looo))
    print(f"  GBM({wg:.1f})+Ridge({wr:.1f}) LOOO: {mae:.4f}")

# Best ensemble check per target
print(f"\n  Best ensemble test predictions distribution:")
best_ens_raw = 0.5 * gbm_test + 0.5 * ridge_test
best_ens5p5 = best_ens_raw + (5.5 - best_ens_raw.mean())
te_lv_u['best_ens5p5'] = best_ens5p5
print(f"  GBM(0.5)+Ridge(0.5): mean={best_ens5p5.mean():.3f}  std={best_ens5p5.std():.3f}  "
      f"min={best_ens5p5.min():.2f}  max={best_ens5p5.max():.2f}")

# Save best ensemble (GBM 50% + Ridge 50%)
lid_to_best_ens = dict(zip(lids, best_ens5p5))
save_blend(lid_to_best_ens, 'GBM(0.5)+Ridge(0.5) [BEST_ENS] scaled5.5',
           'FINAL_NEW_oN_gbmRidgeBest_5p5_OOF8.3825.csv')
best_ens5p77 = best_ens_raw + (5.77 - best_ens_raw.mean())
lid_to_best_ens77 = dict(zip(lids, best_ens5p77))
save_blend(lid_to_best_ens77, 'GBM(0.5)+Ridge(0.5) [BEST_ENS] scaled5.77',
           'FINAL_NEW_oN_gbmRidgeBest_5p77_OOF8.3825.csv')

print(f"\n  All keys comparison for extreme layouts:")
for lid in ['WH_234','WH_241','WH_096','WH_193','WH_214','WH_101','WH_283']:
    row = te_lv_u[te_lv_u['layout_id']==lid]
    if len(row) > 0:
        r = row.iloc[0]
        ens = 0.5*r['gbm5p5'] + 0.5*r['ridge5p5']
        print(f"  {lid}: oN={r['oN_mean']:.1f} pu={r['pu']:.3f}  "
              f"iso={r['iso']:.2f}  ridge={r['ridge5p5']:.2f}  gbm={r['gbm5p5']:.2f}  "
              f"rf={r['rf5p5']:.2f}  ens={ens:.2f}")

print("\nDone.")
