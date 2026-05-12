"""
KNN Residual Correction for oracle_NEW
For each validation/test layout, find K nearest training layouts by feature similarity.
Apply their mean oracle residual as an additive correction.
Validated via GroupKFold (out-of-sample).

Key insight: corr(pack_mean, residual)=0.62. High-pack training layouts have ~+18 residual.
For test layouts WH_283 (pack=0.975), WH_201 (pack=0.936), WH_246 (pack=0.911): no training
analog exists, so the extrapolated correction should be large.

We use features: [pack_mean, inflow_mean, robot_mean] normalized by training std.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls   = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof  = (fw['mega33']*d33['meta_avg_oof'][id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_test = (fw['mega33']*d33['meta_avg_test'][te_id2]
            + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
            + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
            + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
            + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])
xgb_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o_oof  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o_oof  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_o_test = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o_test = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o_test = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
oracle_oof  = np.clip(0.64*fixed_oof + 0.12*xgb_o_oof + 0.16*lv2_o_oof + 0.08*rem_o_oof, 0, None)
oracle_test = np.clip(0.64*fixed_test + 0.12*xgb_o_test + 0.16*lv2_o_test + 0.08*rem_o_test, 0, None)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
print(f'oracle_NEW baseline OOF: {mae(oracle_oof):.5f}  test_mean={oracle_test.mean():.3f}')

# Build layout-level feature matrix
FEAT_COLS = ['order_inflow_15m', 'pack_utilization', 'robot_active', 'congestion_score',
             'fault_count_15m', 'blocked_path_15m']
tr_ly = train_raw.groupby('layout_id')[FEAT_COLS].mean().reset_index()
te_ly = test_raw.groupby('layout_id')[FEAT_COLS].mean().reset_index()

# Normalize by training std
train_raw['oracle_pred'] = oracle_oof
tr_resid = (train_raw.groupby('layout_id')['avg_delay_minutes_next_30m'].mean()
          - train_raw.groupby('layout_id')['oracle_pred'].mean()).reset_index()
tr_resid.columns = ['layout_id','residual']
tr_ly = tr_ly.merge(tr_resid, on='layout_id')

feat_means = tr_ly[FEAT_COLS].mean()
feat_stds = tr_ly[FEAT_COLS].std().clip(lower=0.01)

tr_norm = ((tr_ly[FEAT_COLS] - feat_means) / feat_stds).values
te_norm = ((te_ly[FEAT_COLS] - feat_means) / feat_stds).values

tr_resid_vals = tr_ly['residual'].values
tr_ly_ids = tr_ly['layout_id'].values
te_ly_ids = te_ly['layout_id'].values

# Create layout_id → tr_ly index mapping
tr_ly_idx = {lid: i for i, lid in enumerate(tr_ly_ids)}

groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)

# Test different K values and alpha (correction strength)
print('\n--- KNN Correction Sweep ---')
print(f'{"K":>5}  {"alpha":>6}  {"OOF":>9}  {"delta":>8}  {"test_mean":>10}')

best_result = {'oof': mae(oracle_oof), 'K': None, 'alpha': None}

for K in [3, 5, 10, 20]:
    for alpha in [0.3, 0.5, 0.7, 1.0]:
        oof_corrected = oracle_oof.copy()

        for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
            vs = np.sort(val_idx)
            val_layouts = np.unique(train_raw.iloc[vs]['layout_id'].values)
            tr_layouts  = np.unique(train_raw.iloc[tr_idx]['layout_id'].values)

            # Training layout indices (only those in tr_idx)
            tr_available = [tr_ly_idx[lid] for lid in tr_layouts if lid in tr_ly_idx]

            for val_lid in val_layouts:
                if val_lid not in tr_ly_idx:
                    continue
                val_i = tr_ly_idx[val_lid]
                val_feat = tr_norm[val_i]  # features of this validation layout

                # Find K nearest training layouts
                dists = np.linalg.norm(tr_norm[tr_available] - val_feat, axis=1)
                k_actual = min(K, len(tr_available))
                knn_idx = np.argsort(dists)[:k_actual]
                knn_resid = np.mean(tr_resid_vals[[tr_available[i] for i in knn_idx]])

                # Apply correction to all rows of this validation layout
                val_mask = train_raw.iloc[vs]['layout_id'] == val_lid
                val_row_idx = vs[val_mask.values]
                oof_corrected[val_row_idx] = np.clip(oracle_oof[val_row_idx] + alpha * knn_resid, 0, None)

        oof_mae = mae(oof_corrected)
        marker = ' *' if oof_mae < best_result['oof'] else ''
        print(f'{K:>5}  {alpha:>6.2f}  {oof_mae:>9.5f}  {oof_mae-mae(oracle_oof):>+8.5f}  ---{marker}')
        if oof_mae < best_result['oof']:
            best_result = {'oof': oof_mae, 'K': K, 'alpha': alpha}

if best_result['K'] is None:
    print('\nNo improvement found with KNN correction on OOF.')
    # Still compute test corrections for analysis
    K, alpha = 5, 1.0
    all_tr_idx = list(range(len(tr_ly)))
    test_corrections = []
    for te_i, te_lid in enumerate(te_ly_ids):
        te_feat = te_norm[te_i]
        dists = np.linalg.norm(tr_norm - te_feat, axis=1)
        knn_idx = np.argsort(dists)[:K]
        knn_resid = np.mean(tr_resid_vals[knn_idx])
        test_corrections.append(knn_resid)
    tc = np.array(test_corrections)
    # Map back to rows
    te_lid_to_corr = {lid: tc[i] for i, lid in enumerate(te_ly_ids)}
    test_row_corrections = np.array([te_lid_to_corr.get(lid, 0) for lid in test_raw['layout_id'].values])
    test_corrected = np.clip(oracle_test + alpha * test_row_corrections, 0, None)
    print(f'\nTest correction analysis (K={K}, alpha={alpha}):')
    print(f'  test_mean={test_corrected.mean():.3f}  delta={test_corrected.mean()-oracle_test.mean():+.3f}')
    print('\n  High-pack test layout corrections:')
    for lid in ['WH_201', 'WH_246', 'WH_283']:
        te_i = np.where(te_ly_ids == lid)[0]
        if len(te_i) > 0:
            corr = tc[te_i[0]]
            pred = oracle_test[test_raw['layout_id']==lid].mean()
            print(f'    {lid}: pack={te_norm[te_i[0],1]*feat_stds["pack_utilization"]+feat_means["pack_utilization"]:.3f}  knn_resid={corr:+.2f}  oracle_pred={pred:.2f}  corrected={pred+alpha*corr:.2f}')
else:
    print(f'\nBest: K={best_result["K"]}, alpha={best_result["alpha"]}, OOF={best_result["oof"]:.5f}')
    # Generate test submission
    K, alpha = best_result['K'], best_result['alpha']
    all_tr_idx = list(range(len(tr_ly)))
    test_corrections = []
    for te_i, te_lid in enumerate(te_ly_ids):
        te_feat = te_norm[te_i]
        dists = np.linalg.norm(tr_norm - te_feat, axis=1)
        knn_idx = np.argsort(dists)[:K]
        knn_resid = np.mean(tr_resid_vals[knn_idx])
        test_corrections.append(knn_resid)
    tc = np.array(test_corrections)
    te_lid_to_corr = {lid: tc[i] for i, lid in enumerate(te_ly_ids)}
    test_row_corrections = np.array([te_lid_to_corr.get(lid, 0) for lid in test_raw['layout_id'].values])
    test_corrected = np.clip(oracle_test + alpha * test_row_corrections, 0, None)
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = test_corrected
    fname = f'submission_knn_K{K}_a{alpha}_OOF{best_result["oof"]:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'  test_mean={test_corrected.mean():.3f}  delta={test_corrected.mean()-oracle_test.mean():+.3f}')
    print(f'  Saved: {fname}')

print('\nDone.')
