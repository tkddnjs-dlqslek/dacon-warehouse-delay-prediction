"""
Pack-Utilization Correction for oracle_NEW
Key finding: corr(ly_pack_mean, layout_oracle_residual) = +0.62
High pack_mean (>0.65) layouts have systematic oracle underprediction (+15~20).
Test has WH_201 (pack=0.936), WH_101 (pack=0.810), WH_004 (pack=0.818) — all extreme.

Approach:
1. For each fold: compute per-layout (pack_mean, oracle_residual_mean) on validation layouts
2. Fit OLS: residual = beta × (pack_mean - train_pack_mean) [centered, no intercept bias]
3. Validate OOF improvement via GroupKFold
4. Apply beta × (test_pack_mean - train_pack_mean) correction to oracle_NEW test predictions
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

# Compute layout-level pack_mean from ALL rows (pack_utilization is input, not target)
train_raw['ly_pack_mean'] = train_raw.groupby('layout_id')['pack_utilization'].transform('mean')
test_raw['ly_pack_mean']  = test_raw.groupby('layout_id')['pack_utilization'].transform('mean')

# Global training pack mean (used for centering)
train_pack_global = train_raw['ly_pack_mean'].mean()
print(f'Global training pack_mean: {train_pack_global:.4f}')

# Test pack stats
print(f'Test pack_mean stats: mean={test_raw["ly_pack_mean"].mean():.4f}  max={test_raw["ly_pack_mean"].max():.4f}')
print(f'Test layouts with ly_pack > train max ({train_raw["ly_pack_mean"].max():.4f}): {(test_raw.groupby("layout_id")["ly_pack_mean"].first() > train_raw["ly_pack_mean"].max()).sum()}')

# Layout-level pack
ly_pack = train_raw.groupby('layout_id')['pack_utilization'].mean()
te_ly_pack = test_raw.groupby('layout_id')['pack_utilization'].mean()

# GroupKFold cross-validation of pack correction
groups = train_raw['layout_id'].values
gkf    = GroupKFold(n_splits=5)

print('\n--- Pack Correction Sweep (centered, no intercept bias) ---')
print(f'{"alpha":>8}  {"OOF":>9}  {"delta":>8}  {"test_mean":>10}  {"test_delta":>10}')

for alpha in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 2.0]:
    oof_corrected = oracle_oof.copy()
    test_corrections = []

    for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        val_df = train_raw.iloc[vs]

        # For each validation layout: compute residual mean and pack mean
        val_with_pred = val_df.copy()
        val_with_pred['pred'] = oracle_oof[vs]
        val_with_pred['residual'] = y_true[vs] - oracle_oof[vs]

        ly_resid = val_with_pred.groupby('layout_id').agg(
            resid_mean=('residual','mean'),
            pack_mean=('pack_utilization','mean'),
            n=('residual','count')
        )

        # Fit beta (slope only, no intercept → centered correction)
        # residual = beta * (pack_mean - global_train_mean)
        # Centered so correction = 0 at mean pack level
        x = (ly_resid['pack_mean'] - train_pack_global).values
        y = ly_resid['resid_mean'].values
        # OLS: beta = cov(x,y)/var(x)
        beta = np.cov(x, y)[0,1] / (np.var(x) + 1e-8)

        # Apply correction to validation rows (row-level: add layout-level correction)
        val_pack_centered = val_df['ly_pack_mean'].values - train_pack_global
        correction = alpha * beta * val_pack_centered
        oof_corrected[vs] = np.clip(oracle_oof[vs] + correction, 0, None)

    # Test correction: use ALL training folds to estimate beta
    # Compute residuals on all training data (using full oracle_oof)
    train_raw_tmp = train_raw.copy()
    train_raw_tmp['pred'] = oracle_oof
    train_raw_tmp['residual'] = y_true - oracle_oof
    ly_resid_all = train_raw_tmp.groupby('layout_id').agg(
        resid_mean=('residual','mean'),
        pack_mean=('pack_utilization','mean')
    )
    x_all = (ly_resid_all['pack_mean'] - train_pack_global).values
    y_all = ly_resid_all['resid_mean'].values
    beta_all = np.cov(x_all, y_all)[0,1] / (np.var(x_all) + 1e-8)

    te_pack_centered = test_raw['ly_pack_mean'].values - train_pack_global
    test_correction = alpha * beta_all * te_pack_centered
    test_corrected = np.clip(oracle_test + test_correction, 0, None)

    oof_mae = mae(oof_corrected)
    print(f'{alpha:>8.2f}  {oof_mae:>9.5f}  {oof_mae-mae(oracle_oof):>+8.5f}  {test_corrected.mean():>10.3f}  {test_corrected.mean()-oracle_test.mean():>+10.3f}')

# Detailed analysis: fold-level pack correction
print('\n--- Fold-level analysis at alpha=0.5 and alpha=1.0 ---')
for alpha in [0.5, 1.0]:
    print(f'\nalpha={alpha}:')
    oof_corrected = oracle_oof.copy()
    for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        val_df = train_raw.iloc[vs]
        val_resid = y_true[vs] - oracle_oof[vs]
        val_packs = val_df.groupby('layout_id')['pack_utilization'].mean()
        val_resids = pd.Series(val_resid, index=vs).groupby(train_raw.iloc[vs]['layout_id']).mean()

        x = (val_packs.values - train_pack_global)
        y = val_resids.values
        beta = np.cov(x, y)[0,1] / (np.var(x) + 1e-8)

        val_pack_centered = val_df['ly_pack_mean'].values - train_pack_global
        correction = alpha * beta * val_pack_centered
        oof_corrected[vs] = np.clip(oracle_oof[vs] + correction, 0, None)

        fold_mae_orig = mean_absolute_error(y_true[vs], np.clip(oracle_oof[vs], 0, None))
        fold_mae_corr = mean_absolute_error(y_true[vs], oof_corrected[vs])
        print(f'  Fold {fi+1}: beta={beta:.2f}  orig={fold_mae_orig:.5f}  corrected={fold_mae_corr:.5f}  delta={fold_mae_corr-fold_mae_orig:+.5f}')
    print(f'  Overall OOF: {mae(oof_corrected):.5f}  delta={mae(oof_corrected)-mae(oracle_oof):+.5f}')

# What does beta look like per fold?
print('\n--- Per-fold beta estimates ---')
for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
    vs = np.sort(val_idx)
    val_df = train_raw.iloc[vs]
    val_packs = val_df.groupby('layout_id')['pack_utilization'].mean()
    val_resids = pd.Series(y_true[vs] - oracle_oof[vs]).groupby(train_raw.iloc[vs]['layout_id'].values).mean()
    x = (val_packs.values - train_pack_global)
    y = val_resids.values
    beta = np.cov(x, y)[0,1] / (np.var(x) + 1e-8)
    corr_fold = np.corrcoef(x, y)[0,1]
    print(f'  Fold {fi+1}: n_layouts={len(val_packs)}  beta={beta:.2f}  corr={corr_fold:.4f}  val_pack_mean={val_packs.mean():.4f}')

# Best correction: generate submission
print('\n--- Generate best submission ---')
# Find the alpha with best OOF
best_alpha = None
best_oof_val = mae(oracle_oof)
for alpha in np.arange(0.0, 2.1, 0.1):
    oof_corrected = oracle_oof.copy()
    for fi, (tr_idx, val_idx) in enumerate(gkf.split(np.arange(len(train_raw)), groups=groups)):
        vs = np.sort(val_idx)
        val_df = train_raw.iloc[vs]
        val_resid = y_true[vs] - oracle_oof[vs]
        val_packs = val_df.groupby('layout_id')['pack_utilization'].mean()
        val_resids_g = pd.Series(val_resid, index=vs).groupby(train_raw.iloc[vs]['layout_id']).mean()
        x = (val_packs.values - train_pack_global)
        y = val_resids_g.values
        beta = np.cov(x, y)[0,1] / (np.var(x) + 1e-8)
        val_pack_centered = val_df['ly_pack_mean'].values - train_pack_global
        oof_corrected[vs] = np.clip(oracle_oof[vs] + alpha * beta * val_pack_centered, 0, None)
    cur_oof = mae(oof_corrected)
    if cur_oof < best_oof_val:
        best_oof_val = cur_oof
        best_alpha = alpha

if best_alpha is not None:
    print(f'  Best alpha: {best_alpha:.1f}  OOF: {best_oof_val:.5f}  delta={best_oof_val-mae(oracle_oof):+.5f}')
    # Generate test predictions with best alpha
    train_raw_tmp2 = train_raw.copy()
    train_raw_tmp2['pred'] = oracle_oof
    train_raw_tmp2['residual'] = y_true - oracle_oof
    ly_r_all = train_raw_tmp2.groupby('layout_id').agg(
        resid_mean=('residual','mean'), pack_mean=('pack_utilization','mean'))
    x_all = (ly_r_all['pack_mean'] - train_pack_global).values
    y_all = ly_r_all['resid_mean'].values
    beta_all = np.cov(x_all, y_all)[0,1] / (np.var(x_all) + 1e-8)
    te_corr = best_alpha * beta_all * (test_raw['ly_pack_mean'].values - train_pack_global)
    test_corrected = np.clip(oracle_test + te_corr, 0, None)
    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = test_corrected
    fname = f'submission_pack_correction_a{best_alpha:.1f}_OOF{best_oof_val:.5f}.csv'
    sub.to_csv(fname, index=False)
    print(f'  test_mean={test_corrected.mean():.3f}  delta={test_corrected.mean()-oracle_test.mean():+.3f}')
    print(f'  Saved: {fname}')
    # Per-layout test correction
    test_raw['correction'] = te_corr
    te_ly_corr = test_raw.groupby('layout_id').agg(
        ly_pack_mean=('pack_utilization','mean'),
        correction_mean=('correction','mean')
    ).reset_index()
    print('\n  Test layouts with largest corrections:')
    for _, r in te_ly_corr.sort_values('correction_mean', ascending=False).head(10).iterrows():
        print(f"    {r['layout_id']}: pack={r['ly_pack_mean']:.3f}  correction={r['correction_mean']:+.2f}")
else:
    print('  No alpha improved OOF vs oracle_NEW')

print('\nDone.')
