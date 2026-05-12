import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

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
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

# oracle_NEW reference
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values
print(f"oracle_NEW: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Unexplored model files check")
print("="*70)

def check_model(path_oof, path_test=None, label='', align_id2=False, align_te_id2=False):
    try:
        oof_raw = np.load(path_oof)
        if align_id2: oof = np.clip(oof_raw[id2], 0, None)
        else: oof = np.clip(oof_raw, 0, None)
        oof_mae = mae_fn(oof) if len(oof) == len(y_true) else float('nan')

        test_str = ""
        if path_test:
            try:
                te_raw = np.load(path_test)
                if align_te_id2: te = np.clip(te_raw[te_id2], 0, None)
                else: te = np.clip(te_raw, 0, None)
                if len(te) == len(test_raw):
                    r_oracle, _ = __import__('scipy').stats.pearsonr(te, oracle_new_t)
                    test_str = f"  test={te.mean():.3f}  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r(oracle)={r_oracle:.4f}"
                else:
                    test_str = f"  test_shape={len(te)} (wrong shape)"
            except Exception as e:
                test_str = f"  test_err={str(e)[:50]}"

        print(f"  {label:40s}  OOF={oof_mae:.5f}{test_str}")
        return oof, te if path_test else None
    except Exception as e:
        print(f"  {label:40s}  ERROR: {str(e)[:60]}")
        return None, None

print()

# autogluon
check_model('results/autogluon/ag_oof.npy', 'results/autogluon/ag_test.npy', 'autogluon')

# approach models
check_model('results/approach_2_te/oof.npy', 'results/approach_2_te/test.npy', 'approach_2_te')
check_model('results/approach_3_sd/oof.npy', 'results/approach_3_sd/test.npy', 'approach_3_sd')
check_model('results/approach_4_anom/oof.npy', 'results/approach_4_anom/test.npy', 'approach_4_anom')

# vorth models
check_model('results/vorth/vorth_oof.npy', 'results/vorth/vorth_test.npy', 'vorth')
check_model('results/vorth_ranking/vorth_rank_oof.npy', 'results/vorth_ranking/vorth_rank_test.npy', 'vorth_ranking')

# bucket specialist
check_model('results/bucket_specialist/lgb_oof_b.npy', 'results/bucket_specialist/lgb_test_b.npy', 'bucket_spec_lgb')

# cascade additional
check_model('results/cascade/spec_avg_oof.npy', 'results/cascade/spec_avg_test.npy', 'spec_avg', align_id2=True, align_te_id2=True)
check_model('results/cascade/spec_v2_avg_oof.npy', 'results/cascade/spec_v2_avg_test.npy', 'spec_v2_avg', align_id2=True, align_te_id2=True)
check_model('results/cascade/spec_cb_raw_oof.npy', 'results/cascade/spec_cb_raw_test.npy', 'spec_cb_raw', align_id2=True, align_te_id2=True)
check_model('results/cascade/spec_cb_w30_oof.npy', 'results/cascade/spec_cb_w30_test.npy', 'spec_cb_w30', align_id2=True, align_te_id2=True)
check_model('results/cascade/spec_lgb_raw_mae_oof.npy', 'results/cascade/spec_lgb_raw_mae_test.npy', 'spec_lgb_raw_mae', align_id2=True, align_te_id2=True)

# cluster specialist
check_model('results/cluster_spec/cluster_oof.npy', 'results/cluster_spec/cluster_test.npy', 'cluster_spec')

# base_v31
check_model('results/base_v31/lgb_v31_oof.npy', 'results/base_v31/lgb_v31_test.npy', 'lgb_v31')
check_model('results/base_v31/xgb_v31_oof.npy', 'results/base_v31/xgb_v31_test.npy', 'xgb_v31')
check_model('results/base_v31/cb_v31_oof.npy', 'results/base_v31/cb_v31_test.npy', 'cb_v31')

# v24 quantiles (OOF only)
for q in ['Q30', 'Q50', 'Q70', 'Q90']:
    check_model(f'results/v24_cumsum/oof_{q}.npy', None, f'v24_{q}')

# v24 pseudo OOF experiments (OOF only)
for name in ['oof_pseudo_unseen_w0.5', 'oof_pseudo_unseen_w1.0', 'oof_pseudo_unseen_w2.0']:
    check_model(f'results/v24_cumsum/{name}.npy', None, f'v24_{name}')

print("\nDone.")
