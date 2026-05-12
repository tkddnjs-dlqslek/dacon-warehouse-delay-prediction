import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, os
from scipy.stats import pearsonr

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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

def check_model(path_oof, path_test=None, label='', id2_=None, te_id2_=None):
    try:
        oof_raw = np.load(path_oof)
        if id2_ is not None and len(oof_raw) == len(id2_):
            oof = np.clip(oof_raw[id2_], 0, None)
        else:
            oof = np.clip(oof_raw, 0, None)
        oof_mae = mae_fn(oof) if len(oof) == len(y_true) else float('nan')

        test_info = ""
        te = None
        if path_test:
            try:
                te_raw = np.load(path_test)
                if te_id2_ is not None and len(te_raw) == len(te_id2_):
                    te = np.clip(te_raw[te_id2_], 0, None)
                else:
                    te = np.clip(te_raw, 0, None)
                if len(te) == len(test_raw):
                    r_oracle, _ = pearsonr(te, oracle_new_t)
                    test_info = f"  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r(oN)={r_oracle:.4f}"
                else:
                    test_info = f"  test_shape={len(te)}"
            except Exception as e:
                test_info = f"  te_err={str(e)[:40]}"

        print(f"  {label:40s}  OOF={oof_mae:.5f}{test_info}")
        return oof, te
    except Exception as e:
        print(f"  {label:40s}  ERR: {str(e)[:50]}")
        return None, None

print("="*70)
print("UNEXPLORED MODEL FILES v2")
print("="*70)

print("\n--- iter_pseudo round4 (MISSING from oracle_NEW!) ---")
r4_o, r4_t = check_model('results/iter_pseudo/round4_oof.npy', 'results/iter_pseudo/round4_test.npy', 'iter_pseudo_round4', id2_=id2, te_id2_=te_id2)

print("\n--- lag_target ---")
check_model('results/lag_target/lag_oof.npy', 'results/lag_target/lag_test.npy', 'lag_target')

print("\n--- independent ---")
check_model('results/independent/indep_oof.npy', 'results/independent/indep_test.npy', 'independent')

print("\n--- cv_experiment ---")
check_model('results/cv_experiment/group_oof.npy', 'results/cv_experiment/group_test.npy', 'cv_group')
check_model('results/cv_experiment/row_oof.npy', 'results/cv_experiment/row_test.npy', 'cv_row')
check_model('results/cv_experiment/full_test.npy', None, 'cv_full_test_only')

print("\n--- framework_g ---")
check_model('results/framework_g/pred_A_oof.npy', 'results/framework_g/pred_A_test.npy', 'framework_pred_A')
check_model('results/framework_g/meta_B_oof.npy', 'results/framework_g/meta_B_test.npy', 'framework_meta_B')
check_model('results/framework_g/extreme_oof.npy', None, 'framework_extreme_oof')
check_model('results/framework_g/hurdle_oof.npy', None, 'framework_hurdle_oof')
check_model('results/framework_g/magn_oof.npy', None, 'framework_magn_oof')

print("\n--- expanded_features ---")
check_model('results/expanded_features/exp_oof.npy', 'results/expanded_features/exp_test.npy', 'expanded_features')

print("\n--- feat_reinterp ---")
check_model('results/feat_reinterp/option_A_oof.npy', 'results/feat_reinterp/option_A_test.npy', 'feat_reinterp_A')
check_model('results/feat_reinterp/option_B_oof.npy', 'results/feat_reinterp/option_B_test.npy', 'feat_reinterp_B')

print("\n--- eda_v30 ---")
check_model('results/eda_v30/extreme_oof.npy', None, 'eda_v30_extreme')
check_model('results/eda_v30/hubspoke_oof.npy', None, 'eda_v30_hubspoke')

print("\n--- eda_v31 ---")
check_model('results/eda_v31/v31_lgb_oof.npy', 'results/eda_v31/v31_lgb_test.npy', 'eda_v31_lgb')

print("\n--- v16_ckpt ---")
for fname in ['log_LGB_MAE_s42', 'log_XGB_s42', 'raw_CatBoost_s42', 'raw_LGB_Huber_s42', 'raw_LGB_MAE_s42', 'raw_XGB_s42']:
    check_model(f'results/v16_ckpt/{fname}_oof.npy', f'results/v16_ckpt/{fname}_test.npy', f'v16_{fname}')

print("\n--- v24_cumsum key models ---")
check_model('results/v24_cumsum/oof_A_v23feats.npy', None, 'v24_A_v23feats')
check_model('results/v24_cumsum/oof_B_v23feats_plus_cumsum.npy', None, 'v24_B_cumsum')

print("\n--- zero_cls ---")
check_model('results/zero_cls/p_zero_oof.npy', 'results/zero_cls/p_zero_test.npy', 'zero_cls')

print()
print("="*70)
print("ITER_PSEUDO ROUND4: deeper check if promising")
print("="*70)

if r4_o is not None:
    print(f"\nround4_oof OOF MAE: {mae_fn(r4_o):.5f}")
    print(f"round4_oof shape: {np.load('results/iter_pseudo/round4_oof.npy').shape}")

    # Compare with round1/2/3
    r1_o=np.load('results/iter_pseudo/round1_oof.npy')[id2]
    r2_o=np.load('results/iter_pseudo/round2_oof.npy')[id2]
    r3_o=np.load('results/iter_pseudo/round3_oof.npy')[id2]
    r4_raw = np.load('results/iter_pseudo/round4_oof.npy')
    r4_o_direct = np.clip(r4_raw, 0, None)

    for name, arr in [('round1', r1_o), ('round2', r2_o), ('round3', r3_o)]:
        print(f"  r({name}, round4) = {pearsonr(arr, r4_o if len(r4_o)==len(arr) else r4_o_direct)[0]:.4f}")

    if r4_t is not None:
        r1_t=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
        r2_t=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
        r3_t=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
        print(f"\nround4 test: seen={r4_t[seen_mask].mean():.3f}  unseen={r4_t[unseen_mask].mean():.3f}")
        for name, arr in [('round1', r1_t), ('round2', r2_t), ('round3', r3_t), ('oracle_NEW', oracle_new_t)]:
            r, _ = pearsonr(r4_t, arr)
            print(f"  r(round4_t, {name}) = {r:.4f}")

print("\nDone.")
