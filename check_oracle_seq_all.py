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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

print(f"oracle_NEW reference: seen={oracle_new_t[seen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")

# Known good oracle models for reference
xgb_o=np.clip(np.load('results/oracle_seq/oof_seqC_xgb.npy'),0,None)
lv2_o=np.clip(np.load('results/oracle_seq/oof_seqC_log_v2.npy'),0,None)
rem_o=np.clip(np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'),0,None)
xgbc_o=np.clip(np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),0,None)
mono_o=np.clip(np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'),0,None)
xgb_t=np.clip(np.load('results/oracle_seq/test_C_xgb.npy'),0,None)
lv2_t=np.clip(np.load('results/oracle_seq/test_C_log_v2.npy'),0,None)
rem_t=np.clip(np.load('results/oracle_seq/test_C_xgb_remaining.npy'),0,None)
xgbc_t=np.clip(np.load('results/oracle_seq/test_C_xgb_combined.npy'),0,None)
mono_t=np.clip(np.load('results/oracle_seq/test_C_xgb_monotone.npy'),0,None)
oracle5_t = (xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5

print(f"oracle5_way: OOF={mae_fn((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5):.5f}  seen={oracle5_t[seen_mask].mean():.3f}  unseen={oracle5_t[unseen_mask].mean():.3f}")

print("\n" + "="*75)
print("ALL oracle_seq models — unexplored")
print("="*75)

unexplored_models = [
    'seqC_cb', 'seqC_extended_lead', 'seqC_extended_lead_v2',
    'seqC_global_rank', 'seqC_huber10', 'seqC_knn_scenario',
    'seqC_layout_proxy', 'seqC_lgb_dual', 'seqC_lgb_latepos',
    'seqC_lgb_log1', 'seqC_lgb_remaining', 'seqC_lgb_remaining_v3',
    'seqC_lgb_stack', 'seqC_log', 'seqC_pressure', 'seqC_ranklag',
    'seqC_raw', 'seqC_relload', 'seqC_robust_b', 'seqC_trajectory',
    'seqC_v2', 'seqC_v3', 'seqC_xgb_bestproxy', 'seqC_xgb_lag1',
    'seqC_xgb_sch_raw2', 'seqC_xgb_v31', 'seqC_xgb_v31_sc',
    'seqD_layout_stats', 'seqD_meta_minimal', 'seqD_meta_stacking',
]

good_models = []  # models with OOF < 9.0

for model in unexplored_models:
    oof_path = f'results/oracle_seq/oof_{model}.npy'
    test_path = f'results/oracle_seq/test_{model.replace("seqC","C").replace("seqD","D")}.npy'
    # Fix path
    test_path2 = f'results/oracle_seq/test_C_{model[5:]}.npy'
    test_pathD = f'results/oracle_seq/test_D_{model[5:]}.npy'

    try:
        oof_raw = np.load(oof_path)
        oof = np.clip(oof_raw, 0, None)
        if len(oof) != len(y_true):
            print(f"  {model:35s}: oof_shape={len(oof)} (unexpected)")
            continue
        oof_mae = mae_fn(oof)

        test_info = ""
        te = None
        for tp in [test_path2, test_pathD, test_path]:
            try:
                te_raw = np.load(tp)
                te = np.clip(te_raw, 0, None)
                if len(te) == len(id_order):
                    r_oN, _ = pearsonr(te, oracle_new_t)
                    test_info = f"  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r(oN)={r_oN:.4f}"
                    break
            except: continue

        print(f"  {model:35s}: OOF={oof_mae:.5f}{test_info}")
        if oof_mae < 9.0 and te is not None:
            good_models.append((model, oof, te, oof_mae))
    except Exception as e:
        print(f"  {model:35s}: ERR {str(e)[:50]}")

print()
print("="*75)
print("GOOD MODELS (OOF < 9.0) — detailed analysis")
print("="*75)

if not good_models:
    print("No models found with OOF < 9.0")
else:
    for model, oof, te, oof_mae in sorted(good_models, key=lambda x: x[3]):
        r_oN, _ = pearsonr(te, oracle_new_t)
        r_xgb, _ = pearsonr(oof, xgb_o)
        r_o5, _ = pearsonr(oof, (xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5)
        print(f"\n  {model}: OOF={oof_mae:.5f}  seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}")
        print(f"    r(xgb_oof)={r_xgb:.4f}  r(oracle5_oof)={r_o5:.4f}  r(oracle_NEW_test)={r_oN:.4f}")

        # Test blend with oracle_NEW
        print(f"    Blend oracle_NEW * (1-w) + {model} * w:")
        for w in [0.05, 0.10, 0.20]:
            blend = np.clip((1-w)*oracle_new_t + w*te, 0, None)
            print(f"      w={w}: seen={blend[seen_mask].mean():.3f}  unseen={blend[unseen_mask].mean():.3f}")

print()
print("="*75)
print("seqD models special focus (layout-aware)")
print("="*75)

for model in ['seqD_layout_stats', 'seqD_meta_minimal', 'seqD_meta_stacking']:
    oof_path = f'results/oracle_seq/oof_{model}.npy'
    test_path = f'results/oracle_seq/test_D_{model[5:]}.npy'
    try:
        oof_raw = np.load(oof_path)
        oof = np.clip(oof_raw, 0, None)
        print(f"\n  {model}: shape={len(oof)}")
        if len(oof) == len(y_true):
            mae = mae_fn(oof)
            print(f"    OOF={mae:.5f}  oof_mean={oof.mean():.3f}")
        te_raw = np.load(test_path)
        te = np.clip(te_raw, 0, None)
        print(f"    test shape={len(te)}")
        if len(te) == len(id_order):
            r, _ = pearsonr(te, oracle_new_t)
            print(f"    seen={te[seen_mask].mean():.3f}  unseen={te[unseen_mask].mean():.3f}  r(oN)={r:.4f}")
    except Exception as e:
        print(f"  {model}: {e}")

print("\nDone.")
