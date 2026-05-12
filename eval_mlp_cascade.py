"""
Try mlp_deep as cascade specialist on the neural blend base.
Also: explore triple gate with mlp_deep at high-confidence tier.
Neural base: mega37×0.4424 + rank×0.0747 + log_v2×0.1540 + xgb_combined×0.1406 + xgb_v31×0.0589 + mlp_deep×0.0506 + mlp_s2×0.0539 + mlp_gelu×0.0198
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, warnings
warnings.filterwarnings('ignore')

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

with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
with open('results/mega37_final.pkl','rb') as f: d37 = pickle.load(f)
with open('results/mlp_deep_final.pkl','rb') as f: dm = pickle.load(f)
with open('results/mlp_deep_s2_final.pkl','rb') as f: ds2 = pickle.load(f)
with open('results/mlp_deep_gelu_final.pkl','rb') as f: dg = pickle.load(f)

# Neural base (proven)
w_base = {'mega37':0.4424,'rank_adj':0.0747,'oracle_log_v2':0.1540,
          'oracle_xgb_combined':0.1406,'oracle_xgb_v31':0.0589,
          'mlp_deep':0.0506,'mlp_s2':0.0539,'mlp_gelu':0.0198}
total = sum(w_base.values()); w_base = {k:v/total for k,v in w_base.items()}

oof_c = {
    'mega37':  d37['meta_avg_oof'][id2],
    'mega34':  d34['meta_avg_oof'][id2],
    'rank_adj': np.load('results/ranking/rank_adj_oof.npy')[id2],
    'oracle_log_v2':       np.load('results/oracle_seq/oof_seqC_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/oof_seqC_xgb_v31.npy'),
    'mlp_deep': dm['oof'][id2], 'mlp_s2': ds2['oof'][id2], 'mlp_gelu': dg['oof'][id2],
}
test_c = {
    'mega37':  d37['meta_avg_test'][te_id2],
    'mega34':  d34['meta_avg_test'][te_id2],
    'rank_adj': np.load('results/ranking/rank_adj_test.npy')[te_id2],
    'oracle_log_v2':       np.load('results/oracle_seq/test_C_log_v2.npy'),
    'oracle_xgb_combined': np.load('results/oracle_seq/test_C_xgb_combined.npy'),
    'oracle_xgb_v31':      np.load('results/oracle_seq/test_C_xgb_v31.npy'),
    'mlp_deep': dm['test'][te_id2], 'mlp_s2': ds2['test'][te_id2], 'mlp_gelu': dg['test'][te_id2],
}

base_oof  = np.clip(sum(w_base.get(k,0)*oof_c[k]  for k in oof_c), 0, None)
base_test = np.clip(sum(w_base.get(k,0)*test_c[k] for k in test_c), 0, None)
base_mae  = np.mean(np.abs(base_oof - y_true))
print(f"Neural base OOF: {base_mae:.5f}")

# Cascade specialist options
clf_oof  = np.load('results/cascade/clf_oof.npy')[id2]
clf_test = np.load('results/cascade/clf_test.npy')[te_id2]

# Original specialists
rh_oof   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rm_oof   = np.load('results/cascade/spec_lgb_raw_mae_oof.npy')[id2]
rh_test  = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
rm_test  = np.load('results/cascade/spec_lgb_raw_mae_test.npy')[te_id2]

# mlp_deep as specialist (already in ls-sorted order with id2 applied)
mlp_deep_oof  = dm['oof'][id2]
mlp_deep_test = dm['test'][te_id2]
mlp_s2_oof    = ds2['oof'][id2]
mlp_s2_test   = ds2['test'][te_id2]

# Check specialist quality on high-delay rows
y_high = (y_true > 80).astype(int)
for name, spec in [('lgb_raw_huber', rh_oof), ('lgb_raw_mae', rm_oof),
                   ('mlp_deep', mlp_deep_oof), ('mlp_s2', mlp_s2_oof)]:
    mae_all  = np.mean(np.abs(spec - y_true))
    mae_high = np.mean(np.abs(spec[y_high==1] - y_true[y_high==1]))
    print(f"  {name:20s}: all_mae={mae_all:.3f}  high_mae={mae_high:.3f}")

prev_best = 8.36787
best_mae = prev_best
best_oof_p = None; best_test_p = None; best_cfg = None

# Search: mlp_deep as first-stage specialist
print(f"\n[Search1] mlp_deep as 1st stage specialist", flush=True)
for p1 in np.arange(0.08, 0.16, 0.01):
    m1    = (clf_oof  > p1).astype(float); m1_te = (clf_test > p1).astype(float)
    for w1 in np.arange(0.010, 0.080, 0.005):
        b1    = (1-m1*w1)*base_oof  + m1*w1*mlp_deep_oof
        b1_te = (1-m1_te*w1)*base_test + m1_te*w1*mlp_deep_test
        for p2 in np.arange(0.18, 0.36, 0.01):
            if p2 <= p1: continue
            m2    = (clf_oof  > p2).astype(float); m2_te = (clf_test > p2).astype(float)
            for w2 in np.arange(0.010, 0.080, 0.005):
                blend = (1-m2*w2)*b1 + m2*w2*rm_oof
                mm = np.mean(np.abs(blend - y_true))
                if mm < best_mae:
                    best_mae = mm
                    best_cfg = f'mlp_deep1st+p{p1:.2f}w{w1:.3f}+p{p2:.2f}w{w2:.3f}'
                    best_oof_p = blend
                    best_test_p = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                    print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

# Search2: triple gate (lgb_huber, lgb_mae, mlp_deep at very high confidence)
print(f"\n[Search2] Triple gate with mlp_deep at top tier", flush=True)
for p1 in [0.09, 0.10, 0.11, 0.12]:
    m1 = (clf_oof>p1).astype(float); m1_te = (clf_test>p1).astype(float)
    for w1 in np.arange(0.010, 0.045, 0.005):
        b1 = (1-m1*w1)*base_oof + m1*w1*rh_oof
        b1_te = (1-m1_te*w1)*base_test + m1_te*w1*rh_test
        for p2 in [0.22, 0.25, 0.27, 0.30]:
            if p2 <= p1: continue
            m2 = (clf_oof>p2).astype(float); m2_te = (clf_test>p2).astype(float)
            for w2 in np.arange(0.010, 0.045, 0.005):
                b2 = (1-m2*w2)*b1 + m2*w2*rm_oof
                b2_te = (1-m2_te*w2)*b1_te + m2_te*w2*rm_test
                for p3 in [0.40, 0.45, 0.50, 0.55, 0.60]:
                    if p3 <= p2: continue
                    m3 = (clf_oof>p3).astype(float); m3_te = (clf_test>p3).astype(float)
                    for w3 in np.arange(0.020, 0.100, 0.010):
                        blend = (1-m3*w3)*b2 + m3*w3*mlp_deep_oof
                        mm = np.mean(np.abs(blend - y_true))
                        if mm < best_mae:
                            best_mae = mm
                            best_cfg = f'triple_p{p1}w{w1:.3f}+p{p2}w{w2:.3f}+p{p3}w{w3:.3f}'
                            best_oof_p = blend
                            best_test_p = (1-m3_te*w3)*b2_te + m3_te*w3*mlp_deep_test
                            print(f"★ {best_cfg}  OOF={mm:.5f}", flush=True)

print(f"\n=== FINAL ===")
print(f"  Prev: {prev_best:.5f}  Best: {best_mae:.5f}  ({best_cfg})")

ref_prev = 8.37851
if best_mae < ref_prev - 0.00005:
    sample_sub = pd.read_csv('sample_submission.csv')
    sub = np.maximum(0, best_oof_p if best_oof_p is None else best_test_p)
    sub = np.maximum(0, best_test_p)
    sub_df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    sub_df = sub_df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    fname = f'submission_mlp_cascade_OOF{best_mae:.5f}.csv'
    sub_df.to_csv(fname, index=False)
    print(f"*** SAVED: {fname} ***")
else:
    print(f"No improvement (best={best_mae:.5f})")
print("Done.")
