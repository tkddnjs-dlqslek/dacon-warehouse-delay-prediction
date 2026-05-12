"""
Poisson regression objective for delay prediction.
delay time ≈ count-like (rare events, right-skewed)
Poisson loss may generalize differently from Huber.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = np.array([ls_pos[i] for i in train_raw['ID'].values])

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
id_order = test_raw['ID'].values
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

with open('results/mega33_final.pkl','rb') as f: d33=pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34=pickle.load(f)
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof
wm=fw['mega33']-dr2-dr3; w2_=fw['iter_r2']+dr2; w3_=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2_*r2_oof+w3_*r3_oof
w_rem2=1-wf; wxgb=0.12*w_rem2/0.36; wlv2=0.16*w_rem2/0.36; wrem2=0.08*w_rem2/0.36
bb_oo=np.clip(wf*fx_o+wxgb*xgb_o+wlv2*lv2_o+wrem2*rem_o,0,None)
bb_oo=np.clip((1-w_cb)*bb_oo+w_cb*cb_oof_mega,0,None)
fw4_oo=np.clip(0.74*bb_oo+0.08*slh_o+0.10*xgbc_o+0.08*mono_o,0,None)
oracle_mae=np.mean(np.abs(y_true-fw4_oo))
oracle_new_df=pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df=oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t=oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen=oracle_new_t[unseen_mask].mean()
print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

with open('results/eda_v30/v30_fe_cache.pkl','rb') as f: blob=pickle.load(f)
train_fe=blob['train_fe']; feat_cols=list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f: test_fe=pickle.load(f)
fold_ids=np.load('results/eda_v30/fold_idx.npy')
y=train_fe['avg_delay_minutes_next_30m'].values

X = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)

# Poisson은 양수 타겟 필요 — 0인 경우 1e-6으로 대체
y_pos = np.clip(y, 1e-6, None)

print(f"\n=== Poisson Regression LGB (5-fold) ===")
PARAMS_POISSON = dict(
    objective='poisson', max_delta_step=0.7,  # Poisson 안정화
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    verbose=-1, n_jobs=-1, random_state=42
)

oof_p = np.zeros(len(y))
test_p = np.zeros(len(X_te))

for f in range(5):
    vm=fold_ids==f; tm=~vm
    m=lgb.LGBMRegressor(**PARAMS_POISSON)
    m.fit(X[tm], y_pos[tm], eval_set=[(X[vm], y_pos[vm])],
          callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(0)])
    oof_p[vm] = np.clip(m.predict(X[vm]), 0, None)
    test_p += np.clip(m.predict(X_te), 0, None) / 5
    print(f"  fold {f}: MAE={np.mean(np.abs(y[vm]-oof_p[vm])):.4f}  it={m.best_iteration_}")

oof_p = np.clip(oof_p, 0, None)
oof_rid = oof_p[id2]
test_rid = test_p[te_rid_to_ls]

solo = np.mean(np.abs(y_true - oof_rid))
corr = np.corrcoef(fw4_oo, oof_rid)[0,1]
print(f"\nPoisson LGB OOF: {solo:.4f}  corr={corr:.4f}  (oracle={oracle_mae:.4f})")
print(f"test_unseen: {test_rid[unseen_mask].mean():.3f}  (oracle: {oracle_unseen:.3f})")

best_w, best_bl = 0, oracle_mae
for w in np.arange(0.01, 0.31, 0.01):
    bl=np.clip((1-w)*fw4_oo+w*oof_rid,0,None)
    mv=np.mean(np.abs(y_true-bl))
    if mv<best_bl: best_bl,best_w=mv,w
gain=oracle_mae-best_bl
print(f"Best blend: w={best_w:.2f}  blend_OOF={best_bl:.4f}  gain={gain:+.4f}")

if gain > 0.0003:
    bl_t=np.clip((1-best_w)*oracle_new_t+best_w*test_rid,0,None)
    sub=pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m']=bl_t
    fname=f"FINAL_poisson_OOF{best_bl:.4f}.csv"
    sub.to_csv(fname,index=False)
    np.save('results/poisson_oof.npy', oof_rid.astype(np.float32))
    np.save('results/poisson_test.npy', test_rid.astype(np.float32))
    print(f"*** SAVED: {fname}  unseen={bl_t[unseen_mask].mean():.3f} ***")
else:
    print("No blend improvement. No submission.")

print("\nDone.")
