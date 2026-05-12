"""Oracle-LGB-log-newcols: lv2 + 6 truly-new cols (lag1, sc_mean, sc_rank each). Kill > 8.87."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

KILL_THRESH = 8.87
NEW_COLS = ['task_reassign_15m','avg_recovery_time','replenishment_overlap',
            'staff_on_floor','forklift_active_count','order_wave_count']

print("Loading...", flush=True)
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
train_raw['row_in_sc'] = train_raw.groupby(['layout_id','scenario_id']).cumcount()
test_raw['row_in_sc']  = test_raw.groupby(['layout_id','scenario_id']).cumcount()

y_true = train_raw['avg_delay_minutes_next_30m'].values
y_log  = np.log1p(y_true)
global_mean_log = y_log.mean()

with open('results/eda_v30/v30_fe_cache.pkl','rb') as f: fe_tr=pickle.load(f)
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f: fe_te=pickle.load(f)
base_feat_cols = fe_tr['feat_cols']
fe_train_df = fe_tr['train_fe'].set_index('ID').loc[train_raw['ID'].values].reset_index()
fe_test_df  = pd.DataFrame(fe_te).set_index('ID').loc[test_raw['ID'].values].reset_index()

extra_feat_cols = []
for c in [x for x in NEW_COLS if x in fe_train_df.columns]:
    for df in [fe_train_df, fe_test_df]:
        g = df.groupby(['layout_id','scenario_id'])[c]
        df[f'{c}_lag1']    = g.shift(1).fillna(0)
        df[f'{c}_sc_mean'] = g.transform('mean')
        df[f'{c}_sc_rank'] = g.rank(pct=True)
    extra_feat_cols += [c, f'{c}_lag1', f'{c}_sc_mean', f'{c}_sc_rank']

feat_cols = list(base_feat_cols) + extra_feat_cols
print(f"v30: {len(base_feat_cols)} + new: {len(extra_feat_cols)} = {len(feat_cols)} total", flush=True)

with open('results/mega33_final.pkl','rb') as f: d=pickle.load(f)
train_ls=pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_to_pos={row['ID']:i for i,row in train_ls.iterrows()}
mega_oof_id=d['meta_avg_oof'][[ls_to_pos[i] for i in train_raw['ID'].values]]
test_ls=pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_to_pos={row['ID']:i for i,row in test_ls.iterrows()}
mega_test_id=d['meta_avg_test'][[te_ls_to_pos[i] for i in test_raw['ID'].values]]

train_raw['log_y']=y_log
grp=train_raw.groupby(['layout_id','scenario_id'])['log_y']
for k in [1,2,3]: train_raw[f'lag{k}_logy']=grp.shift(k).fillna(global_mean_log)
test_raw['log_mega33']=np.log1p(mega_test_id)
mg=test_raw.groupby(['layout_id','scenario_id'])['log_mega33']
for k in [1,2,3]: test_raw[f'lag{k}_logmega']=mg.shift(k).fillna(global_mean_log)

X_train_base=fe_train_df[feat_cols].fillna(0).values; row_sc_arr=train_raw['row_in_sc'].values
Xt=fe_test_df[[c for c in feat_cols if c in fe_test_df.columns]].copy()
for c in feat_cols:
    if c not in Xt.columns: Xt[c]=0.0
X_test_base=Xt[feat_cols].fillna(0).values

def make_X(b,l1,l2,l3,rs): return np.hstack([b,np.column_stack([l1,l2,l3,rs])])

LGB_PARAMS=dict(objective='regression_l1',n_estimators=3000,learning_rate=0.05,
    num_leaves=256,min_child_samples=20,subsample=0.8,colsample_bytree=0.8,
    reg_alpha=0.1,reg_lambda=0.1,n_jobs=4,random_state=42,verbose=-1)

gkf=GroupKFold(n_splits=5); groups=train_raw['layout_id'].values
oof=np.full(len(train_raw),np.nan); test_list=[]

print("Training oracle-LGB-newcols...", flush=True)
for fold_i,(tr_idx,val_idx) in enumerate(gkf.split(np.arange(len(train_raw)),groups=groups)):
    t0=time.time()
    X_tr=make_X(X_train_base[tr_idx],train_raw['lag1_logy'].values[tr_idx],
                train_raw['lag2_logy'].values[tr_idx],train_raw['lag3_logy'].values[tr_idx],row_sc_arr[tr_idx])
    X_val=make_X(X_train_base[val_idx],train_raw['lag1_logy'].values[val_idx],
                 train_raw['lag2_logy'].values[val_idx],train_raw['lag3_logy'].values[val_idx],row_sc_arr[val_idx])
    model=lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_tr,y_log[tr_idx],eval_set=[(X_val,y_log[val_idx])],
              callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(-1)])

    val_df=train_raw.iloc[val_idx][['layout_id','scenario_id','row_in_sc']].copy()
    val_df['_orig']=val_idx
    val_df=val_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    val_sorted=val_df['_orig'].values; rsc_vals=val_df['row_in_sc'].values
    log_mega_val=np.log1p(mega_oof_id[val_sorted])
    fold_pred=np.zeros(len(val_sorted))
    for pos in range(25):
        pm=rsc_vals==pos; pi=val_sorted[pm]; n=pm.sum()
        l1=np.full(n,global_mean_log) if pos==0 else log_mega_val[rsc_vals==(pos-1)]
        l2=np.full(n,global_mean_log) if pos<2  else log_mega_val[rsc_vals==(pos-2)]
        l3=np.full(n,global_mean_log) if pos<3  else log_mega_val[rsc_vals==(pos-3)]
        fold_pred[pm]=np.maximum(0,np.expm1(model.predict(make_X(X_train_base[pi],l1,l2,l3,np.full(n,pos)))))
    oof[val_sorted]=fold_pred

    te_df=test_raw[['layout_id','scenario_id','row_in_sc']].copy()
    te_df['_orig']=np.arange(len(test_raw))
    te_df=te_df.sort_values(['layout_id','scenario_id','row_in_sc'])
    te_sorted=te_df['_orig'].values; te_rsc=te_df['row_in_sc'].values
    log_mega_te=np.log1p(mega_test_id[te_sorted])
    test_pred=np.zeros(len(test_raw))
    for pos in range(25):
        pm=te_rsc==pos; pi=te_sorted[pm]; n=pm.sum()
        l1=np.full(n,global_mean_log) if pos==0 else log_mega_te[te_rsc==(pos-1)]
        l2=np.full(n,global_mean_log) if pos<2  else log_mega_te[te_rsc==(pos-2)]
        l3=np.full(n,global_mean_log) if pos<3  else log_mega_te[te_rsc==(pos-3)]
        test_pred[pi]=np.maximum(0,np.expm1(model.predict(make_X(X_test_base[pi],l1,l2,l3,np.full(n,pos)))))
    test_list.append(test_pred)

    mae=np.mean(np.abs(fold_pred-y_true[val_sorted]))
    print(f"Fold {fold_i+1}: oracle-LGB-newcols={mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    if fold_i==0 and mae>KILL_THRESH:
        print(f"*** fold1={mae:.4f} > {KILL_THRESH}: kill. ***", flush=True); import sys; sys.exit(1)

test_avg=np.mean(test_list,axis=0)
os.makedirs('results/oracle_seq',exist_ok=True)
np.save('results/oracle_seq/oof_seqC_lgb_newcols.npy',oof)
np.save('results/oracle_seq/test_C_lgb_newcols.npy',test_avg)
print("Saved.",flush=True)

train_ls2=pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls2={row['ID']:i for i,row in train_ls2.iterrows()}; id2=[ls2[i] for i in train_raw['ID'].values]
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,
        iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
fixed2=(fw['mega33']*d['meta_avg_oof'][id2]+fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
       +fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
       +fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
       +fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
xgb_v30=np.load('results/oracle_seq/oof_seqC_xgb.npy'); lv2_oof=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
fixed_mae=np.mean(np.abs(fixed2-y_true)); oof_mae=np.mean(np.abs(oof-y_true))
print(f"\noracle-LGB-newcols OOF: {oof_mae:.4f}\noracle-LGB-lv2    OOF: {np.mean(np.abs(lv2_oof-y_true)):.4f}\nFIXED: {fixed_mae:.4f}")
print(f"newcols corr lv2: {np.corrcoef(lv2_oof,oof)[0,1]:.4f}  xgb: {np.corrcoef(xgb_v30,oof)[0,1]:.4f}")
best_m=fixed_mae; best_w=0
for w in np.arange(0.02,0.51,0.02):
    mm=np.mean(np.abs((1-w)*fixed2+w*oof-y_true))
    if mm<best_m: best_m=mm; best_w=w
print(f"FIXED+newcols: w={best_w:.2f}  {best_m:.4f}  delta={best_m-fixed_mae:+.4f}")
base5=(1-0.12-0.20)*fixed2+0.12*xgb_v30+0.20*lv2_oof
best_m4=np.mean(np.abs(base5-y_true)); best_w4=0
for w in np.arange(0.02,0.21,0.02):
    mm=np.mean(np.abs((1-w)*base5+w*oof-y_true))
    if mm<best_m4: best_m4=mm; best_w4=w
print(f"base5+newcols:  w={best_w4:.2f}  {best_m4:.4f}  delta={best_m4-np.mean(np.abs(base5-y_true)):+.4f}")
print("Done.")
