"""
1) 새 피처 상호작용 탐색
2) 하이퍼파라미터 랜덤 서치 (많은 seed/config 조합)
목표: OOF < 8.55 AND corr < 0.93 인 조합 발견
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, itertools
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os, time
os.chdir("C:/Users/user/Desktop/데이콘 4월")

# ── oracle_NEW 재구성 ──────────────────────────────────────────────
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
xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw=dict(mega33=0.7636614598089654,rank_adj=0.1588758398901156,iter_r1=0.011855567572749024,iter_r2=0.034568307,iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
cb_oof_mega=np.clip(d33['meta_oofs']['cb'][id2],0,None)
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]
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

# ── v30 피처 로드 ────────────────────────────────────────────────
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f: blob=pickle.load(f)
train_fe=blob['train_fe']; feat_cols=list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f: test_fe=pickle.load(f)
fold_ids=np.load('results/eda_v30/fold_idx.npy')

y=train_fe['avg_delay_minutes_next_30m'].values; ylog=np.log1p(y)

# ── 새 피처 상호작용 추가 ────────────────────────────────────────
def add_interaction_feats(df):
    eps=0.01
    inf  = df['order_inflow_15m'].values
    pu   = df['pack_utilization'].values
    ru   = df['robot_utilization'].values if 'robot_utilization' in df.columns else np.zeros(len(df))
    cong = df['congestion_score'].values if 'congestion_score' in df.columns else np.zeros(len(df))
    conv = df['conveyor_speed_mps'].values if 'conveyor_speed_mps' in df.columns else np.ones(len(df))
    fault= df['fault_count_15m'].values if 'fault_count_15m' in df.columns else np.zeros(len(df))
    ts   = df['timeslot'].values if 'timeslot' in df.columns else np.zeros(len(df))

    return pd.DataFrame({
        'pu_x_ru':          pu * ru,
        'pu_x_cong':        pu * cong,
        'inf_x_cong':       inf * cong,
        'fault_x_pu':       fault * pu,
        'cong_per_inf':     cong / (inf + eps),
        'inf_per_conv':     inf / (conv + eps),
        'dual_bottleneck':  np.maximum(pu, ru),
        'load_x_fault':     inf * fault,
        'remain_x_inf_pu':  np.maximum(25-ts,0) * inf * pu,
        'pu_sq_inf':        pu**2 * inf,
    }, index=df.index)

ix_tr = add_interaction_feats(train_fe).values.astype(np.float32)
ix_te = add_interaction_feats(test_fe).values.astype(np.float32)
X_base = train_fe[feat_cols].values.astype(np.float32)
X_te_base = test_fe[feat_cols].values.astype(np.float32)
X_int = np.hstack([X_base, ix_tr])
X_te_int = np.hstack([X_te_base, ix_te])
print(f"Interaction feats added: {X_int.shape[1]} total (+{ix_tr.shape[1]})")

def quick_eval_2fold(X, y, ylog, fw4_oo, y_true, id2, folds=[0,4], params=None):
    """2개 fold만 빠르게 평가"""
    P = dict(objective='huber',alpha=0.9,n_estimators=2000,learning_rate=0.03,
             num_leaves=63,max_depth=8,min_child_samples=50,subsample=0.7,
             colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,verbose=-1,n_jobs=-1)
    if params: P.update(params)
    oof_partial = np.zeros(len(y))
    mask_used = np.zeros(len(y), dtype=bool)
    for f in folds:
        vm=fold_ids==f; tm=~vm
        m=lgb.LGBMRegressor(**P)
        m.fit(X[tm],ylog[tm],eval_set=[(X[vm],ylog[vm])],
              callbacks=[lgb.early_stopping(80,verbose=False),lgb.log_evaluation(0)])
        oof_partial[vm]=np.clip(np.expm1(m.predict(X[vm])),0,None)
        mask_used[vm]=True
    oof_rid=oof_partial[id2]; mask_rid=mask_used[id2]
    solo=np.mean(np.abs(y_true[mask_rid]-oof_rid[mask_rid]))
    corr=np.corrcoef(fw4_oo[mask_rid],oof_rid[mask_rid])[0,1]
    return solo, corr

# ── Section 1: 피처 상호작용 효과 확인 (2-fold 빠른 평가) ──────────
print("\n=== Section 1: Feature Interaction Quick Test (2-fold) ===")
solo_base, corr_base = quick_eval_2fold(X_base, y, ylog, fw4_oo, y_true, id2)
solo_int,  corr_int  = quick_eval_2fold(X_int,  y, ylog, fw4_oo, y_true, id2)
print(f"  v30 baseline:      OOF(2fold)={solo_base:.4f}  corr={corr_base:.4f}")
print(f"  v30 + interactions: OOF(2fold)={solo_int:.4f}  corr={corr_int:.4f}")
print(f"  OOF delta: {solo_int-solo_base:+.4f}  corr delta: {corr_int-corr_base:+.4f}")

# ── Section 2: 하이퍼파라미터 랜덤 서치 ───────────────────────────
print("\n=== Section 2: Hyperparameter Random Search (2-fold each) ===")
print(f"{'config':<55} {'OOF':>8} {'corr':>7} {'gain_pot':>9}")
print("-"*82)

rng = np.random.RandomState(42)
candidates = []

# 탐색 공간
param_grid = {
    'num_leaves':         [31, 47, 63, 95, 127],
    'max_depth':          [5, 6, 7, 8, -1],
    'min_child_samples':  [30, 50, 80, 120, 200],
    'colsample_bytree':   [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    'subsample':          [0.5, 0.6, 0.7, 0.8],
    'reg_alpha':          [0.1, 0.5, 1.0, 2.0, 4.0],
    'reg_lambda':         [0.1, 0.5, 1.0, 2.0, 4.0],
    'random_state':       list(range(30)),  # 다양한 seed
    'learning_rate':      [0.02, 0.03, 0.05],
}

N_SEARCH = 60  # 60개 랜덤 조합 탐색
best_found = []

for trial in range(N_SEARCH):
    p = {k: rng.choice(v) for k,v in param_grid.items()}
    # num_leaves와 max_depth 일관성
    if p['max_depth'] != -1 and 2**p['max_depth'] < p['num_leaves']:
        p['num_leaves'] = min(p['num_leaves'], 2**p['max_depth'])

    # 두 피처셋 모두 시도 (교대로)
    X_use = X_int if trial % 3 == 0 else X_base
    feat_tag = '+int' if trial % 3 == 0 else 'v30'

    try:
        solo, corr = quick_eval_2fold(X_use, y, ylog, fw4_oo, y_true, id2, params=p)

        # blend potential 추정 (corr과 OOF 기반)
        gain_pot = 0
        for w in [0.02, 0.05, 0.10]:
            bl = np.clip((1-w)*oracle_mae + w*solo, 0, None)
            # 다양성 보너스 근사 (corr 낮을수록 bonus)
            div_bonus = w * (1 - corr) * 0.5
            est_bl = bl - div_bonus
            if est_bl < oracle_mae: gain_pot = max(gain_pot, oracle_mae - est_bl)

        tag = f"{feat_tag} nl={p['num_leaves']} mc={p['min_child_samples']} col={p['colsample_bytree']:.1f} s={p['random_state']}"
        print(f"  {tag:<55} {solo:8.4f}  {corr:7.4f}  {gain_pot:+9.4f}")

        if solo < 8.70 or corr < 0.93:
            best_found.append((solo, corr, gain_pot, p, X_use, feat_tag))

    except Exception as e:
        pass

# ── Section 3: 유망 후보 Full 5-fold 검증 ─────────────────────────
print(f"\n=== Section 3: Promising candidates (OOF<8.70 or corr<0.93) → Full 5-fold ===")
if not best_found:
    print("  No candidates found meeting criteria.")
else:
    best_found.sort(key=lambda x: x[0])  # OOF 기준 정렬
    for rank, (solo2, corr2, gp, p, X_use, feat_tag) in enumerate(best_found[:5]):
        print(f"\n  Candidate {rank+1} ({feat_tag}): 2fold OOF={solo2:.4f} corr={corr2:.4f}")
        P_full = dict(objective='huber',alpha=0.9,n_estimators=2000,learning_rate=0.03,
                      num_leaves=63,max_depth=8,min_child_samples=50,subsample=0.7,
                      colsample_bytree=0.7,reg_alpha=1.0,reg_lambda=1.0,verbose=-1,n_jobs=-1)
        P_full.update(p)
        oof_f=np.zeros(len(y)); test_f=np.zeros(len(X_te_int))
        X_te_use = X_te_int if feat_tag=='+int' else X_te_base
        for f in range(5):
            vm=fold_ids==f; tm=~vm
            m=lgb.LGBMRegressor(**P_full)
            m.fit(X_use[tm],ylog[tm],eval_set=[(X_use[vm],ylog[vm])],
                  callbacks=[lgb.early_stopping(100,verbose=False),lgb.log_evaluation(0)])
            oof_f[vm]=np.clip(np.expm1(m.predict(X_use[vm])),0,None)
            test_f+=np.clip(np.expm1(m.predict(X_te_use)),0,None)/5
        oof_f=np.clip(oof_f,0,None)[id2]; test_f=test_f[te_rid_to_ls]
        solo_f=np.mean(np.abs(y_true-oof_f)); corr_f=np.corrcoef(fw4_oo,oof_f)[0,1]
        bw,bb=0,oracle_mae
        for w in np.arange(0.01,0.31,0.01):
            bl=np.clip((1-w)*fw4_oo+w*oof_f,0,None); mv=np.mean(np.abs(y_true-bl))
            if mv<bb: bb,bw=mv,w
        gain=oracle_mae-bb
        print(f"    Full 5-fold: OOF={solo_f:.4f}  corr={corr_f:.4f}  best_w={bw:.2f}  gain={gain:+.4f}")
        if gain > 0.0003:
            # 제출 파일 생성
            bl_t=np.clip((1-bw)*oracle_new_t+bw*test_f,0,None)
            sub=pd.read_csv('sample_submission.csv')
            sub['avg_delay_minutes_next_30m']=bl_t
            fname=f"FINAL_hpsearch_{feat_tag}_OOF{bb:.4f}.csv"
            sub.to_csv(fname,index=False)
            print(f"    *** SAVED: {fname}  unseen={bl_t[unseen_mask].mean():.3f} ***")
            np.save(f'results/hpsearch_{feat_tag}_oof.npy', oof_f)
            np.save(f'results/hpsearch_{feat_tag}_test.npy', test_f)

print("\nDone.")
