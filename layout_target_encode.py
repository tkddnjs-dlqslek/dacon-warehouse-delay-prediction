"""
Layout-level Target Encoding (CV-safe)
- train fold에서만 layout 평균 delay 계산 → val/test에 적용
- unseen test layout → global mean (스무딩)
- 아이디어: layout 난이도 정보를 피처로 제공
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os; os.chdir("C:/Users/user/Desktop/데이콘 4월")

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

# ── v30 피처 로드 ────────────────────────────────────────────────
with open('results/eda_v30/v30_fe_cache.pkl','rb') as f: blob=pickle.load(f)
train_fe=blob['train_fe']; feat_cols=list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl','rb') as f: test_fe=pickle.load(f)
fold_ids=np.load('results/eda_v30/fold_idx.npy')

y=train_fe['avg_delay_minutes_next_30m'].values; ylog=np.log1p(y)
print(f"Columns in train_fe: {list(train_fe.columns[:10])} ... ({len(train_fe.columns)} total)")

# layout_id 확인
if 'layout_id' not in train_fe.columns:
    print("ERROR: layout_id not found in train_fe — cannot do target encoding")
    import sys; sys.exit(1)

# 레이아웃 목록
layouts_train = train_fe['layout_id'].values
layouts_test  = test_fe['layout_id'].values if 'layout_id' in test_fe.columns else None
print(f"Train layouts: {len(np.unique(layouts_train))}  Test layouts: {len(np.unique(layouts_test)) if layouts_test is not None else 'N/A'}")

global_mean = y.mean()
global_std  = y.std()
print(f"Global target: mean={global_mean:.3f}  std={global_std:.3f}")

# ── CV-safe Target Encoding ────────────────────────────────────────
# 각 fold에서: 훈련 데이터로 layout 통계 계산, validation에 적용
# 스무딩: (n * layout_mean + k * global_mean) / (n + k), k=smoothing factor
SMOOTH = 50  # 레이아웃당 샘플이 적으면 global mean 쪽으로 당김

def smooth_encode(target_vals, global_val, smooth=50):
    n = len(target_vals)
    if n == 0:
        return global_val, global_val
    layout_mean = target_vals.mean()
    layout_std  = target_vals.std() if n > 1 else 0.0
    w = n / (n + smooth)
    enc_mean = w * layout_mean + (1 - w) * global_val
    enc_std  = layout_std  # std는 스무딩 없이
    return enc_mean, enc_std

def compute_target_encoding_cv(train_fe, y, fold_ids, global_mean, smooth=50):
    """CV-safe: 각 fold에서 훈련 데이터만 사용하여 encoding 계산"""
    enc_mean = np.full(len(y), global_mean, dtype=np.float32)
    enc_std  = np.full(len(y), global_mean * 0.5, dtype=np.float32)
    enc_n    = np.zeros(len(y), dtype=np.float32)

    for f in range(5):
        vm = fold_ids == f
        tm = ~vm
        tr_layouts = train_fe['layout_id'].values[tm]
        tr_y = y[tm]
        val_layouts = train_fe['layout_id'].values[vm]

        # 훈련 데이터에서 layout 통계 계산
        layout_stats = {}
        for lid in np.unique(tr_layouts):
            mask_l = tr_layouts == lid
            layout_stats[lid] = smooth_encode(tr_y[mask_l], global_mean, smooth)

        # validation fold에 적용
        for i, lid in enumerate(val_layouts):
            idx = np.where(vm)[0][i]
            if lid in layout_stats:
                enc_mean[idx], enc_std[idx] = layout_stats[lid]
                enc_n[idx] = np.sum(tr_layouts == lid)
            # else: keep global mean (layout not in training fold)

    return enc_mean, enc_std, enc_n

enc_mean, enc_std, enc_n = compute_target_encoding_cv(train_fe, y, fold_ids, global_mean, SMOOTH)
print(f"\nTarget encoding stats (train):")
print(f"  enc_mean: {enc_mean.mean():.3f} ± {enc_mean.std():.3f}  range=[{enc_mean.min():.3f}, {enc_mean.max():.3f}]")
print(f"  enc_std:  {enc_std.mean():.3f} ± {enc_std.std():.3f}")

# 테스트용 encoding: 전체 train 데이터 사용
te_enc_mean = np.full(len(test_fe), global_mean, dtype=np.float32)
te_enc_std  = np.zeros(len(test_fe), dtype=np.float32)
te_enc_n    = np.zeros(len(test_fe), dtype=np.float32)

layout_stats_full = {}
for lid in np.unique(layouts_train):
    mask_l = layouts_train == lid
    layout_stats_full[lid] = smooth_encode(y[mask_l], global_mean, SMOOTH)

if layouts_test is not None:
    for i, lid in enumerate(layouts_test):
        if lid in layout_stats_full:
            te_enc_mean[i], te_enc_std[i] = layout_stats_full[lid]
            te_enc_n[i] = np.sum(layouts_train == lid)
        # else: unseen → global_mean (기본값 유지)
    unseen_te = ~np.isin(layouts_test, list(layout_stats_full.keys()))
    print(f"Test unseen layouts: {np.sum(unseen_te)} rows → global mean applied")

# ── Feature 행렬 구성 ─────────────────────────────────────────────
X_base = train_fe[feat_cols].values.astype(np.float32)
X_te_base = test_fe[feat_cols].values.astype(np.float32)
te_extra = np.stack([te_enc_mean, te_enc_std, te_enc_n], axis=1).astype(np.float32)
tr_extra = np.stack([enc_mean, enc_std, enc_n], axis=1).astype(np.float32)

X_te_full = np.hstack([X_te_base, te_extra])
# NOTE: tr_extra는 fold 별로 다르므로 full X 미리 만들어도 OK (CV loop에서 교체)
print(f"\nFeature shape: {X_base.shape[1]} base + 3 target_enc = {X_base.shape[1]+3} total")

# ── 2-fold 빠른 비교 ──────────────────────────────────────────────
PARAMS = dict(objective='huber', alpha=0.9, n_estimators=2000, learning_rate=0.03,
              num_leaves=63, max_depth=8, min_child_samples=50,
              subsample=0.7, colsample_bytree=0.7,
              reg_alpha=1.0, reg_lambda=1.0, verbose=-1, n_jobs=-1, random_state=42)

print("\n=== 2-fold Quick Comparison ===")
def run_2fold_te(use_te, label):
    oof_p=np.zeros(len(y)); used=np.zeros(len(y),dtype=bool)
    for f in [0,4]:
        vm=fold_ids==f; tm=~vm
        if use_te:
            Xtr_f = np.hstack([X_base[tm], tr_extra[tm]])
            Xvl_f = np.hstack([X_base[vm], tr_extra[vm]])
        else:
            Xtr_f = X_base[tm]; Xvl_f = X_base[vm]
        m=lgb.LGBMRegressor(**PARAMS)
        m.fit(Xtr_f, ylog[tm], eval_set=[(Xvl_f, ylog[vm])],
              callbacks=[lgb.early_stopping(80,verbose=False), lgb.log_evaluation(0)])
        oof_p[vm]=np.clip(np.expm1(m.predict(Xvl_f)),0,None)
        used[vm]=True
    rid=oof_p[id2]; mask=used[id2]
    solo=np.mean(np.abs(y_true[mask]-rid[mask]))
    corr=np.corrcoef(fw4_oo[mask], rid[mask])[0,1]
    print(f"  {label:<30} OOF(2fold)={solo:.4f}  corr={corr:.4f}")
    return solo, corr

s0, c0 = run_2fold_te(False, "v30 baseline")
s1, c1 = run_2fold_te(True,  "v30 + target_enc(3)")
print(f"  delta OOF: {s1-s0:+.4f}  delta corr: {c1-c0:+.4f}")

# ── Full 5-fold ───────────────────────────────────────────────────
print("\n=== Full 5-fold: v30 + target_encoding ===")
oof_full = np.zeros(len(y))
test_full = np.zeros(len(X_te_full))
for f in range(5):
    vm=fold_ids==f; tm=~vm
    Xtr_f = np.hstack([X_base[tm], tr_extra[tm]])
    Xvl_f = np.hstack([X_base[vm], tr_extra[vm]])
    m=lgb.LGBMRegressor(**PARAMS)
    m.fit(Xtr_f, ylog[tm], eval_set=[(Xvl_f, ylog[vm])],
          callbacks=[lgb.early_stopping(100,verbose=False), lgb.log_evaluation(0)])
    oof_full[vm]=np.clip(np.expm1(m.predict(Xvl_f)),0,None)
    test_full += np.clip(np.expm1(m.predict(X_te_full)),0,None)/5
    print(f"  fold {f}: MAE={np.mean(np.abs(y[vm]-oof_full[vm])):.4f}  it={m.best_iteration_}")

oof_full=np.clip(oof_full,0,None)
oof_rid=oof_full[id2]; test_rid=test_full[te_rid_to_ls]
solo_f=np.mean(np.abs(y_true-oof_rid)); corr_f=np.corrcoef(fw4_oo,oof_rid)[0,1]
print(f"\nTE-LGB OOF (rid): {solo_f:.4f}  corr={corr_f:.4f}  (oracle={oracle_mae:.4f})")
print(f"test_unseen: {test_rid[unseen_mask].mean():.3f}  test_seen: {test_rid[~unseen_mask].mean():.3f}")
print(f"oracle_unseen: {oracle_unseen:.3f}")

# ── Blend ─────────────────────────────────────────────────────────
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
    fname=f"FINAL_tenc_OOF{best_bl:.4f}.csv"
    sub.to_csv(fname,index=False)
    np.save('results/tenc_oof.npy', oof_rid.astype(np.float32))
    np.save('results/tenc_test.npy', test_rid.astype(np.float32))
    print(f"*** SAVED: {fname}  unseen={bl_t[unseen_mask].mean():.3f} ***")
else:
    print("No blend improvement. No submission.")

print("\nDone.")
