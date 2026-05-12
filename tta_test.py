"""
Test-Time Augmentation for tabular GBDT.
- Train v30 LGB model (5-fold GroupKFold)
- At inference: add small Gaussian noise to features K times, take median
- Compare TTA OOF vs vanilla OOF
- Check if TTA predictions blend with oracle_NEW
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import lightgbm as lgb
import warnings; warnings.filterwarnings('ignore')
import os, time
os.chdir("C:/Users/user/Desktop/데이콘 4월")

# ── 기본 데이터 로드 ──────────────────────────────────────────────
train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])

# ── oracle_NEW OOF 재구성 ─────────────────────────────────────────
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values
oracle_unseen = oracle_new_t[unseen_mask].mean()

print(f"oracle_NEW: OOF={oracle_mae:.4f}  unseen={oracle_unseen:.3f}")

# ── v30 피처 로드 ─────────────────────────────────────────────────
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    blob = pickle.load(f)
train_fe = blob['train_fe']
feat_cols = list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    test_fe = pickle.load(f)
fold_ids = np.load('results/eda_v30/fold_idx.npy')

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
y = train_fe['avg_delay_minutes_next_30m'].values.astype(np.float64)
y_log = np.log1p(y)

# 피처별 std (TTA noise scale용)
feat_stds = X_tr.std(axis=0).astype(np.float32)
feat_stds = np.maximum(feat_stds, 1e-6)

print(f"Feature shape: {X_tr.shape}, feat_stds mean={feat_stds.mean():.4f}")

PARAMS = dict(
    objective="huber", alpha=0.9, n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
)

# ── TTA 함수 ─────────────────────────────────────────────────────
def predict_tta(model, X, feat_stds, K=20, sigma=0.005, seed=0):
    """K번 노이즈 추가 → 예측 → median"""
    rng = np.random.RandomState(seed)
    preds = []
    # vanilla (no noise) 포함
    preds.append(np.clip(np.expm1(model.predict(X)), 0, None))
    for k in range(K - 1):
        noise = rng.randn(*X.shape).astype(np.float32) * (sigma * feat_stds)
        X_noisy = np.clip(X + noise, 0, None)
        preds.append(np.clip(np.expm1(model.predict(X_noisy)), 0, None))
    return np.median(preds, axis=0)

# ── 5-fold 학습 + TTA OOF ─────────────────────────────────────────
print("\n=== Training v30 LGB + TTA ===")
oof_vanilla = np.zeros(len(y))
oof_tta = np.zeros(len(y))
test_vanilla = np.zeros(len(X_te))
test_tta = np.zeros(len(X_te))

for f in range(5):
    t0 = time.time()
    val_mask = fold_ids == f
    tr_mask = ~val_mask

    m = lgb.LGBMRegressor(**PARAMS)
    m.fit(X_tr[tr_mask], y_log[tr_mask],
          eval_set=[(X_tr[val_mask], y_log[val_mask])],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])

    # vanilla predict
    oof_vanilla[val_mask] = np.clip(np.expm1(m.predict(X_tr[val_mask])), 0, None)
    test_vanilla += np.clip(np.expm1(m.predict(X_te)), 0, None) / 5

    # TTA predict
    oof_tta[val_mask] = predict_tta(m, X_tr[val_mask], feat_stds, K=20, sigma=0.005, seed=f)
    test_tta += predict_tta(m, X_te, feat_stds, K=20, sigma=0.005, seed=f+10) / 5

    van_mae = np.mean(np.abs(y[val_mask] - oof_vanilla[val_mask]))
    tta_mae = np.mean(np.abs(y[val_mask] - oof_tta[val_mask]))
    print(f"  fold {f}: vanilla={van_mae:.4f}  TTA={tta_mae:.4f}  Δ={tta_mae-van_mae:+.4f}  it={m.best_iteration_} ({time.time()-t0:.0f}s)")

oof_vanilla = np.clip(oof_vanilla, 0, None)
oof_tta = np.clip(oof_tta, 0, None)

# ls → rid 변환
oof_van_rid = oof_vanilla[id2]
oof_tta_rid = oof_tta[id2]
test_van_rid = test_vanilla[te_rid_to_ls]
test_tta_rid = test_tta[te_rid_to_ls]

van_mae_rid = np.mean(np.abs(y_true - oof_van_rid))
tta_mae_rid = np.mean(np.abs(y_true - oof_tta_rid))
corr_van = np.corrcoef(fw4_oo, oof_van_rid)[0,1]
corr_tta = np.corrcoef(fw4_oo, oof_tta_rid)[0,1]

print(f"\nVanilla OOF (rid): {van_mae_rid:.4f}  corr={corr_van:.4f}")
print(f"TTA    OOF (rid): {tta_mae_rid:.4f}  corr={corr_tta:.4f}")
print(f"TTA gain over vanilla: {van_mae_rid - tta_mae_rid:+.4f}")
print(f"test_unseen vanilla: {test_van_rid[unseen_mask].mean():.3f}")
print(f"test_unseen TTA:     {test_tta_rid[unseen_mask].mean():.3f}")

# ── TTA가 oracle_NEW와 blend 가능한지 확인 ───────────────────────
print(f"\n=== Blend TTA with oracle_NEW ===")
print(f"{'model':<12} {'solo':>8} {'corr':>7} {'best_w':>7} {'blend':>10} {'gain':>8}")
print("-"*60)

for name, o, t in [("vanilla", oof_van_rid, test_van_rid), ("TTA", oof_tta_rid, test_tta_rid)]:
    solo = np.mean(np.abs(y_true - o))
    corr = np.corrcoef(fw4_oo, o)[0,1]
    best_w, best_bl = 0, oracle_mae
    for w in np.arange(0.01, 0.51, 0.01):
        bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
        m = np.mean(np.abs(y_true - bl))
        if m < best_bl: best_bl, best_w = m, w
    gain = oracle_mae - best_bl
    print(f"  {name:<10} {solo:8.4f}  {corr:7.4f}  {best_w:7.2f}  {best_bl:10.4f}  {gain:+8.4f}")

# ── sigma sweep: TTA 최적 노이즈 수준 탐색 ──────────────────────
print(f"\n=== TTA sigma sweep (last fold model) ===")
# 마지막 fold 모델 재사용 (proxy)
val_mask4 = fold_ids == 4
for sigma in [0.001, 0.003, 0.005, 0.01, 0.02, 0.05]:
    p = predict_tta(m, X_tr[val_mask4], feat_stds, K=30, sigma=sigma, seed=99)
    mae = np.mean(np.abs(y[val_mask4] - p))
    print(f"  sigma={sigma:.3f}: fold4_MAE={mae:.4f}")

# ── 저장 ────────────────────────────────────────────────────────
np.save('results/tta_v30_oof.npy', oof_tta_rid)
np.save('results/tta_v30_test.npy', test_tta_rid)
np.save('results/vanilla_v30_oof.npy', oof_van_rid)
np.save('results/vanilla_v30_test.npy', test_van_rid)
print("\nSaved: results/tta_v30_oof.npy, tta_v30_test.npy")
print("Done.")
