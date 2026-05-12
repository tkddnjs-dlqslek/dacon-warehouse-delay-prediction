"""
로컬에서 실행 → colab_data/ 폴더에 numpy 배열 저장 → zip으로 패키징
Colab에서는 이 zip만 업로드하면 됩니다.
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, zipfile
import warnings; warnings.filterwarnings('ignore')
os.chdir("C:/Users/user/Desktop/데이콘 4월")

os.makedirs('colab_data', exist_ok=True)

# ── 기본 데이터 ─────────────────────────────────────
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

# ── v30 피처 ────────────────────────────────────────
with open('results/eda_v30/v30_fe_cache.pkl', 'rb') as f:
    blob = pickle.load(f)
train_fe = blob['train_fe']
feat_cols = list(blob['feat_cols'])
with open('results/eda_v30/v30_test_fe_cache.pkl', 'rb') as f:
    test_fe = pickle.load(f)
fold_ids = np.load('results/eda_v30/fold_idx.npy')

X_tr_ls = train_fe[feat_cols].values.astype(np.float32)   # ls order
X_te_ls = test_fe[feat_cols].values.astype(np.float32)    # ls order
y_ls    = train_fe['avg_delay_minutes_next_30m'].values.astype(np.float32)

# layout_id → int encoding (GroupKFold용)
layout_ids_ls = train_fe['layout_id'].values
unique_lids = {lid: i for i, lid in enumerate(sorted(set(layout_ids_ls)))}
groups_ls = np.array([unique_lids[lid] for lid in layout_ids_ls], dtype=np.int32)

# ── oracle_NEW OOF + test ───────────────────────────
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

oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values

print(f"oracle_NEW OOF MAE: {np.mean(np.abs(y_true - fw4_oo)):.4f}")
print(f"oracle_NEW unseen:  {oracle_new_t[unseen_mask].mean():.3f}")

# ── 저장 ────────────────────────────────────────────
# 피처 (ls order — Colab 스크립트가 fold_ids와 동일 order 사용)
np.save('colab_data/X_train.npy', X_tr_ls)
np.save('colab_data/X_test.npy',  X_te_ls)
np.save('colab_data/y_train.npy', y_ls)
np.save('colab_data/groups.npy',  groups_ls)
np.save('colab_data/fold_ids.npy', fold_ids)

# oracle_NEW (rid order — 비교용)
np.save('colab_data/oracle_oof.npy',  fw4_oo.astype(np.float32))
np.save('colab_data/oracle_test.npy', oracle_new_t.astype(np.float32))

# index 변환
np.save('colab_data/id2.npy',          np.array(id2, dtype=np.int32))
np.save('colab_data/te_rid_to_ls.npy', te_rid_to_ls.astype(np.int32))
np.save('colab_data/unseen_mask.npy',  unseen_mask)
np.save('colab_data/y_true_rid.npy',   y_true.astype(np.float32))

# 샘플 서브미션
sample = pd.read_csv('sample_submission.csv')
sample.to_csv('colab_data/sample_submission.csv', index=False)

print(f"\nSaved to colab_data/:")
for f in sorted(os.listdir('colab_data')):
    mb = os.path.getsize(f'colab_data/{f}') / 1024**2
    print(f"  {f:<30} {mb:6.1f} MB")

# ── zip 패키징 ──────────────────────────────────────
zip_path = 'colab_data.zip'
with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
    for fname in os.listdir('colab_data'):
        zf.write(f'colab_data/{fname}', fname)
total_mb = os.path.getsize(zip_path) / 1024**2
print(f"\nZipped → {zip_path} ({total_mb:.1f} MB)")
print("Done. Upload colab_data.zip to Colab.")
