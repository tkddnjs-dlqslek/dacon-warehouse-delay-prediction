"""
MM Round: oracle_NEW + oracle_layout_v3 앙상블 (과적합 없음)

layout_v3는 layout-normalized oracle → unseen layout 일반화 특화
oracle_NEW를 베이스로 고정 가중치로 blending (scipy 없음)

MM1: oracle_NEW + layout_v3 (w=0.05)
MM2: oracle_NEW + layout_v3 (w=0.10)
MM3: oracle_NEW + layout_v3 (w=0.15)
MM4: oracle_NEW + layout_v3 (w=0.20)
MM5: oracle_NEW (unseen) + layout_v3 (seen/unseen mixed)
MM6: seen=oracle_NEW, unseen=layout_v3 only
MM7: seen=oracle_NEW, unseen=70% oracle_NEW + 30% layout_v3
MM8: seen=oracle_NEW, unseen=layout_v3-heavy (0.50)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings('ignore')

OOF_LV3  = 'results/oracle_seq/oof_seqC_layout_v3.npy'
TEST_LV3 = 'results/oracle_seq/test_C_layout_v3.npy'

if not os.path.exists(OOF_LV3):
    print("layout_v3 OOF 없음 — train_oracle_layout_v3.py 먼저 실행")
    sys.exit(1)

print("="*60)
print("MM Round: oracle_NEW + oracle_layout_v3")
print("="*60)

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

sample_sub = pd.read_csv('sample_submission.csv')
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw['mega33']*d33['meta_avg_oof'][id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_oof.npy')[id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_oof.npy')[id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_oof.npy')[id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_oof.npy')[id2])
fixed_te  = (fw['mega33']*d33['meta_avg_test'][te_id2]
           + fw['rank_adj']*np.load('results/ranking/rank_adj_test.npy')[te_id2]
           + fw['iter_r1']*np.load('results/iter_pseudo/round1_test.npy')[te_id2]
           + fw['iter_r2']*np.load('results/iter_pseudo/round2_test.npy')[te_id2]
           + fw['iter_r3']*np.load('results/iter_pseudo/round3_test.npy')[te_id2])

xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')

oracle_new_oof = 0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
oracle_new_te  = 0.64*fixed_te  + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t

lv3_oof  = np.load(OOF_LV3)
lv3_te   = np.load(TEST_LV3)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
oof_new = mae(oracle_new_oof)
oof_lv3 = mae(lv3_oof)
corr    = float(np.corrcoef(oracle_new_oof, lv3_oof)[0,1])
print(f"\noracle_NEW OOF: {oof_new:.5f}")
print(f"layout_v3  OOF: {oof_lv3:.5f}")
print(f"corr:           {corr:.4f}")
print()

# seen/unseen
train_layout_ids = set(train_raw['layout_id'].unique())
test_layouts = test_raw['layout_id'].values
seen_mask   = np.array([lid in train_layout_ids for lid in test_layouts])
unseen_mask = ~seen_mask

saved = []
def save_sub(test_pred, oofmae, label):
    fname = f'submission_{label}_OOF{oofmae:.5f}.csv'
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({'ID': test_raw['ID'].values, 'avg_delay_minutes_next_30m': sub})
    df = df.set_index('ID').loc[sample_sub['ID'].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  ★ SAVED [{label}]: OOF={oofmae:.5f}  → {fname}")

# ── MM1~MM4: 전체에 고정 비율 블렌딩 ──────────────────────────
for w, name in [(0.05,'MM1'), (0.10,'MM2'), (0.15,'MM3'), (0.20,'MM4')]:
    print(f"\n=== {name}: oracle_NEW*(1-{w})+layout_v3*{w} ===")
    b_oof = (1-w)*oracle_new_oof + w*lv3_oof
    b_te  = (1-w)*oracle_new_te  + w*lv3_te
    oof_mm = mae(b_oof)
    delta  = oof_mm - oof_new
    print(f"  OOF={oof_mm:.5f}  delta={delta:+.5f}")
    save_sub(b_te, oof_mm, name)

# ── MM5~MM8: layout-conditional (unseen만 layout_v3 혼합) ────────
print(f"\n=== MM5: unseen layout에만 layout_v3 20% 혼합 ===")
bt_MM5 = oracle_new_te.copy()
bt_MM5[unseen_mask] = 0.80*oracle_new_te[unseen_mask] + 0.20*lv3_te[unseen_mask]
b5_oof = oracle_new_oof.copy()  # train OOF = oracle_NEW (train은 모두 seen)
oof_mm5 = mae(b5_oof)
print(f"  OOF={oof_mm5:.5f} [unseen에만 lv3 20%]")
save_sub(bt_MM5, oof_mm5, 'MM5_unseen_lv3_20pct')

print(f"\n=== MM6: seen=oracle_NEW, unseen=layout_v3 only ===")
bt_MM6 = oracle_new_te.copy()
bt_MM6[unseen_mask] = lv3_te[unseen_mask]
print(f"  OOF={oof_mm5:.5f} [unseen: layout_v3 단독]")
save_sub(bt_MM6, oof_mm5, 'MM6_unseen_lv3_only')

print(f"\n=== MM7: seen=oracle_NEW, unseen=70% oracle + 30% lv3 ===")
bt_MM7 = oracle_new_te.copy()
bt_MM7[unseen_mask] = 0.70*oracle_new_te[unseen_mask] + 0.30*lv3_te[unseen_mask]
print(f"  OOF={oof_mm5:.5f} [unseen: 70% oracle + 30% lv3]")
save_sub(bt_MM7, oof_mm5, 'MM7_unseen_oracle70_lv3_30')

print(f"\n=== MM8: seen=oracle_NEW, unseen=50% oracle + 50% lv3 ===")
bt_MM8 = oracle_new_te.copy()
bt_MM8[unseen_mask] = 0.50*oracle_new_te[unseen_mask] + 0.50*lv3_te[unseen_mask]
print(f"  OOF={oof_mm5:.5f} [unseen: 50/50 blend]")
save_sub(bt_MM8, oof_mm5, 'MM8_unseen_oracle50_lv3_50')

# ── Summary ───────────────────────────────────────────────────
print("\n" + "="*60)
print("MM ROUND SUMMARY")
for label, oofmae, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label:<40}  OOF={oofmae:.5f}")
print(f"\nTotal: {len(saved)} files")
print("\n제출 우선순위 (lv3 corr 낮으면 MM1-MM4, 높으면 MM5-MM8):")
print(f"  corr(oracle_NEW, layout_v3) = {corr:.4f}")
if corr < 0.97:
    print("  corr < 0.97 → 다양성 있음 → MM2/MM3 우선")
else:
    print("  corr >= 0.97 → 다양성 낮음 → MM5/MM7 (layout-conditional) 우선")
