"""Quick blend eval: v31-LGB OOF vs FIXED"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

ROOT = r"C:\Users\user\Desktop\데이콘 4월"

train_raw = pd.read_csv(os.path.join(ROOT, "train.csv")).sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
y_true = train_raw["avg_delay_minutes_next_30m"].values

with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
    d = pickle.load(f)

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568306692235380, iter_r3=0.031038826035934514)
mega2 = d['meta_avg_oof']
rank2 = np.load(os.path.join(ROOT, "results", "ranking", "rank_adj_oof.npy"))
it1   = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round1_oof.npy"))
it2   = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round2_oof.npy"))
it3   = np.load(os.path.join(ROOT, "results", "iter_pseudo", "round3_oof.npy"))
fixed = fw['mega33']*mega2 + fw['rank_adj']*rank2 + fw['iter_r1']*it1 + fw['iter_r2']*it2 + fw['iter_r3']*it3

xgb_oof = np.load(os.path.join(ROOT, "results", "oracle_seq", "oof_seqC_xgb.npy"))
lv2_oof = np.load(os.path.join(ROOT, "results", "oracle_seq", "oof_seqC_log_v2.npy"))

# v31-LGB is sorted by layout_id, scenario_id — same as train_raw
with open(os.path.join(ROOT, "results", "eda_v31", "v31_fe_cache.pkl"), "rb") as f:
    v31 = pickle.load(f)
train_v31 = v31['train_fe']
# align v31 OOF to train_raw order
v31_oof_raw = np.load(os.path.join(ROOT, "results", "eda_v31", "v31_lgb_oof.npy"))
ls2 = {row['ID']:i for i,row in train_raw.iterrows()}
id2 = [ls2[i] for i in train_v31['ID'].values]
v31_oof = np.zeros(len(train_raw))
v31_oof[id2] = v31_oof_raw

fixed_mae = np.mean(np.abs(fixed - y_true))
xgb_mae   = np.mean(np.abs(xgb_oof - y_true))
lv2_mae   = np.mean(np.abs(lv2_oof - y_true))
v31_mae   = np.mean(np.abs(v31_oof - y_true))

print(f"FIXED MAE:   {fixed_mae:.4f}")
print(f"XGB OOF:     {xgb_mae:.4f}")
print(f"LV2 OOF:     {lv2_mae:.4f}")
print(f"v31-LGB OOF: {v31_mae:.4f}")
print(f"v31 corr w/ FIXED: {np.corrcoef(fixed, v31_oof)[0,1]:.4f}")
print(f"v31 corr w/ XGB:   {np.corrcoef(xgb_oof, v31_oof)[0,1]:.4f}")
print(f"v31 corr w/ LV2:   {np.corrcoef(lv2_oof, v31_oof)[0,1]:.4f}")

# Current best 5-way: FIXED*(1-0.12-0.20) + XGB*0.12 + LV2*0.20
base5 = (1-0.12-0.20)*fixed + 0.12*xgb_oof + 0.20*lv2_oof
base5_mae = np.mean(np.abs(base5 - y_true))
print(f"\nCurrent best (FIXED+xgb0.12+lv2=0.20) MAE: {base5_mae:.4f}")

# 1-way blend search: FIXED + v31
print("\n--- 1-way: FIXED + v31 ---")
best_m = fixed_mae; best_w = 0
for w in np.arange(0, 0.51, 0.02):
    bl = (1-w)*fixed + w*v31_oof
    mm = np.mean(np.abs(bl - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"Best: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}")

# Add v31 to current best 5-way
print("\n--- 4-way: base5 + v31 ---")
best_m4 = base5_mae; best_w4 = 0
for w in np.arange(0.02, 0.21, 0.02):
    bl = (1-w)*base5 + w*v31_oof
    mm = np.mean(np.abs(bl - y_true))
    if mm < best_m4: best_m4 = mm; best_w4 = w
print(f"Best: w={best_w4:.2f}  MAE={best_m4:.4f}  delta={best_m4-base5_mae:+.4f}")
print("Done.")
