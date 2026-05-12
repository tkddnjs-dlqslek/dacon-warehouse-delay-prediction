
import sys; sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd, pickle, os, warnings
warnings.filterwarnings("ignore")

OOF_LP  = "results/oracle_seq/oof_seqC_layout_proxy.npy"
TEST_LP = "results/oracle_seq/test_C_layout_proxy.npy"
if not os.path.exists(OOF_LP):
    print("layout_proxy 없음"); sys.exit(1)

train_raw = pd.read_csv("train.csv")
train_raw["_row_id"] = train_raw["ID"].str.replace("TRAIN_","").astype(int)
train_raw = train_raw.sort_values("_row_id").reset_index(drop=True)
y_true = train_raw["avg_delay_minutes_next_30m"].values
test_raw = pd.read_csv("test.csv")
test_raw["_row_id"] = test_raw["ID"].str.replace("TEST_","").astype(int)
test_raw = test_raw.sort_values("_row_id").reset_index(drop=True)

train_ls = pd.read_csv("train.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
test_ls  = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
ls_pos   = {row["ID"]:i for i,row in train_ls.iterrows()}
te_ls_pos= {row["ID"]:i for i,row in test_ls.iterrows()}
id2    = [ls_pos[i]    for i in train_raw["ID"].values]
te_id2 = [te_ls_pos[i] for i in test_raw["ID"].values]
sample_sub = pd.read_csv("sample_submission.csv")
with open("results/mega33_final.pkl","rb") as f: d33 = pickle.load(f)

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed_oof = (fw["mega33"]*d33["meta_avg_oof"][id2]
           + fw["rank_adj"]*np.load("results/ranking/rank_adj_oof.npy")[id2]
           + fw["iter_r1"]*np.load("results/iter_pseudo/round1_oof.npy")[id2]
           + fw["iter_r2"]*np.load("results/iter_pseudo/round2_oof.npy")[id2]
           + fw["iter_r3"]*np.load("results/iter_pseudo/round3_oof.npy")[id2])
fixed_te  = (fw["mega33"]*d33["meta_avg_test"][te_id2]
           + fw["rank_adj"]*np.load("results/ranking/rank_adj_test.npy")[te_id2]
           + fw["iter_r1"]*np.load("results/iter_pseudo/round1_test.npy")[te_id2]
           + fw["iter_r2"]*np.load("results/iter_pseudo/round2_test.npy")[te_id2]
           + fw["iter_r3"]*np.load("results/iter_pseudo/round3_test.npy")[te_id2])

xgb_o = np.load("results/oracle_seq/oof_seqC_xgb.npy")
lv2_o = np.load("results/oracle_seq/oof_seqC_log_v2.npy")
rem_o = np.load("results/oracle_seq/oof_seqC_xgb_remaining.npy")
xgb_t = np.load("results/oracle_seq/test_C_xgb.npy")
lv2_t = np.load("results/oracle_seq/test_C_log_v2.npy")
rem_t = np.load("results/oracle_seq/test_C_xgb_remaining.npy")
oracle_new_oof = 0.64*fixed_oof + 0.12*xgb_o + 0.16*lv2_o + 0.08*rem_o
oracle_new_te  = 0.64*fixed_te  + 0.12*xgb_t + 0.16*lv2_t + 0.08*rem_t

lp_oof = np.load(OOF_LP)
lp_te  = np.load(TEST_LP)

mae = lambda p: float(np.mean(np.abs(np.clip(p,0,None) - y_true)))
oof_new = mae(oracle_new_oof); oof_lp = mae(lp_oof)
corr = float(np.corrcoef(oracle_new_oof, lp_oof)[0,1])
print(f"oracle_NEW OOF: {oof_new:.5f}")
print(f"layout_proxy OOF: {oof_lp:.5f}")
print(f"corr: {corr:.4f}")

train_layout_ids = set(train_raw["layout_id"].unique())
test_layouts = test_raw["layout_id"].values
seen_mask   = np.array([lid in train_layout_ids for lid in test_layouts])
unseen_mask = ~seen_mask

saved = []
def save_sub(test_pred, oofmae, label):
    fname = f"submission_{label}_OOF{oofmae:.5f}.csv"
    sub = np.maximum(0, test_pred)
    df = pd.DataFrame({"ID": test_raw["ID"].values, "avg_delay_minutes_next_30m": sub})
    df = df.set_index("ID").loc[sample_sub["ID"].values].reset_index()
    df.to_csv(fname, index=False)
    saved.append((label, oofmae, fname))
    print(f"  SAVED [{label}]: OOF={oofmae:.5f}")

for w, name in [(0.05,"MM1lp"), (0.10,"MM2lp"), (0.15,"MM3lp"), (0.20,"MM4lp")]:
    b_oof = (1-w)*oracle_new_oof + w*lp_oof
    b_te  = (1-w)*oracle_new_te  + w*lp_te
    save_sub(b_te, mae(b_oof), name)

# layout-conditional
for alpha, name in [(0.20,"MM5lp_unseen20"), (0.30,"MM6lp_unseen30"), (0.50,"MM7lp_unseen50")]:
    bt = oracle_new_te.copy()
    bt[unseen_mask] = (1-alpha)*oracle_new_te[unseen_mask] + alpha*lp_te[unseen_mask]
    save_sub(bt, oof_new, name)

print()
for label, oof, fname in sorted(saved, key=lambda x: x[1]):
    print(f"  {label}: OOF={oof:.5f}")
print("Done.")
