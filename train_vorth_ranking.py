"""
V_ORTH features + Ranking loss (double orthogonality).

Different from:
- Original ranking: used v23 features
- Original V_ORTH: used Huber loss

This combines: V_ORTH feature set + ranking loss.
Expected: even lower corr with mega33 (different features + different loss).
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, os, gc
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import lightgbm as lgb

OUT = "results/vorth_ranking"
os.makedirs(OUT, exist_ok=True)
TARGET = "avg_delay_minutes_next_30m"

print("=" * 64)
print("V_ORTH + Ranking Loss (double orthogonality)")
print("=" * 64)
t0 = time.time()

# ---------- Load raw data and build V_ORTH features ----------
print("[1] Loading data + building V_ORTH features...")
train_raw = pd.read_csv("train.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
test_raw = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
layout = pd.read_csv("layout_info.csv")
train_raw["timeslot"] = train_raw.groupby(["layout_id","scenario_id"]).cumcount()
test_raw["timeslot"] = test_raw.groupby(["layout_id","scenario_id"]).cumcount()
train_raw = train_raw.merge(layout, on="layout_id", how="left")
test_raw = test_raw.merge(layout, on="layout_id", how="left")

fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_raw[TARGET].values.astype(np.float64)

def engineer_vorth(df, is_train=True, pca_model=None, km10=None, km20=None,
                   col_mean=None, col_std=None, raw_num_cols=None):
    exclude = {"ID","layout_id","scenario_id",TARGET,"layout_type","timeslot_sq","timeslot_norm"}
    if raw_num_cols is None:
        raw_num_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    df = df.copy()
    df["ts_sin"] = np.sin(2*np.pi*df["timeslot"]/25.0)
    df["ts_cos"] = np.cos(2*np.pi*df["timeslot"]/25.0)
    df["ts_is_start"] = (df["timeslot"]<5).astype(np.int8)
    df["ts_is_mid"] = ((df["timeslot"]>=5)&(df["timeslot"]<20)).astype(np.int8)
    df["ts_is_end"] = (df["timeslot"]>=20).astype(np.int8)
    key8 = ["order_inflow_15m","congestion_score","robot_utilization","battery_mean",
            "fault_count_15m","blocked_path_15m","pack_utilization","charge_queue_length"]
    grp = df.groupby(["layout_id","scenario_id"], sort=False)
    for c in key8:
        if c in df.columns:
            sc_mean = grp[c].transform("mean")
            df[f"{c}_abs_dev_from_mean"] = np.abs(df[c]-sc_mean)
    for c in key8:
        if c in df.columns:
            df[f"{c}_rank_in_sc"] = grp[c].rank(pct=True)
    for t in ["narrow","grid","hybrid","hub_spoke"]:
        df[f"layout_type_{t}"] = (df["layout_type"]==t).astype(np.int8)
    mat_cols = raw_num_cols + ["timeslot","ts_sin","ts_cos"]
    mat_cols = [c for c in mat_cols if c in df.columns]
    mat = df[mat_cols].values.astype(np.float64)
    if col_mean is None:
        col_mean = np.nanmean(mat, axis=0)
    mat = np.where(np.isnan(mat), col_mean, mat)
    if col_std is None:
        col_std = mat.std(axis=0)+1e-6
    mat_std = (mat - mat.mean(axis=0)) / col_std
    if pca_model is None:
        pca_model = PCA(n_components=20, random_state=42); pca_model.fit(mat_std)
    pca_feats = pca_model.transform(mat_std)
    for i in range(20):
        df[f"pca_{i:02d}"] = pca_feats[:,i].astype(np.float32)
    if km10 is None:
        km10 = MiniBatchKMeans(n_clusters=10, random_state=42, batch_size=4096, n_init="auto"); km10.fit(mat_std)
    df["kmeans10"] = km10.predict(mat_std).astype(np.int16)
    if km20 is None:
        km20 = MiniBatchKMeans(n_clusters=20, random_state=42, batch_size=4096, n_init="auto"); km20.fit(mat_std)
    df["kmeans20"] = km20.predict(mat_std).astype(np.int16)
    drop_cols = {"ID","layout_id","scenario_id",TARGET,"layout_type"}
    feat_cols_out = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]
    return df, feat_cols_out, {"pca":pca_model,"k10":km10,"k20":km20,
                               "col_mean":col_mean,"col_std":col_std,"raw_num_cols":raw_num_cols}

train_vo, feat_cols, fitted = engineer_vorth(train_raw, True)
test_vo, _, _ = engineer_vorth(test_raw, False,
    pca_model=fitted["pca"], km10=fitted["k10"], km20=fitted["k20"],
    col_mean=fitted["col_mean"], col_std=fitted["col_std"],
    raw_num_cols=fitted["raw_num_cols"])
feat_cols = [c for c in feat_cols if c in test_vo.columns]
print(f"V_ORTH feats: {len(feat_cols)}")

X_tr = train_vo[feat_cols].values.astype(np.float32)
X_te = test_vo[feat_cols].values.astype(np.float32)

# mega33 baseline
with open("results/mega33_final.pkl","rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
baseline = mean_absolute_error(y, mega33_oof)
print(f"mega33 baseline: {baseline:.5f}")

scenario_key = (train_raw["layout_id"].astype(str)+"_"+train_raw["scenario_id"].astype(str)).values
scenario_key_te = (test_raw["layout_id"].astype(str)+"_"+test_raw["scenario_id"].astype(str)).values

# rank target
grp = train_vo.groupby(
    (train_raw["layout_id"].astype(str)+"_"+train_raw["scenario_id"].astype(str)), sort=False)
rank_within = grp[TARGET].rank(method="average", ascending=True).values
rel = (rank_within - 1).astype(np.int32)

def _gs(idx):
    sizes = []; cnt = 0; prev = None
    for i in idx:
        k = scenario_key[i]
        if k != prev:
            if cnt > 0: sizes.append(cnt)
            cnt = 1; prev = k
        else: cnt += 1
    if cnt > 0: sizes.append(cnt)
    return sizes

def rank_adjust(scores, mega_preds, sc_keys):
    out = mega_preds.copy().astype(np.float64)
    df = pd.DataFrame({"sc":sc_keys,"rs":scores,"m":mega_preds,"i":np.arange(len(mega_preds))})
    for sc, g in df.groupby("sc", sort=False):
        if len(g) < 2: continue
        sorted_m = np.sort(g["m"].values)
        order = np.argsort(g["rs"].values)
        new = np.empty(len(g))
        for pos, idx_ in enumerate(order):
            new[idx_] = sorted_m[pos]
        out[g["i"].values] = new
    return out

print("\n[2] Training LGB-lambdarank on V_ORTH, 5-fold...")
PARAMS = dict(
    objective="lambdarank", metric="ndcg",
    n_estimators=2000, learning_rate=0.03,
    num_leaves=63, max_depth=8, min_child_samples=50,
    subsample=0.7, colsample_bytree=0.7,
    reg_alpha=1.0, reg_lambda=1.0,
    random_state=42, verbose=-1, n_jobs=-1,
    label_gain=list(range(25)),
)

rank_raw_oof = np.zeros(len(y))
rank_raw_test = np.zeros(len(X_te))
for f in range(5):
    tv = time.time()
    tr = np.where(fold_ids != f)[0]; val = np.where(fold_ids == f)[0]
    trs = _gs(tr); vals = _gs(val)
    m = lgb.LGBMRanker(**PARAMS)
    m.fit(X_tr[tr], rel[tr], group=trs,
          eval_set=[(X_tr[val], rel[val])], eval_group=[vals],
          callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    rank_raw_oof[val] = m.predict(X_tr[val])
    rank_raw_test += m.predict(X_te) / 5
    print(f"  fold {f}: best_iter={m.best_iteration_} ({time.time()-tv:.0f}s)")
    del m

print("\n[3] Rank-adjusting mega33 by V_ORTH rank scores...")
vr_oof = rank_adjust(rank_raw_oof, mega33_oof, scenario_key)
vr_test = rank_adjust(rank_raw_test, mega33_test, scenario_key_te)

vr_mae = mean_absolute_error(y, vr_oof)
corr = float(np.corrcoef(y - mega33_oof, y - vr_oof)[0, 1])
print(f"V_ORTH+rank single_mae: {vr_mae:.5f}")
print(f"residual_corr(mega33, vorth_rank): {corr:.4f}")

# compare to original ranking
orig_rank_oof = np.load("results/ranking/rank_adj_oof.npy")
corr_orig_vr = float(np.corrcoef(y - orig_rank_oof, y - vr_oof)[0, 1])
print(f"residual_corr(orig_rank, vorth_rank): {corr_orig_vr:.4f}")

# --- blend with mega33 ---
print("\n[4] Blend with mega33:")
best_w, best_mae = 0, float(baseline)
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    b = (1-w)*mega33_oof + w*vr_oof
    mae = float(mean_absolute_error(y, b))
    d = mae - baseline
    tag = " ***" if d < -0.005 else (" **" if d < -0.003 else "")
    print(f"  w={w:.2f}: mae={mae:.5f} delta={d:+.5f}{tag}")
    if mae < best_mae: best_w, best_mae = w, mae

np.save(f"{OUT}/vorth_rank_oof.npy", vr_oof)
np.save(f"{OUT}/vorth_rank_test.npy", vr_test)

# --- full mega-blend v3 with vorth_rank added ---
print("\n[5] Full mega-blend v3 (adding vorth_rank)...")
from scipy.optimize import minimize
sources = [
    ("mega33", mega33_oof, mega33_test),
    ("rank_orig", np.load("results/ranking/rank_adj_oof.npy"),
     np.load("results/ranking/rank_adj_test.npy")),
    ("vorth_rank", vr_oof, vr_test),
    ("iter_r1", np.load("results/iter_pseudo/round1_oof.npy"),
     np.load("results/iter_pseudo/round1_test.npy")),
    ("iter_r2", np.load("results/iter_pseudo/round2_oof.npy"),
     np.load("results/iter_pseudo/round2_test.npy")),
    ("iter_r3", np.load("results/iter_pseudo/round3_oof.npy"),
     np.load("results/iter_pseudo/round3_test.npy")),
]
names = [s[0] for s in sources]
oofs_M = np.column_stack([s[1] for s in sources])
tests_M = np.column_stack([s[2] for s in sources])
n = len(sources)

def obj_fn(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    return mean_absolute_error(y, oofs_M @ w)

x0 = np.zeros(n); x0[0] = 0.75
for i in range(1, n): x0[i] = 0.25 / (n-1)
res = minimize(obj_fn, x0, method="Nelder-Mead", options={"xatol":1e-7, "maxiter":100000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
delta = res.fun - baseline

print(f"\nFinal mega-blend v3 weights:")
for nm, w_ in zip(names, w_opt):
    if w_ > 0.001:
        print(f"  {nm:15s}: {w_:.4f}")
print(f"\nOOF: {res.fun:.5f}  delta: {delta:+.6f}")
print(f"Previous v2 (mega33+rank+iter): -0.005434")
print(f"Improvement from v2: {delta - (-0.005434):+.6f}")

# submission
blend_test = tests_M @ w_opt
sorted_ids = pd.read_csv("test.csv").sort_values(["layout_id","scenario_id"])["ID"].values
pd.DataFrame({"ID": sorted_ids, TARGET: np.clip(blend_test, 0, None)}).to_csv(
    f"{OUT}/submission_megablend_v3.csv", index=False)
print(f"\nSubmission: {OUT}/submission_megablend_v3.csv")

json.dump({
    "vorth_rank_oof_mae": float(vr_mae),
    "residual_corr_mega33": corr,
    "residual_corr_orig_rank": corr_orig_vr,
    "best_single_blend_w": float(best_w),
    "best_single_blend_delta": float(best_mae - baseline),
    "full_blend_weights": dict(zip(names, w_opt.tolist())),
    "full_blend_mae": float(res.fun),
    "full_blend_delta": float(delta),
    "elapsed_min": round((time.time()-t0)/60, 1),
}, open(f"{OUT}/vorth_ranking_summary.json","w"), indent=2, ensure_ascii=False)

print(f"\n{'='*64}")
print(f"FINAL: delta {delta:+.6f}  elapsed: {(time.time()-t0)/60:.1f}min")
print(f"{'='*64}")
