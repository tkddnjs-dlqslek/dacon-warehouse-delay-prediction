"""
Colab TabNet + FT-Transformer for stacking diversity.

Run in Google Colab with T4 GPU.

Pre-requisites:
1. Upload to Colab: train.csv, test.csv, layout_info.csv
   AND results/eda_v30/fold_idx.npy, results/mega33_final.pkl
2. Install: !pip install pytorch-tabnet rtdl-revisiting-models torch

Expected runtime: TabNet ~45 min, FT-Transformer ~45 min per model (T4).

Output: TabNet/FT-T OOF and test predictions, to be added as new bases in mega34 stacking.

Goal: residual correlation with mega33 < 0.95 → genuine diversity → -0.02+ OOF improvement on stacking.

This is the "paradigm change" attempt. GBDT/NN (MLP/CNN/LSTM) all tried on v23 features;
TabNet and FT-Transformer use attention-based feature interactions → fundamentally different.
"""
# =======================================================================
# SECTION 1: Setup (Colab cell)
# =======================================================================
# !pip install pytorch-tabnet rtdl-revisiting-models torch pandas numpy scikit-learn lightgbm
# from google.colab import files
# files.upload()  # upload train.csv, test.csv, layout_info.csv, fold_idx.npy, mega33_final.pkl

import numpy as np
import pandas as pd
import pickle
import time
import torch
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
print(f"torch: {torch.__version__}")

# =======================================================================
# SECTION 2: Feature engineering (v23 recipe, no extreme_prob)
# =======================================================================
TARGET = "avg_delay_minutes_next_30m"

def engineer_features_v23(df, layout_df):
    df = df.merge(layout_df, on="layout_id", how="left")
    df["timeslot"] = df.groupby(["layout_id", "scenario_id"]).cumcount()
    df["timeslot_sq"] = df["timeslot"] ** 2
    df["timeslot_norm"] = df["timeslot"] / 24.0
    df = df.sort_values(["layout_id", "scenario_id", "timeslot"]).reset_index(drop=True)
    group = df.groupby(["layout_id", "scenario_id"])

    key_cols = ["order_inflow_15m", "congestion_score", "robot_utilization",
                "battery_mean", "fault_count_15m", "blocked_path_15m",
                "pack_utilization", "charge_queue_length"]
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lag1"] = g.shift(1)
        df[f"{col}_lag2"] = g.shift(2)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_rmean3"] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f"{col}_rstd3"] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f"{col}_rmean5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        df[f"{col}_cummean"] = g.transform(lambda x: x.expanding().mean())
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_lead2"] = g.shift(-2)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]
    extra_cols = ["max_zone_density", "robot_charging", "low_battery_ratio",
                  "robot_idle", "near_collision_15m"]
    for col in extra_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_lag1"] = g.shift(1)
        df[f"{col}_diff1"] = df[col] - df[f"{col}_lag1"]
        df[f"{col}_lead1"] = g.shift(-1)
        df[f"{col}_diff_lead1"] = df[f"{col}_lead1"] - df[col]
    for col in key_cols:
        if col not in df.columns: continue
        g = group[col]
        df[f"{col}_sc_mean"] = g.transform("mean")
        df[f"{col}_sc_std"] = g.transform("std")
        df[f"{col}_sc_max"] = g.transform("max")
        df[f"{col}_sc_min"] = g.transform("min")
        df[f"{col}_sc_range"] = df[f"{col}_sc_max"] - df[f"{col}_sc_min"]
        df[f"{col}_sc_rank"] = g.rank(pct=True)
        df[f"{col}_sc_dev"] = df[col] - df[f"{col}_sc_mean"]

    # interactions + ratios (abbreviated, copy from train_v23.py for full)
    df["order_per_robot"] = df["order_inflow_15m"] / (df["robot_active"] + 1)
    df["congestion_x_utilization"] = df["congestion_score"] * df["robot_utilization"]
    df["fault_x_congestion"] = df["fault_count_15m"] * df["congestion_score"]
    df["floor_area_per_robot"] = df["floor_area_sqm"] / (df["robot_total"] + 1)
    df["charger_ratio"] = df["charger_count"] / (df["robot_total"] + 1)
    layout_static = ["layout_type", "aisle_width_avg", "intersection_count", "one_way_ratio",
                     "pack_station_count", "charger_count", "layout_compactness", "zone_dispersion",
                     "robot_total", "building_age_years", "floor_area_sqm", "ceiling_height_m",
                     "fire_sprinkler_count", "emergency_exit_count"]
    df = df.drop(columns=[c for c in layout_static if c in df.columns], errors="ignore")
    return df

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
layout = pd.read_csv("layout_info.csv")
train_fe = engineer_features_v23(train, layout)
test_fe = engineer_features_v23(test, layout)

# select numeric feature cols (exclude IDs and target)
exclude = {"ID", "layout_id", "scenario_id", TARGET}
feat_cols = [c for c in train_fe.columns if c not in exclude and pd.api.types.is_numeric_dtype(train_fe[c])]
feat_cols = [c for c in feat_cols if c in test_fe.columns]
print(f"features: {len(feat_cols)}")

# fillna (NNs don't handle NaN)
for c in feat_cols:
    m = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(m)
    test_fe[c] = test_fe[c].fillna(m)

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
y = train_fe[TARGET].values.astype(np.float32)
y_log = np.log1p(y)

# standardize (important for NN)
mean = X_tr.mean(axis=0); std = X_tr.std(axis=0) + 1e-6
X_tr = (X_tr - mean) / std
X_te = (X_te - mean) / std

# load fold_idx
fold_ids = np.load("fold_idx.npy")
print(f"X_tr={X_tr.shape} y_log range=[{y_log.min():.2f},{y_log.max():.2f}]")

# =======================================================================
# SECTION 3: TabNet training
# =======================================================================
from pytorch_tabnet.tab_model import TabNetRegressor

tabnet_oof = np.zeros(len(y))
tabnet_test = np.zeros(len(X_te))
for f in range(5):
    t0 = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]
    model = TabNetRegressor(
        n_d=64, n_a=64, n_steps=5, gamma=1.5,
        n_independent=2, n_shared=2,
        momentum=0.3, lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=10, gamma=0.9),
        mask_type="entmax",
        seed=42,
        device_name=device,
    )
    model.fit(
        X_train=X_tr[tr], y_train=y_log[tr].reshape(-1, 1),
        eval_set=[(X_tr[val], y_log[val].reshape(-1, 1))],
        eval_metric=["mae"],
        max_epochs=100, patience=20,
        batch_size=4096, virtual_batch_size=512,
        num_workers=0, drop_last=False,
        loss_fn=torch.nn.HuberLoss(delta=0.9),
    )
    pred = np.clip(np.expm1(model.predict(X_tr[val]).flatten()), 0, None)
    tabnet_oof[val] = pred
    tabnet_test += np.clip(np.expm1(model.predict(X_te).flatten()), 0, None) / 5
    print(f"TabNet fold {f}: mae={mean_absolute_error(y[val], pred):.4f} ({time.time()-t0:.0f}s)")

tabnet_mae = mean_absolute_error(y, tabnet_oof)
print(f"TabNet OOF: {tabnet_mae:.5f}")
np.save("tabnet_oof.npy", tabnet_oof)
np.save("tabnet_test.npy", tabnet_test)

# =======================================================================
# SECTION 4: FT-Transformer training
# =======================================================================
import rtdl_revisiting_models as rtdl

ft_oof = np.zeros(len(y))
ft_test = np.zeros(len(X_te))
n_feat = X_tr.shape[1]

for f in range(5):
    t0 = time.time()
    tr = np.where(fold_ids != f)[0]
    val = np.where(fold_ids == f)[0]

    model = rtdl.FTTransformer(
        n_num_features=n_feat,
        cat_cardinalities=[],
        d_out=1,
        d_block=192,
        n_blocks=3,
        attention_n_heads=8,
        attention_dropout=0.2,
        ffn_d_hidden=384,
        ffn_dropout=0.1,
        residual_dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=60)
    loss_fn = torch.nn.HuberLoss(delta=0.9)

    Xt = torch.from_numpy(X_tr[tr]).to(device)
    yt = torch.from_numpy(y_log[tr]).to(device)
    Xv = torch.from_numpy(X_tr[val]).to(device)
    yv = torch.from_numpy(y_log[val]).to(device)

    best_mae = 99; best_state = None; patience = 0
    bs = 4096
    for epoch in range(60):
        model.train()
        perm = torch.randperm(len(Xt))
        total_loss = 0
        for i in range(0, len(Xt), bs):
            idx = perm[i:i+bs]
            opt.zero_grad()
            pred = model(Xt[idx], None).flatten()
            loss = loss_fn(pred, yt[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        scheduler.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xv, None).flatten().cpu().numpy()
        pv = np.clip(np.expm1(pv), 0, None)
        mae = mean_absolute_error(y[val], pv)
        if mae < best_mae:
            best_mae = mae; best_state = {k: v.clone() for k, v in model.state_dict().items()}; patience = 0
        else:
            patience += 1
        if patience >= 10: break
        if epoch % 5 == 0:
            print(f"  FT fold{f} ep{epoch}: loss={total_loss/len(Xt):.4f} val_mae={mae:.4f} best={best_mae:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(Xv, None).flatten().cpu().numpy()
        pt = model(torch.from_numpy(X_te).to(device), None).flatten().cpu().numpy()
    ft_oof[val] = np.clip(np.expm1(pv), 0, None)
    ft_test += np.clip(np.expm1(pt), 0, None) / 5
    print(f"FT fold {f}: best_mae={best_mae:.4f} ({time.time()-t0:.0f}s)")

ft_mae = mean_absolute_error(y, ft_oof)
print(f"FT-Transformer OOF: {ft_mae:.5f}")
np.save("ft_oof.npy", ft_oof)
np.save("ft_test.npy", ft_test)

# =======================================================================
# SECTION 5: Diversity check with mega33
# =======================================================================
with open("mega33_final.pkl", "rb") as fp:
    mega = pickle.load(fp)
mega33_oof = np.asarray(mega["meta_avg_oof"])
mega33_test = np.asarray(mega["meta_avg_test"])
base_mae = mean_absolute_error(y, mega33_oof)

for name, oof in [("tabnet", tabnet_oof), ("ft_transformer", ft_oof)]:
    r_mega = y - mega33_oof; r_new = y - oof
    corr = np.corrcoef(r_mega, r_new)[0, 1]
    print(f"{name} residual corr with mega33: {corr:.4f} | OOF mae: {mean_absolute_error(y, oof):.4f}")

    best_w, best_mae = 0, base_mae
    for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20]:
        b = (1-w)*mega33_oof + w*oof
        mae = mean_absolute_error(y, b)
        if mae < best_mae:
            best_w, best_mae = w, mae
    print(f"  best blend: w={best_w} mae={best_mae:.5f} delta={best_mae - base_mae:+.5f}")

# Final submission if either helps
# pd.DataFrame({"ID": test_fe["ID"].values, TARGET: best_blend_test}).to_csv("submission_colab.csv", index=False)
print("\nDone. Download tabnet_*.npy and ft_*.npy back to local for mega34 stacking.")
