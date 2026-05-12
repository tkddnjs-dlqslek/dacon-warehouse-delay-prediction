"""
Adversarial Sample Weighting:
1) Train LGB classifier to distinguish train vs test on v23 features
2) For each train sample, compute P(test|x) / (1 - P(test|x)) as importance weight
3) Train LGB-Huber with these weights → model focuses on train samples similar to test
4) Evaluate OOF MAE + residual correlation with mega33 + blend with mega33

Goal: create a base model whose residuals differ from mega33 (corr < 0.95)
      → new diversity axis for stacking
"""
import pickle, numpy as np, pandas as pd, time, json, warnings, gc, sys
warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error, roc_auc_score
import lightgbm as lgb

sys.stdout.reconfigure(line_buffering=True)

OUT = "results/v24_cumsum"
TARGET = "avg_delay_minutes_next_30m"

print("=" * 60, flush=True)
print("Adversarial Sample Weighting", flush=True)
print("=" * 60, flush=True)
t0 = time.time()

# Load
with open("results/eda_v30/v30_fe_cache.pkl", "rb") as f:
    blob = pickle.load(f)
train_fe = blob["train_fe"]
feat_cols = list(blob["feat_cols"])
with open("results/eda_v30/v30_test_fe_cache.pkl", "rb") as f:
    test_fe = pickle.load(f)
fold_ids = np.load("results/eda_v30/fold_idx.npy")
y = train_fe[TARGET].values.astype(np.float64)

with open("results/mega33_final.pkl", "rb") as f:
    mega = pickle.load(f)
mega33_oof = np.asarray(mega["meta_avg_oof"])
baseline_mega = mean_absolute_error(y, mega33_oof)

X_tr = train_fe[feat_cols].values.astype(np.float32)
X_te = test_fe[feat_cols].values.astype(np.float32)
y_log = np.log1p(y)

del blob, test_fe, mega, train_fe
gc.collect()

print(f"X_tr={X_tr.shape} X_te={X_te.shape} mega33_baseline={baseline_mega:.5f} (loaded {time.time()-t0:.0f}s)", flush=True)

# ----- Step 1: Adversarial classifier -----
print("\nStep 1: adversarial classifier (train vs test)...", flush=True)
X_adv = np.vstack([X_tr, X_te])
y_adv = np.concatenate([np.zeros(len(X_tr), dtype=np.int8), np.ones(len(X_te), dtype=np.int8)])
# simple 3-fold to get OOF proba for train part (we only need train proba)
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
adv_oof = np.zeros(len(X_adv))
for fi, (tri, vai) in enumerate(skf.split(X_adv, y_adv)):
    m = lgb.LGBMClassifier(
        n_estimators=500, learning_rate=0.05, num_leaves=31, max_depth=6,
        min_child_samples=100, subsample=0.7, colsample_bytree=0.7,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    m.fit(X_adv[tri], y_adv[tri],
          eval_set=[(X_adv[vai], y_adv[vai])],
          callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])
    adv_oof[vai] = m.predict_proba(X_adv[vai])[:, 1]
auc = roc_auc_score(y_adv, adv_oof)
print(f"  adversarial AUC = {auc:.4f}", flush=True)

# train P(test|x)
p_test = adv_oof[:len(X_tr)]
print(f"  P(test|x_train) dist: min={p_test.min():.4f} max={p_test.max():.4f} mean={p_test.mean():.4f}", flush=True)

# importance weights: w = P(test) / P(train) = p / (1-p)
# clip to avoid extreme values
eps = 1e-3
p_clipped = np.clip(p_test, eps, 1-eps)
weights_raw = p_clipped / (1 - p_clipped)
# normalize to mean=1
weights_norm = weights_raw / weights_raw.mean()
# cap extremes
weights_capped = np.clip(weights_norm, 0.1, 5.0)
print(f"  weights after norm+cap: min={weights_capped.min():.3f} max={weights_capped.max():.3f} mean={weights_capped.mean():.3f}", flush=True)

del X_adv, y_adv, adv_oof
gc.collect()

# ----- Step 2: Train LGB-Huber with adversarial weights -----
def train_variant(weights, name):
    print(f"\n--- {name} ---", flush=True)
    oof = np.zeros(len(y))
    PARAMS = dict(
        objective="huber", n_estimators=5000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1,
    )
    for f in range(5):
        t1 = time.time()
        tr = np.where(fold_ids != f)[0]
        val = np.where(fold_ids == f)[0]
        m = lgb.LGBMRegressor(**PARAMS)
        m.fit(X_tr[tr], y_log[tr], sample_weight=weights[tr] if weights is not None else None,
              eval_set=[(X_tr[val], y_log[val])],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        oof[val] = np.clip(np.expm1(m.predict(X_tr[val])), 0, None)
        print(f"  {name} fold {f}: mae={mean_absolute_error(y[val], oof[val]):.5f} it={m.best_iteration_} ({time.time()-t1:.0f}s)", flush=True)
        del m
        gc.collect()
    mae = mean_absolute_error(y, oof)
    np.save(f"{OUT}/oof_{name}.npy", oof)
    return oof, mae

# baseline reuse
oof_A = np.load(f"{OUT}/oof_raw_y_log1p_huber.npy")
mae_A = mean_absolute_error(y, oof_A)
print(f"\nA baseline reused: {mae_A:.5f}", flush=True)

oof_adv, mae_adv = train_variant(weights_capped, "adv_weighted")
delta = mae_adv - mae_A
print(f"\nadv_weighted OOF: {mae_adv:.5f} delta vs A={delta:+.5f}", flush=True)

# ----- Step 3: Diversity analysis -----
r_mega = y - mega33_oof
r_adv = y - oof_adv
r_base = y - oof_A
corr_adv = np.corrcoef(r_mega, r_adv)[0, 1]
corr_base = np.corrcoef(r_mega, r_base)[0, 1]
print(f"\nresidual correlation with mega33:", flush=True)
print(f"  baseline (no weight): {corr_base:.4f}", flush=True)
print(f"  adv_weighted:         {corr_adv:.4f}", flush=True)

# ----- Step 4: Blend scan with mega33 -----
print(f"\nBlend scan (mega33 + adv_weighted):", flush=True)
best = (None, baseline_mega, 0)
for w in [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
    b = (1-w)*mega33_oof + w*oof_adv
    mae = mean_absolute_error(y, b)
    d = mae - baseline_mega
    if mae < best[1]:
        best = (w, mae, d)
    tag = " ***" if d < -0.003 else (" **" if d < -0.001 else "")
    print(f"  w={w:.2f}: mae={mae:.5f} delta={d:+.5f}{tag}", flush=True)

# Multi-blend with pseudo (combining adv + pseudo diversity axes)
oof_pseudo = np.load(f"{OUT}/oof_pseudo_all_w1.0.npy")
from scipy.optimize import minimize
def obj(w):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-6: return 99
    w = w / w.sum()
    blend = w[0]*mega33_oof + w[1]*oof_adv + w[2]*oof_pseudo
    return mean_absolute_error(y, blend)
res = minimize(obj, [0.8, 0.1, 0.1], method="Nelder-Mead", options={"xatol": 1e-6, "maxiter": 10000})
w_opt = np.clip(res.x, 0, None); w_opt = w_opt / w_opt.sum()
print(f"\nMulti-blend (mega33, adv, pseudo):", flush=True)
print(f"  opt weights: {w_opt.round(4)}", flush=True)
print(f"  mae={res.fun:.5f} delta={res.fun - baseline_mega:+.5f}", flush=True)

verdict = "STRONG_GO" if best[2] < -0.01 else ("GO" if best[2] < -0.003 else ("WEAK" if best[2] < 0 else "NO_GO"))
print(f"\n>>> VERDICT: {verdict} best_blend_delta={best[2]:+.5f} corr_gain={corr_base - corr_adv:+.4f} <<<", flush=True)
print(f"elapsed: {time.time()-t0:.0f}s", flush=True)

json.dump({
    "verdict": verdict,
    "adv_auc": float(auc),
    "adv_weighted_oof_mae": float(mae_adv),
    "delta_vs_baseline_A": float(delta),
    "residual_corr_mega33_base": float(corr_base),
    "residual_corr_mega33_adv": float(corr_adv),
    "corr_reduction": float(corr_base - corr_adv),
    "best_blend_w": float(best[0]) if best[0] else None,
    "best_blend_mae": float(best[1]),
    "best_blend_delta": float(best[2]),
    "multi_blend_weights": w_opt.tolist(),
    "multi_blend_mae": float(res.fun),
    "multi_blend_delta": float(res.fun - baseline_mega),
    "elapsed": round(time.time()-t0, 1),
}, open(f"{OUT}/adversarial_summary.json", "w"), indent=2)
