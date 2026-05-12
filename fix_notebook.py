"""
fix_notebook.py
노트북 오류 수정 스크립트
"""
import json

NB_PATH = 'code_share/solution.ipynb'

with open(NB_PATH, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
fixed_count = 0


def fix_cell(idx, old, new, desc):
    global fixed_count
    src = ''.join(cells[idx]['source'])
    if old not in src:
        print(f"  [WARN] Cell {idx} ({desc}): old string NOT found — skipping")
        return
    new_src = src.replace(old, new, 1)
    # Re-split into lines preserving newlines (notebook format)
    lines = new_src.splitlines(keepends=True)
    cells[idx]['source'] = lines
    fixed_count += 1
    print(f"  [OK]   Cell {idx} ({desc}): fixed")


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: Cell 0 (markdown) — 파이프라인 공식 수정
# ─────────────────────────────────────────────────────────────────────────────
fix_cell(
    0,
    '[최종 블렌드: mega33 × 0.76 + rank_adj × 0.16 + oracle_seq × 0.36]',
    '[최종 블렌드: FIXED = mega33×0.7637 + rank_adj×0.1589 + iter_r1~r3\noracle_NEW = 0.64×FIXED + 0.12×oracle_xgb + 0.16×oracle_lv2 + 0.08×oracle_rem]',
    'markdown pipeline formula'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: Cell 10 (code) — lag_cols 잘못된 컬럼 수정
# ─────────────────────────────────────────────────────────────────────────────
fix_cell(
    10,
    "    lag_cols = ['order_inflow_15m', 'pack_utilization', 'congestion_score',\n                'robot_utilization', 'warehouse_temp', 'humidity']",
    "    lag_cols = ['order_inflow_15m', 'pack_utilization', 'congestion_score',\n                'robot_utilization', 'battery_mean', 'fault_count_15m',\n                'blocked_path_15m', 'charge_queue_length']",
    'build_features_v31 lag_cols'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 3a: Cell 19 (code) — 저장 경로 수정
# ─────────────────────────────────────────────────────────────────────────────
fix_cell(
    19,
    "with open('results/mega33/mega33_final.pkl', 'wb') as f:\n    pickle.dump({'meta_avg_oof': mega33_oof}, f)",
    "with open('results/mega33_final.pkl', 'wb') as f:\n    pickle.dump({'meta_avg_oof': mega33_oof}, f)",
    'cell 19 save path'
)

# Fix 3b: Cell 19 (code) — oof_meta_cb 미학습 → CatBoost meta 학습 코드 추가
# 기존: print line 직전 / loop 내 CB 학습 없음 → loop 내부에 CB 추가
fix_cell(
    19,
    "    print(f\"  Meta fold{f_i+1}: LGB={mean_absolute_error(y[val_idx], oof_meta_lgb[val_idx]):.4f}  \"\n          f\"XGB={mean_absolute_error(y[val_idx], oof_meta_xgb[val_idx]):.4f}\")",
    """    # Meta CB
    m_cb = CatBoostRegressor(
        loss_function='MAE', iterations=500, learning_rate=0.05,
        depth=4, min_data_in_leaf=100, subsample=0.8, rsm=0.8,
        l2_leaf_reg=3.0, random_seed=SEED, verbose=0,
        early_stopping_rounds=50, task_type='CPU', thread_count=4
    )
    m_cb.fit(stack_train[tr_idx], y_log[tr_idx],
             eval_set=(stack_train[val_idx], y_log[val_idx]), use_best_model=True)
    oof_meta_cb[val_idx] = np.expm1(m_cb.predict(stack_train[val_idx]))

    print(f"  Meta fold{f_i+1}: LGB={mean_absolute_error(y[val_idx], oof_meta_lgb[val_idx]):.4f}  "
          f"XGB={mean_absolute_error(y[val_idx], oof_meta_xgb[val_idx]):.4f}  "
          f"CB={mean_absolute_error(y[val_idx], oof_meta_cb[val_idx]):.4f}")""",
    'cell 19 CatBoost meta training'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 4: Cell 21 (code) — 로드 경로 수정
# ─────────────────────────────────────────────────────────────────────────────
fix_cell(
    21,
    "with open('results/mega33/mega33_final.pkl','rb') as f:",
    "with open('results/mega33_final.pkl','rb') as f:",
    'cell 21 load path'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 5: Cell 22 (code) — Oracle 학습 로직 수정
# ─────────────────────────────────────────────────────────────────────────────

# 5a: make_X_oracle — lag3 파라미터 제거 (2 lags + row_sc only)
fix_cell(
    22,
    "def make_X_oracle(base_feats, lag1, lag2, lag3, row_sc):\n    \"\"\"base 피처 + oracle lag 피처 결합\"\"\"\n    extra = np.column_stack([lag1, lag2, lag3, row_sc])\n    return np.hstack([base_feats, extra])",
    "def make_X_oracle(base_feats, lag1, lag2, row_sc):\n    \"\"\"base 피처 + oracle lag 피처 결합\"\"\"\n    extra = np.column_stack([lag1, lag2, row_sc])\n    return np.hstack([base_feats, extra])",
    'cell 22 make_X_oracle signature (remove lag3)'
)

# 5b: X_tr_oracle 호출 — lag3 제거
fix_cell(
    22,
    "    X_tr_oracle = make_X_oracle(\n        X_base_train[tr_idx],\n        train['lag1_y'].values[tr_idx],\n        train['lag2_y'].values[tr_idx],\n        train['lag3_y'].values[tr_idx],\n        train['row_in_sc'].values[tr_idx]\n    )",
    "    X_tr_oracle = make_X_oracle(\n        X_base_train[tr_idx],\n        train['lag1_y'].values[tr_idx],\n        train['lag2_y'].values[tr_idx],\n        train['row_in_sc'].values[tr_idx]\n    )",
    'cell 22 X_tr_oracle call (remove lag3)'
)

# 5b: model.fit — log1p → raw y
fix_cell(
    22,
    "    model.fit(X_tr_oracle, np.log1p(y_true[tr_idx]),\n              eval_set=[(X_val_proxy, np.log1p(y_true[val_idx]))],\n              verbose=False)",
    "    model.fit(X_tr_oracle, y_true[tr_idx],\n              eval_set=[(X_val_proxy, y_true[val_idx])],\n              verbose=False)",
    'cell 22 model.fit raw y (not log1p)'
)

# 5b: X_val_proxy — lag3 proxy 제거 (lag1, lag2만)
fix_cell(
    22,
    "    X_val_proxy = make_X_oracle(\n        X_base_train[val_idx],\n        train['lag1_mega'].values[val_idx],\n        train['lag2_mega'].values[val_idx],\n        train['lag1_mega'].values[val_idx],   # lag3 → lag1 proxy로 대체\n        train['row_in_sc'].values[val_idx]\n    )",
    "    X_val_proxy = make_X_oracle(\n        X_base_train[val_idx],\n        train['lag1_mega'].values[val_idx],\n        train['lag2_mega'].values[val_idx],\n        train['row_in_sc'].values[val_idx]\n    )",
    'cell 22 X_val_proxy (remove lag3)'
)

# 5c: val 순차 예측 내부 — lag3 변수 및 make_X_oracle 호출 수정
fix_cell(
    22,
    "        lag1 = hist[-1] if len(hist) >= 1 else global_mean\n        lag2 = hist[-2] if len(hist) >= 2 else global_mean\n        lag3 = hist[-3] if len(hist) >= 3 else global_mean\n        x = make_X_oracle(\n            X_base_train[idx:idx+1], [lag1], [lag2], [lag3], [pos]\n        )\n        pred_val = float(np.expm1(model.predict(x)[0]))",
    "        lag1 = hist[-1] if len(hist) >= 1 else global_mean\n        lag2 = hist[-2] if len(hist) >= 2 else global_mean\n        x = make_X_oracle(\n            X_base_train[idx:idx+1], [lag1], [lag2], [pos]\n        )\n        pred_val = float(model.predict(x)[0])",
    'cell 22 val sequential pred (lag2 only, no expm1)'
)

# ─────────────────────────────────────────────────────────────────────────────
# Fix 6: Cell 27 (code) — 파일 경로 수정
# ─────────────────────────────────────────────────────────────────────────────

# 6a: mega_test — results/mega33/mega33_test.npy → pkl에서 로드
fix_cell(
    27,
    "mega_test = np.load('results/mega33/mega33_test.npy')[te_id2]",
    "with open('results/mega33_final.pkl', 'rb') as _f:\n    d_final = pickle.load(_f)\nmega_test = d_final['meta_avg_test'][te_id2]",
    'cell 27 mega_test load from pkl'
)

# 6b: oracle_seq 파일명 seqC → C
fix_cell(
    27,
    "xgb_test  = np.load('results/oracle_seq/test_seqC_xgb.npy')",
    "xgb_test  = np.load('results/oracle_seq/test_C_xgb.npy')",
    'cell 27 xgb_test path'
)

fix_cell(
    27,
    "lv2_test  = np.load('results/oracle_seq/test_seqC_log_v2.npy')",
    "lv2_test  = np.load('results/oracle_seq/test_C_log_v2.npy')",
    'cell 27 lv2_test path'
)

fix_cell(
    27,
    "rem_test  = np.load('results/oracle_seq/test_seqC_xgb_remaining.npy')",
    "rem_test  = np.load('results/oracle_seq/test_C_xgb_remaining.npy')",
    'cell 27 rem_test path'
)

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
with open(NB_PATH, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"\nFixed {fixed_count} cells")
