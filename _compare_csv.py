"""
원본 make_best_submission.py의 float64 예측값 vs
embedded notebook의 float32 복원값 비교
"""
import numpy as np
import pandas as pd
import pickle
import base64
import io
import json
import sys
sys.stdout.reconfigure(encoding='utf-8')

PROJ = r'C:\Users\user\Desktop\데이콘 4월'

# ─── 1. 원본 float64 예측값 재계산 (make_best_submission.py 동일 로직) ───
fw = dict(
    mega33   = 0.7636614598089654,
    rank_adj = 0.1588758398901156,
    iter_r1  = 0.011855567572749024,
    iter_r2  = 0.034568307,
    iter_r3  = 0.031038826,
)

def load(rel):
    return np.load(f"{PROJ}/{rel}")

with open(f"{PROJ}/results/mega33_final.pkl", "rb") as f:
    d = pickle.load(f)

train_raw = pd.read_csv(f"{PROJ}/train.csv")
test_raw  = pd.read_csv(f"{PROJ}/test.csv")
train_raw["_row_id"] = train_raw["ID"].str.replace("TRAIN_", "").astype(int)
test_raw["_row_id"]  = test_raw["ID"].str.replace("TEST_",  "").astype(int)
train_rid = train_raw.sort_values("_row_id").reset_index(drop=True)
test_rid  = test_raw.sort_values("_row_id").reset_index(drop=True)
y_true = train_rid["avg_delay_minutes_next_30m"].values

train_ls = train_raw.sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
test_ls  = test_raw.sort_values(["layout_id","scenario_id"]).reset_index(drop=True)
ls_pos    = {row["ID"]: i for i, row in train_ls.iterrows()}
te_ls_pos = {row["ID"]: i for i, row in test_ls.iterrows()}
id2    = np.array([ls_pos[i]    for i in train_rid["ID"].values])
te_id2 = np.array([te_ls_pos[i] for i in test_rid["ID"].values])

fixed_oof_f64 = (
    fw["mega33"]   * d["meta_avg_oof"][id2]
  + fw["rank_adj"] * load("results/ranking/rank_adj_oof.npy")[id2]
  + fw["iter_r1"]  * load("results/iter_pseudo/round1_oof.npy")[id2]
  + fw["iter_r2"]  * load("results/iter_pseudo/round2_oof.npy")[id2]
  + fw["iter_r3"]  * load("results/iter_pseudo/round3_oof.npy")[id2]
)
fixed_test_f64 = (
    fw["mega33"]   * d["meta_avg_test"][te_id2]
  + fw["rank_adj"] * load("results/ranking/rank_adj_test.npy")[te_id2]
  + fw["iter_r1"]  * load("results/iter_pseudo/round1_test.npy")[te_id2]
  + fw["iter_r2"]  * load("results/iter_pseudo/round2_test.npy")[te_id2]
  + fw["iter_r3"]  * load("results/iter_pseudo/round3_test.npy")[te_id2]
)
oxgb_oof_f64  = load("results/oracle_seq/oof_seqC_xgb.npy")
oxgb_test_f64 = load("results/oracle_seq/test_C_xgb.npy")
olv2_oof_f64  = load("results/oracle_seq/oof_seqC_log_v2.npy")
olv2_test_f64 = load("results/oracle_seq/test_C_log_v2.npy")
orem_oof_f64  = load("results/oracle_seq/oof_seqC_xgb_remaining.npy")
orem_test_f64 = load("results/oracle_seq/test_C_xgb_remaining.npy")

oracle_NEW_oof_f64  = 0.64*fixed_oof_f64 + 0.12*oxgb_oof_f64 + 0.16*olv2_oof_f64 + 0.08*orem_oof_f64
oracle_NEW_test_f64 = np.maximum(0,
    0.64*fixed_test_f64 + 0.12*oxgb_test_f64 + 0.16*olv2_test_f64 + 0.08*orem_test_f64)

# ─── 2. float32 복원값 (embedded notebook과 동일) ───────────────
with open(f"{PROJ}/code_share/solution.ipynb", encoding='utf-8') as f:
    nb = json.load(f)

# Cell 7 source에서 base64 문자열 추출
cell7_src = ''.join(nb['cells'][7]['source'])

def extract_b64(src, varname):
    prefix = f'{varname}  = _b64dec("'
    if prefix not in src:
        prefix = f'{varname} = _b64dec("'
    if prefix not in src:
        prefix = f'{varname}= _b64dec("'
    start = src.index(prefix) + len(prefix)
    end = src.index('")', start)
    return src[start:end]

def decode_f32(b64str):
    return np.load(io.BytesIO(base64.b64decode(b64str))).astype(np.float64)

fixed_oof_f32  = decode_f32(extract_b64(cell7_src, 'fixed_oof'))
fixed_test_f32 = decode_f32(extract_b64(cell7_src, 'fixed_test'))
oxgb_oof_f32   = decode_f32(extract_b64(cell7_src, 'oracle_xgb_oof'))
oxgb_test_f32  = decode_f32(extract_b64(cell7_src, 'oracle_xgb_test'))
olv2_oof_f32   = decode_f32(extract_b64(cell7_src, 'oracle_lv2_oof'))
olv2_test_f32  = decode_f32(extract_b64(cell7_src, 'oracle_lv2_test'))
orem_oof_f32   = decode_f32(extract_b64(cell7_src, 'oracle_rem_oof'))
orem_test_f32  = decode_f32(extract_b64(cell7_src, 'oracle_rem_test'))

oracle_NEW_oof_f32  = 0.64*fixed_oof_f32 + 0.12*oxgb_oof_f32 + 0.16*olv2_oof_f32 + 0.08*orem_oof_f32
oracle_NEW_test_f32 = np.maximum(0,
    0.64*fixed_test_f32 + 0.12*oxgb_test_f32 + 0.16*olv2_test_f32 + 0.08*orem_test_f32)

# ─── 3. 비교 ─────────────────────────────────────────────────────
from sklearn.metrics import mean_absolute_error

mae_f64 = mean_absolute_error(y_true, oracle_NEW_oof_f64)
mae_f32 = mean_absolute_error(y_true, oracle_NEW_oof_f32)

diff_test = np.abs(oracle_NEW_test_f64 - oracle_NEW_test_f32)

print("="*55)
print("float64(원본) vs float32(내장) 예측값 비교")
print("="*55)
print(f"\n[OOF MAE]")
print(f"  float64 원본 : {mae_f64:.6f}  → 파일명: {mae_f64:.4f}")
print(f"  float32 내장 : {mae_f32:.6f}  → 파일명: {mae_f32:.4f}")
print(f"  파일명 동일  : {f'{mae_f64:.4f}' == f'{mae_f32:.4f}'}")

print(f"\n[test 예측값 차이 (50,000행)]")
print(f"  최대 절대 오차 : {diff_test.max():.2e}")
print(f"  평균 절대 오차 : {diff_test.mean():.2e}")
print(f"  95th percentile: {np.percentile(diff_test, 95):.2e}")

print(f"\n[결론]")
if f'{mae_f64:.4f}' == f'{mae_f32:.4f}' and diff_test.max() < 0.001:
    print("  CSV 파일명 동일, 예측값 차이 < 0.001분 → 사실상 동일")
    print("  데이콘 재현 기준 충족 ✅")
else:
    print("  ❌ 차이 있음 — float64 직접 인코딩 필요")
