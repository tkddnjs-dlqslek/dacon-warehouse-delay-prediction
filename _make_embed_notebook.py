"""
solution.ipynb → 배열 내장(self-contained) 버전 생성
pkl/npy 파일 없이 train.csv/test.csv만으로 submission CSV 재현 가능
"""
import numpy as np
import pandas as pd
import pickle
import base64
import io
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')

PROJ   = r'C:\Users\user\Desktop\데이콘 4월'
OUT    = rf'{PROJ}\code_share\solution.ipynb'
TARGET = "avg_delay_minutes_next_30m"

# ─── Step 1: 배열 계산 ───────────────────────────────────────────
print("배열 로드 중...")
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

train_ls = train_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
test_ls  = test_raw.sort_values(["layout_id", "scenario_id"]).reset_index(drop=True)
ls_pos    = {row["ID"]: i for i, row in train_ls.iterrows()}
te_ls_pos = {row["ID"]: i for i, row in test_ls.iterrows()}
id2    = np.array([ls_pos[i]    for i in train_rid["ID"].values])
te_id2 = np.array([te_ls_pos[i] for i in test_rid["ID"].values])

fixed_oof = (
    fw["mega33"]   * d["meta_avg_oof"][id2]
  + fw["rank_adj"] * load("results/ranking/rank_adj_oof.npy")[id2]
  + fw["iter_r1"]  * load("results/iter_pseudo/round1_oof.npy")[id2]
  + fw["iter_r2"]  * load("results/iter_pseudo/round2_oof.npy")[id2]
  + fw["iter_r3"]  * load("results/iter_pseudo/round3_oof.npy")[id2]
)
fixed_test = (
    fw["mega33"]   * d["meta_avg_test"][te_id2]
  + fw["rank_adj"] * load("results/ranking/rank_adj_test.npy")[te_id2]
  + fw["iter_r1"]  * load("results/iter_pseudo/round1_test.npy")[te_id2]
  + fw["iter_r2"]  * load("results/iter_pseudo/round2_test.npy")[te_id2]
  + fw["iter_r3"]  * load("results/iter_pseudo/round3_test.npy")[te_id2]
)
oracle_xgb_oof  = load("results/oracle_seq/oof_seqC_xgb.npy")
oracle_xgb_test = load("results/oracle_seq/test_C_xgb.npy")
oracle_lv2_oof  = load("results/oracle_seq/oof_seqC_log_v2.npy")
oracle_lv2_test = load("results/oracle_seq/test_C_log_v2.npy")
oracle_rem_oof  = load("results/oracle_seq/oof_seqC_xgb_remaining.npy")
oracle_rem_test = load("results/oracle_seq/test_C_xgb_remaining.npy")

print("배열 계산 완료")

# ─── Step 2: base64 인코딩 (float32로 저장) ─────────────────────
def enc(arr):
    buf = io.BytesIO()
    np.save(buf, arr.astype(np.float32))
    return base64.b64encode(buf.getvalue()).decode('ascii')

print("base64 인코딩 중...")
B = {
    'fixed_oof':       enc(fixed_oof),
    'fixed_test':      enc(fixed_test),
    'oracle_xgb_oof':  enc(oracle_xgb_oof),
    'oracle_xgb_test': enc(oracle_xgb_test),
    'oracle_lv2_oof':  enc(oracle_lv2_oof),
    'oracle_lv2_test': enc(oracle_lv2_test),
    'oracle_rem_oof':  enc(oracle_rem_oof),
    'oracle_rem_test': enc(oracle_rem_test),
}
for k, v in B.items():
    print(f"  {k:20s}: {len(v)/1024:6.0f} KB")
total_mb = sum(len(v) for v in B.values()) / 1024 / 1024
print(f"  총 base64 크기: {total_mb:.1f} MB")

# ─── Step 3: 셀 빌더 ─────────────────────────────────────────────
def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [text], "outputs": [], "execution_count": None}

def code(text, outputs=None):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "source": [text], "outputs": outputs or []}

# ─── Step 4: 셀 구성 ─────────────────────────────────────────────
cells = []

# Cell 0: 제목
cells.append(md(
"""# 스마트 창고 출고 지연 예측 — 솔루션 노트북

**대회**: 데이콘 스마트 창고 출고 지연 예측 경진대회 (2026.04)
**결과**: Public LB 9.7527 (13위) / Private LB 10.01261 (17위) / 608팀 (상위 2.8%)
**평가 지표**: MAE

---

## 전체 파이프라인

```
train.csv + test.csv
    |
    +--[Section 1] Mega33 앙상블 (33개 Base → 3 Meta → FIXED base)
    |      33개 Base 모델 (v23/v24/v26 x LGB/XGB/CB x 다중 시드)
    |      + Ranking 보정 + Pseudo-Labeling 3 rounds
    |      → FIXED = mega33 x 0.7637 + rank_adj x 0.1589
    |              + pseudo_r1 x 0.0119 + pseudo_r2 x 0.0346 + pseudo_r3 x 0.0310
    |
    +--[Section 2] Sequential Oracle 모델 (FIXED OOF를 proxy lag으로 재학습)
    |      oracle_xgb / oracle_lv2 / oracle_remaining
    |
    +--[Section 3] 최종 블렌드 (oracle_NEW)
           oracle_NEW = FIXED x 0.64 + oracle_xgb x 0.12
                      + oracle_lv2 x 0.16 + oracle_remaining x 0.08
           OOF MAE: 8.3825  |  Public LB: 9.7527
```

---

## 개발 환경

```
OS      : Windows 11 Home
Python  : 3.10+ (Anaconda)
lightgbm==4.6.0  xgboost==3.0.4  catboost==1.2.8
scikit-learn==1.2.2  numpy==1.26.4  pandas==2.2.3
```

## 입력 파일 (노트북과 같은 폴더)

```
train.csv  /  test.csv  /  sample_submission.csv
```

> 모델 학습 결과(33개 base + 3 oracle)는 노트북에 base64로 내장됨 (데이콘 파일 형식 제한 대응).
> 원본 모델 파일: `results/mega33_final.pkl`, `results/oracle_seq/*.npy` 등 18개.

**예상 실행 시간**: 1분 미만"""
))

# Cell 1: 라이브러리 헤더
cells.append(md("## 라이브러리 임포트"))

# Cell 2: 임포트
cells.append(code(
"""import numpy as np
import pandas as pd
import base64
import io
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

TARGET = "avg_delay_minutes_next_30m"
print("라이브러리 임포트 완료")"""
))

# Cell 3: 문제 설명
cells.append(md(
"""## Section 0: 문제 및 핵심 도전 과제

AMR 로봇 물류창고 시뮬레이션 데이터에서 **향후 30분간 평균 출고 지연(분)을 예측**하는 회귀 문제.

```
Train: 300 layouts x 10,000 scenarios x 25 timeslots = 250,000 rows
Test : 100 layouts x  2,000 scenarios x 25 timeslots =  50,000 rows
```

**핵심 난점: test layouts 절반(50개)이 train에 없는 unseen layout**

| 구분 | Order Inflow | Congestion |
|------|-------------|------------|
| Test seen layouts   | +14% vs train | +13% |
| Test unseen layouts | +87% vs train | +45% |

이 분포 차이가 OOF-LB 갭의 주요 원인. seen layout에만 최적화되는 접근은 전부 LB 악화로 이어짐.

**CV 전략**: `GroupKFold(layout_id, n=5)` — val fold에 unseen layout이 자연스럽게 포함되어
GroupKFold(scenario_id), Temporal Blocked CV 대비 LB 일반화가 가장 우수함을 실험으로 확인."""
))

# Cell 4: 데이터 로드 헤더
cells.append(md(
"""## Section 1: 데이터 로드

원래 파이프라인에서는 피처 엔지니어링을 `(layout_id, scenario_id)` 정렬(ls order)로 수행하고,
타겟/제출은 ID 숫자 정렬(rid order)을 기준으로 했습니다.
이 노트북에서는 모델 학습 결과가 내장되어 있으므로, CSV 파일에서
**타겟(y_true)**, **ID 순서**, **seen/unseen layout 구분**만 추출합니다."""
))

# Cell 5: 데이터 로드 코드
cells.append(code(
"""train_raw  = pd.read_csv("train.csv")
test_raw   = pd.read_csv("test.csv")
sample_sub = pd.read_csv("sample_submission.csv")

# rid order: ID 숫자 기준 정렬
train_raw["_row_id"] = train_raw["ID"].str.replace("TRAIN_", "").astype(int)
test_raw["_row_id"]  = test_raw["ID"].str.replace("TEST_",  "").astype(int)
train_rid = train_raw.sort_values("_row_id").reset_index(drop=True)
test_rid  = test_raw.sort_values("_row_id").reset_index(drop=True)
y_true    = train_rid[TARGET].values

# seen / unseen layout 구분
train_layouts = set(train_raw["layout_id"].unique())
test_layouts  = set(test_raw["layout_id"].unique())
seen_test     = test_layouts & train_layouts
unseen_test   = test_layouts - train_layouts

print(f"train: {train_raw.shape}  /  test: {test_raw.shape}")
print(f"train layouts: {len(train_layouts)}  /  test layouts: {len(test_layouts)}")
print(f"  seen: {len(seen_test)},  unseen: {len(unseen_test)}")
print(f"target  mean={y_true.mean():.3f}  std={y_true.std():.3f}")"""
))

# Cell 6: Mega33 + Oracle 설명
cells.append(md(
"""## Section 2: 사전 학습 결과 (Mega33 + FIXED + Oracle)

### Mega33 Base 모델 구성 (33개)

| 피처 버전 | 모델 종류 | 시드 | 수 |
|-----------|----------|------|----|
| v23 (lag/lead/rolling/SC 집계, ~150개) | LGB(Huber) / XGB / CatBoost | 42, 123, 2024 | 9 |
| v24 (v23 + SC 확장 + 분위수 집계) | LGB(MAE) / LGB(Huber) / XGB / CatBoost | 42 | 4 |
| v26 (v23 + 고급 튜닝 변형) | Tuned_Huber / Tuned_sqrt / Tuned_pow / DART | 42 | 4 |
| domain_aug (v23 + 도메인 피처) | LGB(Huber) / XGB / CatBoost | 42 | 3 |
| offset (시나리오 평균 + 잔차 분해) | LGB(Huber) / LGB(MAE) / XGB | 42 | 3 |
| neural_army (시계열 신경망) | BiLSTM / CNN / MLP_wide / MLP_deep / MLP_resnet / TCN | — | 6 |
| **합계** | | | **33** |

모든 base 모델: `GroupKFold(layout_id, n=5)`, 타겟 `log1p(y)` 변환 후 학습.

### FIXED base 구성

33 base OOF → Level-2 Meta (LGB/XGB/CB) 스태킹 → `mega33`
+ Ranking 보정 (`rank_adj`) + Pseudo-Labeling 3 rounds

```
FIXED = mega33 × 0.7637 + rank_adj × 0.1589
      + round1 × 0.0119 + round2 × 0.0346 + round3 × 0.0310
```

### Sequential Oracle (3종)

FIXED OOF를 proxy lag으로 추가해 재학습한 auto-regressive XGB.

| 이름 | 피처셋 | 손실 | OOF MAE | corr(FIXED) |
|------|--------|------|---------|-------------|
| oracle_xgb | v31(335개) + lag1/2 + row_in_sc | MAE | 8.43846 | 0.9733 |
| oracle_lv2 | v31 + lag1/2 + log1p 타겟 | MAE | 8.44093 | 0.9721 |
| oracle_remaining | v31 서브셋 (pack 계열 제외) + lag1/2 | MAE | 8.47155 | 0.9760 |

세 oracle 모두 `corr(FIXED) ≈ 0.97` — FIXED 대비 다양성은 제한적이나 oracle 간 corr ≈ 0.90~0.92로 상호 보완.

> 이하 코드 셀에서 위 학습 결과를 base64로 복원합니다."""
))

# Cell 7: 내장 배열 복원
embed_code = f"""# 사전 학습 결과 복원 (base64 → numpy float32 → float64)
# 원본: results/mega33_final.pkl + results/**/*.npy 18개 파일
def _b64dec(s):
    return np.load(io.BytesIO(base64.b64decode(s))).astype(np.float64)

fixed_oof  = _b64dec("{B['fixed_oof']}")
fixed_test = _b64dec("{B['fixed_test']}")

oracle_xgb_oof  = _b64dec("{B['oracle_xgb_oof']}")
oracle_xgb_test = _b64dec("{B['oracle_xgb_test']}")

oracle_lv2_oof  = _b64dec("{B['oracle_lv2_oof']}")
oracle_lv2_test = _b64dec("{B['oracle_lv2_test']}")

oracle_rem_oof  = _b64dec("{B['oracle_rem_oof']}")
oracle_rem_test = _b64dec("{B['oracle_rem_test']}")

print(f"FIXED  OOF MAE : {{mean_absolute_error(y_true, fixed_oof):.5f}}")
print(f"배열 복원 완료  (float32 저장 → float64 복원, 정밀도 손실 < 1e-6)")"""

cells.append(code(embed_code))

# Cell 8: 블렌드 설명
cells.append(md(
"""## Section 3: 최종 블렌드 (oracle_NEW) 및 제출 파일 생성

OOF 기준 grid search로 탐색한 최적 블렌드:

```
oracle_NEW = FIXED       x 0.64
           + oracle_xgb  x 0.12
           + oracle_lv2  x 0.16
           + oracle_rem  x 0.08
```

OOF MAE: **8.3825**  |  Public LB: **9.7527** (13위)  |  Private LB: **10.01261** (17위)

### 주요 실패 실험 (OOF 개선 ≠ LB 개선)

| 시도 | OOF | LB | 결론 |
|------|-----|----|------|
| Gate (seen/unseen 분기 최적화) | 개선 | 악화 | seen 과적합 |
| Scipy 가중치 최적화 | 개선 | 미제출 | OOF 과적합 |
| Temporal CV (sc_num 기준) | 8.63 | blend 불가 | corr=0.957, GroupKFold가 실제로 더 일반화됨 |
| test_unseen 보정 (+5.5) | — | 악화 | Δunseen 상승 → LB 파국적 악화 |
| M/M/1 물리 모델 | 14.47 | blend 불가 | 단일 변수로 한계 |
| neural_army blend | 9.77+ | 악화 | unseen 과적합 |

**결론**: 250개 이상 실험 중 LB를 개선한 접근은 oracle_NEW 블렌드뿐."""
))

# Cell 9: Oracle MAE + corr 출력
cells.append(code(
"""print("Oracle OOF MAE:")
print(f"  oracle_xgb      : {mean_absolute_error(y_true, oracle_xgb_oof):.5f}")
print(f"  oracle_lv2      : {mean_absolute_error(y_true, oracle_lv2_oof):.5f}")
print(f"  oracle_remaining: {mean_absolute_error(y_true, oracle_rem_oof):.5f}")

print("\\ncorr(FIXED, oracle):")
print(f"  oracle_xgb      : {np.corrcoef(fixed_oof, oracle_xgb_oof)[0,1]:.4f}")
print(f"  oracle_lv2      : {np.corrcoef(fixed_oof, oracle_lv2_oof)[0,1]:.4f}")
print(f"  oracle_remaining: {np.corrcoef(fixed_oof, oracle_rem_oof)[0,1]:.4f}")
print("  (oracle 간 corr ≈ 0.90~0.92, FIXED 대비는 0.97 수준)")"""
))

# Cell 10: 블렌드 + 제출 코드
cells.append(code(
"""# oracle_NEW 블렌드 가중치
w_fixed = 0.64
w_xgb   = 0.12
w_lv2   = 0.16
w_rem   = 0.08
print(f"가중치 합계: {w_fixed+w_xgb+w_lv2+w_rem:.2f}")

# OOF 블렌드
oracle_NEW_oof = (w_fixed * fixed_oof
                + w_xgb   * oracle_xgb_oof
                + w_lv2   * oracle_lv2_oof
                + w_rem   * oracle_rem_oof)

# test 블렌드
oracle_NEW_test = np.maximum(0, (
    w_fixed * fixed_test
  + w_xgb   * oracle_xgb_test
  + w_lv2   * oracle_lv2_test
  + w_rem   * oracle_rem_test
))

mae = mean_absolute_error(y_true, oracle_NEW_oof)
print(f"\\noracle_NEW OOF MAE: {mae:.6f}  (제출 파일명 기준: {mae:.4f})")

# seen / unseen 예측 분포
seen_mask   = test_rid["layout_id"].isin(seen_test)
unseen_mask = test_rid["layout_id"].isin(unseen_test)
print(f"\\ntest 예측 분포:")
print(f"  전체:   mean={oracle_NEW_test.mean():.3f}")
print(f"  seen:   mean={oracle_NEW_test[seen_mask].mean():.3f}")
print(f"  unseen: mean={oracle_NEW_test[unseen_mask].mean():.3f}")

# 제출 파일 생성 (ID 매핑으로 순서 보장)
sub = sample_sub[["ID"]].copy()
sub[TARGET] = sub["ID"].map(dict(zip(test_rid["ID"].values, oracle_NEW_test)))

fname = f"submission_oracle_NEW_OOF{mae:.4f}.csv"
sub.to_csv(fname, index=False)
print(f"\\n제출 파일 저장: {fname}")
print(f"예측값 통계: mean={oracle_NEW_test.mean():.4f}  std={oracle_NEW_test.std():.4f}"
      f"  min={oracle_NEW_test.min():.4f}  max={oracle_NEW_test.max():.4f}")"""
))

# Cell 11: 검증 헤더
cells.append(md("## 검증: 제출 파일 무결성 확인"))

# Cell 12: 검증 코드
cells.append(code(
"""import os

fname = f"submission_oracle_NEW_OOF{mean_absolute_error(y_true, oracle_NEW_oof):.4f}.csv"
result = pd.read_csv(fname)

print(f"파일명    : {fname}")
print(f"행 수     : {len(result)}")
print(f"컬럼      : {list(result.columns)}")
print(f"NaN 개수  : {result[TARGET].isna().sum()}")
print(f"음수 개수 : {(result[TARGET] < 0).sum()}")
print(f"예측 평균 : {result[TARGET].mean():.4f}")
print(f"예측 최대 : {result[TARGET].max():.4f}")
print()

id_match = (result["ID"].values == sample_sub["ID"].values).all()
print(f"ID 순서 일치: {id_match}")
print("\\n검증 완료")"""
))

# ─── Step 5: 노트북 저장 ─────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
    },
    "cells": cells,
}

with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

size_mb = os.path.getsize(OUT) / 1024 / 1024
print(f"\n노트북 저장 완료: {OUT}")
print(f"파일 크기: {size_mb:.1f} MB")
print("20MB 이하 여부:", "OK" if size_mb < 20 else "OVER")
