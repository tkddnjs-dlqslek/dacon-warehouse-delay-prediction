# -*- coding: utf-8 -*-
import json

def md(source):
    if isinstance(source, list):
        pass
    else:
        source = [source]
    return {"cell_type": "markdown", "metadata": {}, "source": source}

def code(source):
    lines = source.strip().split("\n")
    src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src}

cells = []

# ── Cell 1: 제목 ──────────────────────────────────────────────
cells.append(md([
    "# 스마트 창고 출고 지연 예측 — 솔루션 노트북\n",
    "\n",
    "**대회**: 데이콘 스마트 창고 출고 지연 예측 경진대회 (2026.04)  \n",
    "**결과**: Public LB 9.7527 (13위) / Private LB 10.01261 (17위) / 608팀 (상위 2.8%)  \n",
    "**평가 지표**: MAE  \n",
    "\n",
    "---\n",
    "\n",
    "## 전체 파이프라인\n",
    "\n",
    "```\n",
    "train.csv + test.csv\n",
    "    |\n",
    "    +--[Section 1] Mega33 앙상블 (사전 학습 결과 로드)\n",
    "    |      33개 Base 모델 (v23/v24/v26 x LGB/XGB/CB x 다중 시드)\n",
    "    |      + 3개 Meta 학습기 (LGB/XGB/CB) → mega33_final.pkl\n",
    "    |\n",
    "    +--[Section 2] FIXED Base 계산\n",
    "    |      FIXED = mega33 x 0.7637\n",
    "    |            + rank_adj x 0.1589\n",
    "    |            + pseudo_round1 x 0.0119\n",
    "    |            + pseudo_round2 x 0.0346\n",
    "    |            + pseudo_round3 x 0.0310\n",
    "    |\n",
    "    +--[Section 3] Sequential Oracle 모델 (사전 학습 결과 로드)\n",
    "    |      oracle_xgb / oracle_lv2 / oracle_remaining\n",
    "    |      (FIXED OOF를 proxy lag으로 활용한 auto-regressive XGB)\n",
    "    |\n",
    "    +--[Section 4] 최종 블렌드 (oracle_NEW)\n",
    "           oracle_NEW = FIXED x 0.64\n",
    "                      + oracle_xgb x 0.12\n",
    "                      + oracle_lv2 x 0.16\n",
    "                      + oracle_remaining x 0.08\n",
    "           OOF MAE: 8.3825  |  Public LB: 9.7527\n",
    "```\n",
    "\n",
    "---\n",
    "\n",
    "## 개발 환경\n",
    "\n",
    "```\n",
    "OS      : Windows 11 Home\n",
    "Python  : 3.10+ (Anaconda)\n",
    "lightgbm==4.6.0  xgboost==3.0.4  catboost==1.2.8\n",
    "scikit-learn==1.2.2  numpy==1.26.4  pandas==2.2.3\n",
    "```\n",
    "\n",
    "## 입력 파일 (노트북과 같은 폴더 기준)\n",
    "\n",
    "```\n",
    "train.csv  /  test.csv  /  sample_submission.csv\n",
    "results/mega33_final.pkl\n",
    "results/ranking/rank_adj_oof.npy  /  rank_adj_test.npy\n",
    "results/iter_pseudo/round1_oof.npy  /  round1_test.npy\n",
    "results/iter_pseudo/round2_oof.npy  /  round2_test.npy\n",
    "results/iter_pseudo/round3_oof.npy  /  round3_test.npy\n",
    "results/oracle_seq/oof_seqC_xgb.npy  /  test_C_xgb.npy\n",
    "results/oracle_seq/oof_seqC_log_v2.npy  /  test_C_log_v2.npy\n",
    "results/oracle_seq/oof_seqC_xgb_remaining.npy  /  test_C_xgb_remaining.npy\n",
    "```\n",
    "\n",
    "**예상 실행 시간**: 1분 미만",
]))

# ── Cell 2: 라이브러리 ─────────────────────────────────────────
cells.append(md(["## 라이브러리 임포트"]))
cells.append(code(
"""import numpy as np
import pandas as pd
import pickle
import os
import warnings

warnings.filterwarnings("ignore")
from sklearn.metrics import mean_absolute_error

TARGET = "avg_delay_minutes_next_30m"
print("라이브러리 임포트 완료")"""
))

# ── Cell 3: 문제 설명 ─────────────────────────────────────────
cells.append(md([
    "## Section 0: 문제 및 핵심 도전 과제\n",
    "\n",
    "AMR 로봇 물류창고 시뮬레이션 데이터에서 **향후 30분간 평균 출고 지연(분)을 예측**하는 회귀 문제.\n",
    "\n",
    "```\n",
    "Train: 300 layouts x 10,000 scenarios x 25 timeslots = 250,000 rows\n",
    "Test : 100 layouts x  2,000 scenarios x 25 timeslots =  50,000 rows\n",
    "```\n",
    "\n",
    "**핵심 난점: test layouts 절반(50개)이 train에 없는 unseen layout**\n",
    "\n",
    "| 구분 | Order Inflow | Congestion |\n",
    "|------|-------------|------------|\n",
    "| Test seen layouts   | +14% vs train | +13% |\n",
    "| Test unseen layouts | +87% vs train | +45% |\n",
    "\n",
    "이 분포 차이가 OOF-LB 갭의 주요 원인. seen layout에만 최적화되는 접근은 전부 LB 악화로 이어짐.\n",
    "\n",
    "**CV 전략**: `GroupKFold(layout_id, n=5)` — val fold에 unseen layout이 자연스럽게 포함되어\n",
    "GroupKFold(scenario_id), Temporal Blocked CV 대비 LB 일반화가 가장 우수함을 실험으로 확인.",
]))

# ── Cell 4: 데이터 로드 헤더 ──────────────────────────────────
cells.append(md([
    "## Section 1: 데이터 로드 및 인덱스 설정\n",
    "\n",
    "피처 엔지니어링은 `(layout_id, scenario_id)` 정렬(ls order)로 수행되고,\n",
    "타겟/제출은 ID 숫자 정렬(rid order)을 기준으로 합니다.\n",
    "두 순서를 `id2` / `te_id2` 맵으로 연결합니다.\n",
    "\n",
    "| 순서 | 기준 | 용도 |\n",
    "|------|------|----- |\n",
    "| rid order | ID 숫자 순 | 타겟(y), 제출 파일, oracle 예측 |\n",
    "| ls order  | (layout_id, scenario_id) 순 | 피처 엔지니어링, mega33 OOF/test 저장 |\n",
    "\n",
    "`d['meta_avg_oof'][id2]` 처럼 ls order 배열을 rid order로 변환해 사용.",
]))

# ── Cell 5: 데이터 로드 코드 ──────────────────────────────────
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

# ls order: (layout_id, scenario_id) 기준 정렬
SORT_KEY = ["layout_id", "scenario_id"]
train_ls = train_raw.sort_values(SORT_KEY).reset_index(drop=True)
test_ls  = test_raw.sort_values(SORT_KEY).reset_index(drop=True)

# rid → ls 인덱스 변환 맵
ls_pos    = {row["ID"]: i for i, row in train_ls.iterrows()}
te_ls_pos = {row["ID"]: i for i, row in test_ls.iterrows()}
id2    = np.array([ls_pos[i]    for i in train_rid["ID"].values])
te_id2 = np.array([te_ls_pos[i] for i in test_rid["ID"].values])

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

# ── Cell 6: Mega33 설명 ───────────────────────────────────────
cells.append(md([
    "## Section 2: Mega33 앙상블\n",
    "\n",
    "### Base 모델 구성 (33개)\n",
    "\n",
    "| 피처 버전 | 모델 종류 | 시드 | 수 |\n",
    "|-----------|----------|------|----|\n",
    "| v23 (lag/lead/rolling/SC 집계, ~150개) | LGB(Huber) / XGB / CatBoost | 42, 123, 2024 | 9 |\n",
    "| v24 (v23 + SC 확장 + 분위수 집계) | LGB(MAE) / LGB(Huber) / XGB / CatBoost | 42 | 4 |\n",
    "| v26 (v23 + 고급 튜닝 변형) | Tuned_Huber / Tuned_sqrt / Tuned_pow / DART | 42 | 4 |\n",
    "| domain_aug (v23 + 도메인 피처) | LGB(Huber) / XGB / CatBoost | 42 | 3 |\n",
    "| offset (시나리오 평균 + 잔차 분해) | LGB(Huber) / LGB(MAE) / XGB | 42 | 3 |\n",
    "| neural_army (시계열 신경망) | BiLSTM / CNN / MLP_wide / MLP_deep / MLP_resnet / TCN | — | 6 |\n",
    "| **합계** | | | **33** |\n",
    "\n",
    "모든 base 모델: `GroupKFold(layout_id, n=5)`, 타겟 `log1p(y)` 변환 후 학습.\n",
    "\n",
    "### Meta 스태킹\n",
    "\n",
    "33개 base 모델 OOF를 `log1p` 변환 후 스택 행렬 구성 →\n",
    "Level-2 Meta (LGB/XGB/CatBoost) 재학습 → 세 meta 평균 = `meta_avg_oof` / `meta_avg_test`.\n",
    "\n",
    "> 사전 학습 결과가 `results/mega33_final.pkl`에 저장되어 있습니다.\n",
    "\n",
    "### Ranking 보정 (rank_adj)\n",
    "\n",
    "layout_id 기준 예측값 순위 피처를 추가한 별도 XGB → OOF corr(mega33)≈0.88로 다양성 기여.\n",
    "\n",
    "### Pseudo-Labeling (round 1~3)\n",
    "\n",
    "test 예측값을 train에 추가해 반복 재학습. 3라운드로 unseen layout 일반화 소폭 개선.",
]))

# ── Cell 7: Mega33 로드 + FIXED 계산 ─────────────────────────
cells.append(code(
"""# mega33_final.pkl 로드
with open("results/mega33_final.pkl", "rb") as f:
    d = pickle.load(f)

print("mega33_final.pkl keys:", list(d.keys()))
print(f"  meta_avg_oof  shape: {d['meta_avg_oof'].shape}")
print(f"  meta_avg_test shape: {d['meta_avg_test'].shape}")
print(f"  mega33 OOF MAE: {mean_absolute_error(y_true, d['meta_avg_oof'][id2]):.5f}")

# FIXED base 가중치 (OOF grid search로 최적화)
fw = dict(
    mega33   = 0.7636614598089654,
    rank_adj = 0.1588758398901156,
    iter_r1  = 0.011855567572749024,
    iter_r2  = 0.034568307,
    iter_r3  = 0.031038826,
)
print(f"\\nFIXED 가중치 합계: {sum(fw.values()):.6f}")

# FIXED OOF (train, rid order)
fixed_oof = (
    fw["mega33"]   * d["meta_avg_oof"][id2]
  + fw["rank_adj"] * np.load("results/ranking/rank_adj_oof.npy")[id2]
  + fw["iter_r1"]  * np.load("results/iter_pseudo/round1_oof.npy")[id2]
  + fw["iter_r2"]  * np.load("results/iter_pseudo/round2_oof.npy")[id2]
  + fw["iter_r3"]  * np.load("results/iter_pseudo/round3_oof.npy")[id2]
)

# FIXED test (test, rid order)
fixed_test = (
    fw["mega33"]   * d["meta_avg_test"][te_id2]
  + fw["rank_adj"] * np.load("results/ranking/rank_adj_test.npy")[te_id2]
  + fw["iter_r1"]  * np.load("results/iter_pseudo/round1_test.npy")[te_id2]
  + fw["iter_r2"]  * np.load("results/iter_pseudo/round2_test.npy")[te_id2]
  + fw["iter_r3"]  * np.load("results/iter_pseudo/round3_test.npy")[te_id2]
)

fixed_mae = mean_absolute_error(y_true, fixed_oof)
print(f"\\nFIXED OOF MAE: {fixed_mae:.5f}")
print(f"FIXED test  mean={fixed_test.mean():.3f}  std={fixed_test.std():.3f}")"""
))

# ── Cell 8: Oracle 설명 ───────────────────────────────────────
cells.append(md([
    "## Section 3: Sequential Oracle\n",
    "\n",
    "**핵심 아이디어**: 이전 타임슬롯의 실제 지연 시간이 가장 강력한 예측 피처.\n",
    "test 예측 시 실제 y를 알 수 없으므로 FIXED OOF를 proxy lag으로 대체.\n",
    "\n",
    "| 상황 | lag 피처 |\n",
    "|------|----------|\n",
    "| Train (학습) | 진짜 y(t-1) — 완벽한 정보 |\n",
    "| Val (검증) | FIXED_OOF(t-1) — proxy |\n",
    "| Test (추론) | 직전 타임슬롯 예측값 — 순차적으로 채워나감 |\n",
    "\n",
    "### Oracle 변형 3종\n",
    "\n",
    "| 이름 | 피처셋 | 손실 | OOF MAE | corr(FIXED) |\n",
    "|------|--------|------|---------|-------------|\n",
    "| oracle_xgb | v31(335개) + lag1/2 + row_in_sc | MAE | 8.43846 | 0.9733 |\n",
    "| oracle_lv2 | v31 + lag1/2 + log1p 타겟 | MAE | 8.44093 | 0.9721 |\n",
    "| oracle_remaining | v31 서브셋 (pack 계열 제외) + lag1/2 | MAE | 8.47155 | 0.9760 |\n",
    "\n",
    "세 oracle 모두 `corr(FIXED) ≈ 0.97` — FIXED 대비 다양성은 제한적이나 oracle 간 corr ≈ 0.90~0.92로 상호 보완.",
]))

# ── Cell 9: Oracle 로드 ───────────────────────────────────────
cells.append(code(
"""# Sequential Oracle OOF/test 로드 (모두 rid order)
oracle_xgb_oof  = np.load("results/oracle_seq/oof_seqC_xgb.npy")
oracle_xgb_test = np.load("results/oracle_seq/test_C_xgb.npy")

oracle_lv2_oof  = np.load("results/oracle_seq/oof_seqC_log_v2.npy")
oracle_lv2_test = np.load("results/oracle_seq/test_C_log_v2.npy")

oracle_rem_oof  = np.load("results/oracle_seq/oof_seqC_xgb_remaining.npy")
oracle_rem_test = np.load("results/oracle_seq/test_C_xgb_remaining.npy")

print("Oracle OOF MAE:")
print(f"  oracle_xgb      : {mean_absolute_error(y_true, oracle_xgb_oof):.5f}")
print(f"  oracle_lv2      : {mean_absolute_error(y_true, oracle_lv2_oof):.5f}")
print(f"  oracle_remaining: {mean_absolute_error(y_true, oracle_rem_oof):.5f}")

print("\\ncorr(FIXED, oracle):")
print(f"  oracle_xgb      : {np.corrcoef(fixed_oof, oracle_xgb_oof)[0,1]:.4f}")
print(f"  oracle_lv2      : {np.corrcoef(fixed_oof, oracle_lv2_oof)[0,1]:.4f}")
print(f"  oracle_remaining: {np.corrcoef(fixed_oof, oracle_rem_oof)[0,1]:.4f}")
print("  (oracle 간 corr ≈ 0.90~0.92, FIXED 대비는 0.97 수준)")"""
))

# ── Cell 10: 최종 블렌드 헤더 ─────────────────────────────────
cells.append(md([
    "## Section 4: 최종 블렌드 (oracle_NEW) 및 제출 파일 생성\n",
    "\n",
    "OOF 기준 grid search로 탐색한 최적 블렌드:\n",
    "\n",
    "```\n",
    "oracle_NEW = FIXED       x 0.64\n",
    "           + oracle_xgb  x 0.12\n",
    "           + oracle_lv2  x 0.16\n",
    "           + oracle_rem  x 0.08\n",
    "```\n",
    "\n",
    "OOF MAE: **8.3825**  |  Public LB: **9.7527** (13위)  |  Private LB: **10.01261** (17위)\n",
    "\n",
    "### 주요 실패 실험 (OOF 개선 ≠ LB 개선)\n",
    "\n",
    "| 시도 | OOF | LB | 결론 |\n",
    "|------|-----|----|------|\n",
    "| Gate (seen/unseen 분기 최적화) | 개선 | 악화 | seen 과적합 |\n",
    "| Scipy 가중치 최적화 | 개선 | 미제출 | OOF 과적합 |\n",
    "| Temporal CV (sc_num 기준) | 8.63 | blend 불가 | corr=0.957, GroupKFold가 실제로 더 일반화됨 |\n",
    "| test_unseen 보정 (+5.5) | — | 악화 | Δunseen 상승 → LB 파국적 악화 |\n",
    "| M/M/1 물리 모델 | 14.47 | blend 불가 | 단일 변수로 한계 |\n",
    "| neural_army blend | 9.77+ | 악화 | unseen 과적합 |\n",
    "\n",
    "**결론**: 250개 이상 실험 중 LB를 개선한 접근은 oracle_NEW 블렌드뿐.",
]))

# ── Cell 11: 최종 블렌드 코드 ─────────────────────────────────
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

# ── Cell 12: 검증 ─────────────────────────────────────────────
cells.append(md(["## 검증: 제출 파일 무결성 확인"]))
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

# sample_submission ID 순서 일치 확인
id_match = (result["ID"].values == sample_sub["ID"].values).all()
print(f"ID 순서 일치: {id_match}")
print("\\n검증 완료")"""
))

# ── 노트북 완성 ───────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = r"C:\Users\user\Desktop\데이콘 4월\code_share\solution.ipynb"
with open(out, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"저장 완료: {out}")
print(f"총 셀 수: {len(cells)}개")
