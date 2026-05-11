# 스마트 창고 출고 지연 예측

월간 데이콘 - 스마트 창고 출고 지연 예측 AI 경진대회 (2026.04)

| | |
|--|--|
| 최종 순위 | **17등 / 608명** (상위 2.8%) |
| Public LB | 9.7527 (13위, 30%) |
| Private LB | 10.01261 (17위, 70%) |
| 평가 지표 | MAE |
| Train OOF | 8.3825 |

---

## 문제

AMR 로봇 물류창고 시뮬레이션 데이터에서 향후 30분간 평균 출고 지연(분)을 예측하는 회귀 문제.

```
300 layouts × 10,000 scenarios × 25 timeslots = 250,000 train rows
100 test layouts × 2,000 scenarios × 25 timeslots = 50,000 test rows
```

가장 큰 난점은 **test layouts 절반(50개)이 train에 없다**는 것. 이 unseen layouts의 order inflow가 train 대비 +87%, 혼잡도 +45%로 분포 자체가 다르다.

---

## 접근

### 기본 앙상블 (FIXED base)

LightGBM + XGBoost + CatBoost를 GroupKFold(layout_id, 5-fold)로 학습한 **mega33** 앙상블에, ranking 보정과 pseudo-labeling 3 rounds를 고정 가중치로 블렌드.

```python
fixed = (mega33 × 0.7637 + rank_adj × 0.1589
       + round1 × 0.0119 + round2 × 0.0346 + round3 × 0.0310)
```

**mega33 = 33 base 모델 + 3 meta-learner stacking**:

| 그룹 | 모델 | 개수 |
|------|------|------|
| v23 GBDT | LGB_Huber/XGB/CatBoost × 3 seeds | 9 |
| v24 GBDT | LGB_MAE/Huber/XGB/CatBoost | 4 |
| v26 variants | Tuned_Huber/sqrt/pow/DART | 4 |
| MLP/CNN | mlp1, mlp2, cnn | 3 |
| Domain-specific | domain × 3 | 3 |
| Offset decomposition | offset × 3 | 3 |
| mlp_aug | mlp_aug | 1 |
| Neural Army | BiLSTM/DeepCNN/TCN/MLP variants | 6 |
| **합계** | | **33** |

### Sequential Oracle

FIXED 예측값을 피처로 추가해 재학습한 oracle 모델들을 추가 블렌드. 서로 다른 피처 서브셋과 loss를 써서 다양성 확보.

```python
oracle_NEW = 0.64×FIXED + 0.12×oracle_xgb + 0.16×oracle_lv2 + 0.08×oracle_remaining
```

### 피처 (v31, 335개)

| 카테고리 | 주요 피처 |
|---------|---------|
| 로봇 상태 | robot_available_ratio, robot_charging |
| 패킹 병목 | pack_utilization SC 집계, pack_util_lead |
| 시나리오 집계 | inflow SC mean/max/std |
| 타임슬롯 위치 | row_in_scenario, sc_elapsed_ratio |

---

## 삽질 기록

250개 넘는 실험을 돌렸는데, 대부분 OOF는 개선되고 LB는 나빠지는 패턴이었다.

| 시도 | OOF | LB | 이유 |
|------|-----|----|------|
| Gate (seen/unseen 분기) | 개선 | 악화 | seen layout 패턴에만 맞음 |
| Scipy로 가중치 최적화 | 개선 | 미제출 | 동일한 이유 |
| Temporal CV (sc_num 기준) | 8.63 | blend 불가 | corr 0.96, MAE 너무 나쁨 |
| Unseen layouts 보정 (+5.5) | — | 미제출 | 실증 결과 LB 악화 |
| M/M/1 물리 모델 | 14.47 | blend 불가 | pack_util 하나로는 부족 |
| Meta-learner blend | — | 미제출 | unseen 범위 밖 예측 artifact |

결국 깨달은 건 **seen layout에서만 좋아지는 최적화는 전부 LB를 망친다**는 것. OOF가 좋아질수록 의심해야 한다.

새 모델이 기존 oracle과 corr ≥ 0.95면 블렌드해봤자 의미 없다. 피처셋이 같은 GBDT 모델끼리는 구조적으로 corr이 0.96 이상으로 수렴하기 때문에, 다양성 확보가 이 대회에서 가장 어려운 부분이었다.

---

## Public 13위 → Private 17위

갭 +0.26 (9.7527 → 10.01261). 심각한 과적합은 아니고, unseen layouts에 대한 체계적 과소예측이 30%보다 70% 샘플에서 더 안정적으로 드러난 것. 4명한테 밀린 건 걔네 모델이 unseen 일반화가 소폭 더 나았던 것으로 보인다.

---

## 재현 방법 (★ 노트북 1개 only, 완전 self-contained)

### 통합 노트북 1개

**`스마트_창고 출고 지연 예측 _ Mega33 앙상블 + Sequential Oracle (학습코드 포함).ipynb`** 단 하나로 EDA + 33-base 학습 + stacking + ranking + pseudo + oracle + blend + 최종 CSV 생성까지 전부.

**외부 .py 의존 0개** — 모든 학습 스크립트가 셀로 인라인됨 (Phase 0 EDA 코드도 포함).

**필요 입력 (4개, 모두 대회 데이터셋):**
- `train.csv`
- `test.csv`
- `layout_info.csv`
- `sample_submission.csv`

**출력:**
- `submission_oracle_NEW_OOF8.3825.csv` (Public 9.7527 / Private 10.01261)

### 실행 방법 2가지

#### A. 즉시 재현 (Part 2만, 1~2분)

학습된 8개 예측 배열이 base64로 임베드되어 있어 디코드 + 블렌드만 수행. **MD5 byte-identical** 검증 완료 (`94cba8c4f12ffa48e5ad5663e1719535`).

```
1. 통합 노트북과 4개 CSV를 같은 폴더에 둔다
2. 노트북 Part 2 셀들만 실행 (Cell 30+)
3. submission_oracle_NEW_OOF8.3825.csv 생성 (1~2분)
```

#### B. 처음부터 학습 (Part 1+2, 5~10시간) — 진짜 무인도 시나리오

```
1. 빈 폴더에 통합 노트북 + 4개 CSV 배치 (외부 .py 일체 불필요)
2. 노트북 셀 순서대로 전체 실행 (Restart & Run All)
3. Phase 0 (EDA cache 자동 생성, 8분) → Phase A~E 학습 (5~10h) → submission CSV

인터넷 연결 불필요 (오프라인 실행 가능).
검증: Phase 0 단독 실행 통과 (무인도_검증/ 폴더에 노트북+4CSV만으로 v30/v31 cache 정상 생성)
```

### 노트북 구조 (Part 1 / Phase A~E)

| Phase | 셀 | 내용 |
|---|---|---|
| A0 | Phase 0 | EDA cache (v30/v31) 자동 생성 — `engineer_features_v23()` 호출 |
| A1~A3 | train_v23 / v24 / v26 | GBDT base 17개 (v23×9 + v24×4 + v26×4) |
| B1~B7 | MLP / CNN / Domain / Offset / mlp_army / neural_army | Neural & domain 16개 base |
| C1a, C1b | train_mega_stacking / retrain_mega33_v31 | 33-base → 3-meta stacking |
| C2 | train_ranking | lambdarank — scenario rank-adjust |
| C3 | iterative_pseudo | 3-round pseudo-labeling |
| D0a, D0b | train_oracle_v2 / log | oracle_seq base |
| D1, D2, D3 | train_oracle_xgb / log_v2 / remaining | Sequential Oracle |
| E1 | final_multi_blend | FIXED + oracle_NEW 최종 블렌드 → CSV |

### 필요 라이브러리

```
pandas
numpy
scikit-learn
lightgbm
xgboost
catboost
torch          # neural army용
```

`requirements.txt` 참조.

---

## 저장소 구조

```
★ 메인 — 노트북 1개 only로 모든 것 (외부 .py 의존 0)
├── 스마트_창고 출고 지연 예측 _ Mega33 앙상블 + Sequential Oracle (학습코드 포함).ipynb
│   ├── Phase 0: EDA cache 생성 (engineer_features_v23 + build_fe_v31 인라인)
│   ├── Phase A1~A3: v23 / v24 / v26 GBDT base (17개)
│   ├── Phase B1~B7: MLP / CNN / Domain / Offset / mlp_army / neural_army (16개)
│   ├── Phase C1a~C3: mega33 stacking + ranking + pseudo
│   ├── Phase D0~D3: Oracle Sequential 5종
│   ├── Phase E1: 최종 블렌드 → submission CSV
│   └── Part 2: base64 디코드 (1~2분 즉시 재현용)
│
── 단독 .py 스크립트 (참고용, 노트북에 모두 인라인됨 — 단독 실행도 가능)
├── train_v23.py / train_v24.py / train_v26.py
├── train_mlp.py / train_mlp2.py / train_cnn.py
├── train_domain_aug.py / train_offset.py
├── train_mlp_army.py / train_neural_army.py
├── train_mega_stacking.py / retrain_mega33_v31.py
├── train_ranking.py / iterative_pseudo.py
├── train_oracle_v2.py / train_oracle_log.py
├── train_oracle_xgb.py / train_oracle_log_v2.py / train_oracle_remaining.py
├── final_multi_blend.py
├── build_fe_v31.py              # v31 feature engineering (노트북 Phase 0에 인라인됨)
│
── 기타
├── code_share/solution.ipynb    # (deprecated) 학습 코드 없는 초기 버전 — 통합본 사용 권장
├── make_best_submission.py      # 별도 entry point (CLI)
├── requirements.txt
├── src/
│   ├── temporal_oracle.py       # 실패한 Temporal CV 실험 (참고용)
│   ├── temporal_oracle_v2.py    # + M/M/1 피처 (참고용)
│   └── v32_quick_test.py
└── docs/
    ├── pipeline.md
    └── analysis.md
```

---

## 라이센스 / 인용

본 코드는 데이콘 대회 참가 목적으로 작성된 코드입니다. 자유롭게 참고하시되 상업적 이용은 금합니다.
