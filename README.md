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

LightGBM + XGBoost + CatBoost를 GroupKFold(layout_id, 5-fold)로 학습한 mega33 앙상블에, ranking 보정과 pseudo-labeling 3 rounds를 고정 가중치로 블렌드.

```python
fixed = (mega33 × 0.7637 + rank_adj × 0.1589
       + round1 × 0.0119 + round2 × 0.0346 + round3 × 0.0310)
```

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

## 재현

```bash
pip install -r requirements.txt
python make_best_submission.py
# → submission_oracle_NEW_OOF8.3825.csv
```

데이터/모델 파일(train.csv, mega33_final.pkl 등)은 용량 문제로 미포함.

---

## 파일 구조

```
├── make_best_submission.py       # 최종 제출 스크립트
├── src/
│   ├── temporal_oracle.py        # Temporal CV 실험 (OOF 8.63, 실패)
│   ├── temporal_oracle_v2.py     # + M/M/1 피처 추가 실험 (실패)
│   └── v32_quick_test.py         # 신규 피처 빠른 검증용
└── docs/
    ├── pipeline.md               # 파이프라인 상세 및 인덱스 주의사항
    └── analysis.md               # 주요 의사결정 근거
```
