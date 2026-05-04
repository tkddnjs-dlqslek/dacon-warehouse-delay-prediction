# Dacon 스마트 창고 출고 지연 예측

**월간 데이콘 - 스마트 창고 출고 지연 예측 AI 경진대회 (2026.04)**

| 항목 | 결과 |
|------|------|
| 최종 순위 | **17th / 608명** (상위 2.8%) |
| Public LB | 9.7527 (30%, 13위) |
| Private LB | 10.01261 (70%, 17위) |
| 평가 지표 | MAE (avg_delay_minutes_next_30m) |
| OOF MAE | 8.3825 |

---

## 문제 구조

AMR(자율 이동 로봇) 기반 물류창고 시뮬레이션 데이터에서 **향후 30분간 평균 출고 지연(분)**을 예측.

```
데이터 계층:
300 layouts × 10,000 scenarios × 25 timeslots = 250,000 train rows
100 test layouts × 2,000 scenarios × 25 timeslots = 50,000 test rows

핵심 난점: test layouts 중 50%가 train에 없는 unseen layouts
→ train 분포와 다른 물류 환경 (inflow +87%, 혼잡도 +45%)
```

---

## 핵심 발견

### 1. Seen/Unseen 도메인 시프트

| 구분 | Order Inflow | Pack Utilization | 예측값 |
|------|-------------|-----------------|-------|
| Seen (50 layouts) | 104.0 | 0.41 | 17.05 min |
| Unseen (50 layouts) | 172.2 (+87%) | 0.67 (+44%) | 22.72 min |

Unseen 레이아웃은 완전히 다른 물류 환경. 이 도메인 시프트가 OOF-LB 갭(8.38 vs 9.75)의 주원인.

### 2. GroupKFold OOF-LB 갭 분석

CV를 layout_id 기준 GroupKFold로 설계했기 때문에 validation에서 각 레이아웃의 패턴을 학습할 수 없었음. 그러나 역설적으로 **이 구조가 과적합을 방지**했다 — seen layout에 최적화하는 모든 시도(gate, scipy 최적화)가 LB에서 역효과를 냈음.

### 3. 앙상블 다양성 법칙

새 모델이 oracle_NEW와의 corr ≥ 0.95이면 블렌드 가중치가 0으로 수렴한다. 유효한 블렌드 파트너를 찾기 위해 250+ 실험을 체계적으로 수행했지만, 동일 피처셋 기반 모델은 모두 corr ≥ 0.96.

### 4. Public → Private 하락 분석 (13위 → 17위)

- Public(30%) 9.7527 → Private(70%) 10.01261, 갭 +0.26
- **과적합보다는 구조적 문제**: unseen 레이아웃에 대한 체계적 과소예측이 더 큰 private 샘플에서 더 정확히 반영됨
- 4명에게 추월당한 것은 그 모델들이 unseen 레이아웃 일반화가 더 나았을 가능성

---

## 파이프라인 구조

```
FIXED base (mega33 앙상블)
├── LightGBM × CatBoost × XGBoost (GroupKFold × 5)
├── Ranking 조정 모델 (rank_adj)
└── Pseudo-labeling 3 rounds (iter_r1~r3)
          ↓ 가중합 (fixed weight)
        FIXED
          ↓
    oracle_NEW = 0.64 × FIXED
              + 0.12 × oracle_xgb
              + 0.16 × oracle_lv2
              + 0.08 × oracle_remaining
```

**oracle_seq**: FIXED 예측값을 proxy 피처로 활용한 Sequential 모델군  
각 oracle은 서로 다른 피처 서브셋과 학습 전략을 사용해 다양성 확보

---

## 피처 엔지니어링 (v31, 335개)

| 카테고리 | 대표 피처 | 기여도 |
|---------|---------|--------|
| 로봇 상태 | robot_available_ratio, robot_charging | 상위 10 |
| 패킹 병목 | pack_utilization SC 집계, pack_util_lead | 상위 10 |
| 시나리오 집계 | inflow SC mean/max/std | 상위 20 |
| 타임슬롯 위치 | row_in_scenario, sc_elapsed_ratio | 중위권 |
| 레이아웃 통계 | layout_mean_delay (GroupKFold-safe) | 중위권 |

---

## 실패한 접근들 (주요 교훈)

| 접근 | OOF | 결과 | 원인 |
|------|-----|------|------|
| Gate (seen/unseen 분기) | 개선 | LB 악화 | seen layout에 과적합 |
| Scipy OOF 최적화 | 개선 | LB 악화 | OOF에 과적합 |
| Temporal CV + true lag | 8.63 | blend 불가 | corr 0.96, MAE 나쁨 |
| GP 보정 (unseen +5.5) | — | 미제출 | tw11 실증: unseen 상승=LB 악화 |
| M/M/1 물리 모델 | 14.47 | blend 불가 | pack_util만으론 예측 불가 |
| Meta-learner blend | — | 미제출 | Tree extrapolation failure |

---

## 재현 방법

```bash
# 환경
pip install -r requirements.txt

# 최종 제출 파일 생성
python make_best_submission.py
# → submission_oracle_NEW_OOF8.3825.csv 생성
```

**필요 파일** (용량 문제로 미포함):
- `train.csv`, `test.csv`, `sample_submission.csv`
- `results/mega33_final.pkl` (핵심 앙상블)
- `results/ranking/`, `results/iter_pseudo/`, `results/oracle_seq/` (OOF/test arrays)

---

## 디렉토리

```
├── make_best_submission.py     # 최종 제출 스크립트
├── src/
│   ├── temporal_oracle.py      # Temporal CV 실험 (OOF 8.63, 실패)
│   ├── temporal_oracle_v2.py   # + M/M/1 피처 실험 (OOF 8.63, 실패)
│   └── v32_quick_test.py       # 신규 피처 quick validation
└── docs/
    ├── pipeline.md             # 파이프라인 상세
    └── analysis.md             # 분석 방법론
```
