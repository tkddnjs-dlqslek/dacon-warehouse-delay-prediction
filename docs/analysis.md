# 분석 방법론 — 의사결정 근거

## 1. CV 전략 선택: GroupKFold vs Temporal

### 배경
Train: SC_00001~SC_10000 / Test: SC_10001~SC_12000으로 시나리오 범위가 완전히 다름.
초기에는 Temporal Blocked CV(sc_num 기준 5-fold)가 test 분포를 더 잘 모사할 것으로 가설 수립.

### 실험 결과

| CV 전략 | OOF MAE | corr(oracle) | blend 결과 |
|---------|---------|-------------|-----------|
| GroupKFold (oracle_NEW) | 8.3825 | — | 기준 |
| Temporal CV v1 (sc_num) | 8.6259 | 0.9571 | 모든 blend 악화 |
| Temporal CV v2 (+mm1, +shift) | 8.6271 | 0.9410 | 모든 blend 악화 |

### 결론
GroupKFold oracle이 실제로 더 잘 일반화된다. test 시나리오가 새롭더라도 **레이아웃 구조**는 train에 존재하므로, layout 기준 CV가 핵심 패턴을 더 잘 포착.

---

## 2. Unseen Layout 도메인 시프트

### 발견
Test 100 layouts 중 50개(50%)가 train에 없는 unseen layouts.

| 구분 | Inflow 평균 | Robot Active | Congestion |
|------|------------|-------------|-----------|
| Train seen | 104.0 | — | — |
| Test seen | 104.0 × 1.115 | +19% | +13% |
| Test unseen | 104.0 × 1.819 | +45% | +45% |

### 시도한 보정 전략과 결과

**GP 보정 (LOOO 분석 기반)**
- 5개 독립 방법 수렴: unseen layout에 Δ≈+5.5 추가
- Tweedie blend 실증: unseen +0.087 → LB +0.0056 악화
- 결론: oracle 예측이 이미 최적, 수동 보정은 역효과

**OOD 레이아웃 직접 보정 (WH_283/201/246, pack_util > 0.85)**
- WH_217 잔차 기반 +21~24 보정 시도
- LB: 9.7527 → 9.9698 (+0.2171 악화)
- 결론: OOD 레이아웃 내부 분산이 극도로 높아(y=0~184) 패턴 추출 불가

### 핵심 인사이트
test_unseen 예측값을 올리는 방향 = 무조건 LB 악화. oracle_seq 컴포넌트들이 test_unseen을 낮추는 방향으로 작용하는데, 이것이 오히려 LB에서 이득이었음 (22.974→22.716).

---

## 3. 앙상블 다양성 분석

### 블렌드 가능 조건
corr(신규모델, oracle_NEW) < 0.95인 경우만 블렌드 가중치 > 0

### 250+ 실험 체계적 분류

| 카테고리 | corr 범위 | 결과 |
|---------|----------|------|
| v31 기반 신규 LGB/XGB | 0.96~0.98 | 전부 blend 불가 |
| Sequential oracle 32종 | 0.96~0.99 | 전부 blend 불가 |
| Temporal CV (Blocked) | 0.94~0.96 | OOF 너무 나빠서 불가 |
| M/M/1 물리 모델 | 0.15 | OOF 14.47로 너무 나쁨 |

### 결론
동일 피처셋(v30/v31) 기반 GBDT 모델은 구조적으로 corr ≥ 0.96. 다양성 확보를 위해서는 근본적으로 다른 피처셋이 필요하나, 그 경우 OOF 품질이 너무 낮아짐.

---

## 4. 과적합 방지 원칙 (실험으로 확인됨)

### OOF 개선 → LB 악화 패턴

| 기법 | OOF 변화 | LB 변화 |
|------|---------|--------|
| Gate (seen/unseen 분기) | 개선 | +0.012~+0.025 악화 |
| Scipy 100trial 최적화 | 개선 | 미제출 (예측됨) |
| Per-position 75-param 조정 | 8.3797 | 9.7558 (9.7527보다 악화) |
| Static global 2-4 param | 8.3831 | 9.7527 (동일 수준) |

**핵심 법칙**: Seen layout에서만 작동하는 최적화는 항상 LB 역효과.

### 안전한 블렌드 판별 기준
1. 고정 가중치 (grid search 아닌 사전 결정)
2. 신규 모델 OOF가 oracle_NEW보다 독립적으로 낮을 것
3. seen/unseen 예측 분포 비교 필수 (unseen 과소예측 여부)

---

## 5. Public → Private 하락 원인 분석

- Public(30%) MAE: 9.7527 / 13위
- Private(70%) MAE: 10.01261 / 17위
- 갭: +0.26 (2.7% 상대적 degradation)

### 주요 원인

**구조적 과소예측**: oracle_NEW의 unseen 예측 22.7 vs 실제 ~28 추정. 15,000행(public)보다 35,000행(private)에서 이 체계적 오차가 더 안정적으로 측정됨.

**4명 추월**: 심각한 overfitting이 아니라 경쟁자들이 unseen 일반화를 소폭 더 잘한 것. rank 13→17은 catastrophic 하락이 아님.

**grid search 영향**: make_best_submission.py의 4/5-way grid search가 OOF에 약간 최적화됨. 하지만 search space가 좁고(가중치 0.02 간격) 이 영향은 제한적.
