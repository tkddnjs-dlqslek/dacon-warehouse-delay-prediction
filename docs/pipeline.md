# 파이프라인 상세 — oracle_NEW (LB 9.7527)

## 재현 명령어

```bash
python make_best_submission.py
```

검증: `submission_oracle_NEW_OOF8.3825.csv`와 max_diff=0.000000 (완벽 일치 확인됨)

---

## oracle_NEW 블렌드 공식

```python
# STEP 1: FIXED base
fw = {
    'mega33':   0.7636614598089654,
    'rank_adj': 0.1588758398901156,
    'iter_r1':  0.011855567572749024,
    'iter_r2':  0.034568307,
    'iter_r3':  0.031038826,
}
fixed = (fw['mega33']   * mega33_pred
       + fw['rank_adj'] * rank_adj_pred
       + fw['iter_r1']  * round1_pred
       + fw['iter_r2']  * round2_pred
       + fw['iter_r3']  * round3_pred)

# STEP 2: 최종 블렌드
oracle_NEW = max(0, 0.64*fixed + 0.12*xgb + 0.16*lv2 + 0.08*remaining)

# OOF MAE: 8.3825  |  LB(Public 30%): 9.7527  |  LB(Private 70%): 10.01261
```

---

## 인덱스 순서 주의사항

```
rid order  = train_raw['_row_id'] 기준 정렬 (TRAIN_000000 ~ TRAIN_249999)
ls order   = train_raw.sort_values(['layout_id','scenario_id']) 기준

mega33/34 pkl       → ls order → [id2] 인덱싱 필요
rank_adj/iter_pseudo → ls order → [id2] 인덱싱 필요
oracle_seq/oof*.npy → rid order → 인덱싱 불필요

# ls→rid 변환
train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id'])
ls_pos = {row['ID']: i for i, row in train_ls.iterrows()}
id2 = np.array([ls_pos[i] for i in train_raw['ID'].values])
```

---

## 필요 파일 목록

### 입력 데이터
| 파일 | 크기 |
|------|------|
| `train.csv` | 118 MB |
| `test.csv` | 22 MB |
| `sample_submission.csv` | 781 KB |

### 학습된 모델
| 파일 | 설명 | 인덱스 |
|------|------|--------|
| `results/mega33_final.pkl` | 핵심 앙상블 (LGB+XGB+CB) | ls order |

### OOF / Test 예측값
| 파일 | 설명 | 인덱스 |
|------|------|--------|
| `results/ranking/rank_adj_oof.npy` | 랭킹 조정 OOF | ls order |
| `results/ranking/rank_adj_test.npy` | 랭킹 조정 test | ls order |
| `results/iter_pseudo/round1~3_oof.npy` | Pseudo-label OOF | ls order |
| `results/iter_pseudo/round1~3_test.npy` | Pseudo-label test | ls order |
| `results/oracle_seq/oof_seqC_xgb.npy` | Sequential XGB OOF | rid order |
| `results/oracle_seq/test_C_xgb.npy` | Sequential XGB test | rid order |
| `results/oracle_seq/oof_seqC_log_v2.npy` | Sequential LGB v2 OOF | rid order |
| `results/oracle_seq/test_C_log_v2.npy` | Sequential LGB v2 test | rid order |
| `results/oracle_seq/oof_seqC_xgb_remaining.npy` | Remaining XGB OOF | rid order |
| `results/oracle_seq/test_C_xgb_remaining.npy` | Remaining XGB test | rid order |

---

## CV 전략

**GroupKFold(layout_id, n=5)**  
- 같은 레이아웃의 모든 시나리오가 동일 fold에 배정
- Seen/unseen 도메인 시프트 완화 목적
- Temporal CV 대비 OOF가 더 안정적 (temporal_oracle 실험에서 확인)
