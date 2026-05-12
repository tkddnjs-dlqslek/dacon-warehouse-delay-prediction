"""
실험 결과 요약 리포트 — 사용자 복귀 시 실행
모든 실험 출력 파일을 읽고 핵심 결과만 정리
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import os, glob
import numpy as np, pandas as pd

print("=" * 70)
print("SESSION 6 실험 결과 요약")
print("=" * 70)

# oracle_NEW 기준값
ORACLE_OOF = 8.3762
ORACLE_LB  = 9.7527
TARGET_LB  = 9.699

print(f"\n기준: oracle_NEW OOF={ORACLE_OOF:.4f}  LB={ORACLE_LB}  (1등 목표: {TARGET_LB})")
print(f"갭:   {ORACLE_LB - TARGET_LB:.4f} LB 개선 필요\n")

# 각 실험 출력 파일 확인
experiments = [
    ("/tmp/hp_out.txt",         "hp_feat_search    (60-trial HP + interactions)"),
    ("/tmp/layout_ctx_out.txt", "layout_context     (test 분포 통계 피처)"),
    ("/tmp/layout_tenc_out.txt","layout_target_enc  (layout_id 타겟 인코딩)"),
    ("/tmp/scenlag_out.txt",    "scenario_lag       (이전 시나리오 lag)"),
    ("/tmp/physics_out.txt",    "physics_blend      (M/M/1 물리 보정)"),
    ("/tmp/alpha_out.txt",      "alpha_variants     (Huber alpha 변형)"),
    ("/tmp/poisson_out.txt",    "poisson_blend      (Poisson 회귀)"),
]

print("-" * 70)
print(f"{'실험':<45} {'상태':>8} {'OOF':>9} {'corr':>7} {'gain':>8}")
print("-" * 70)

for fpath, label in experiments:
    if not os.path.exists(fpath):
        print(f"  {label:<45} {'대기':>8}")
        continue

    with open(fpath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()

    if not content.strip():
        print(f"  {label:<45} {'실행중':>8}")
        continue

    # 핵심 결과 파싱
    oof_val, corr_val, gain_val = None, None, None
    done = "Done." in content

    for line in content.split('\n'):
        if 'OOF (rid):' in line or 'OOF:' in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if 'OOF' in p and i+1 < len(parts):
                    try: oof_val = float(parts[i+1].split('=')[-1]); break
                    except: pass
        if 'corr=' in line:
            try: corr_val = float(line.split('corr=')[1].split()[0].split(',')[0])
            except: pass
        if 'gain=' in line and 'Best' in line:
            try: gain_val = float(line.split('gain=')[1].split()[0])
            except: pass

    saved = "SAVED" in content
    status = "완료✓" if done else "실행중"
    oof_s  = f"{oof_val:.4f}" if oof_val else "  --  "
    corr_s = f"{corr_val:.4f}" if corr_val else " -- "
    gain_s = f"{gain_val:+.4f}" if gain_val else "  -- "
    flag   = " *** 제출후보 ***" if saved else ""
    print(f"  {label:<45} {status:>8} {oof_s:>9} {corr_s:>7} {gain_s:>8}{flag}")

print("-" * 70)

# 생성된 제출 파일
print("\n생성된 제출 파일 (FINAL_*.csv, 오늘):")
dacon_dir = "C:/Users/user/Desktop/데이콘 4월"
finals = sorted(glob.glob(f"{dacon_dir}/FINAL_*.csv"), key=os.path.getmtime, reverse=True)
today_finals = [f for f in finals if '2026-05-03' in pd.Timestamp(os.path.getmtime(f), unit='s').strftime('%Y-%m-%d')]
if today_finals:
    for f in today_finals[:10]:
        mtime = pd.Timestamp(os.path.getmtime(f), unit='s').strftime('%H:%M')
        print(f"  {os.path.basename(f):55} ({mtime})")
else:
    print("  (새로 생성된 제출 파일 없음)")

# hp_feat_search 세부 결과
print("\n--- hp_feat_search 상세 (trial 결과) ---")
if os.path.exists("/tmp/hp_out.txt"):
    with open("/tmp/hp_out.txt", 'r', encoding='utf-8', errors='replace') as f:
        hp_lines = f.readlines()
    # Section 1 결과
    in_sec1 = False
    for line in hp_lines:
        if 'Section 1' in line: in_sec1 = True
        if 'Section 2' in line: in_sec1 = False
        if in_sec1 and ('baseline' in line or 'interaction' in line or 'delta' in line):
            print(f"  {line.rstrip()}")
    # Best candidate 결과
    for line in hp_lines:
        if 'Best' in line and ('blend' in line or 'gain' in line):
            print(f"  {line.rstrip()}")
        if 'Candidate' in line or 'SAVED' in line:
            print(f"  {line.rstrip()}")

print("\n" + "=" * 70)
print("파이프라인 상태:")
if os.path.exists("/tmp/experiments_log.txt"):
    with open("/tmp/experiments_log.txt", 'r', encoding='utf-8', errors='replace') as f:
        print(f.read())
else:
    print("  파이프라인 아직 시작 전 (hp_feat_search 실행 중)")
print("=" * 70)
