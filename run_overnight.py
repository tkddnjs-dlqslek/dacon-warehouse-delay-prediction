"""
밤새 실험 순서: KK → layout_v3 학습 → MM (oracle_NEW + layout_v3)
"""
import subprocess, sys, os, time
os.chdir(r'C:\Users\user\Desktop\데이콘 4월')

steps = [
    ('build_submissions_kk.py',   'KK batch (oracle_NEW variants)'),
    ('train_oracle_layout_v3.py', 'Layout-v3 oracle 학습 (~1시간)'),
    ('build_submissions_mm.py',   'MM batch (oracle_NEW + layout_v3)'),
]

print("="*60)
print("Overnight Run 시작")
print(f"시작: {time.strftime('%H:%M:%S')}")
print("="*60)

for script, desc in steps:
    print(f"\n{'='*60}")
    print(f"[{time.strftime('%H:%M:%S')}] {desc}")
    print(f"Script: {script}")
    print('='*60)
    if not os.path.exists(script):
        print(f"  SKIP: {script} 없음")
        continue
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    print(f"\n[{time.strftime('%H:%M:%S')}] {script} 완료 (exit={result.returncode})")

print(f"\n{'='*60}")
print(f"전체 완료: {time.strftime('%H:%M:%S')}")
print("check_status.py로 결과 확인하세요")
