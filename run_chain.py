"""
v10 완료 대기 → v11 → v12 자동 체이닝
v10이 이미 백그라운드에서 돌고 있으므로, submission_v10.csv가 생성될 때까지 대기 후 v11, v12 순차 실행
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import subprocess

print("=" * 60, flush=True)
print("체이닝 시작: v10 완료 대기 → v11 → v12", flush=True)
print("=" * 60, flush=True)

# v10 완료 대기
print("\n[1/3] v10 완료 대기 중...", flush=True)
while not os.path.exists('./submission_v10.csv'):
    time.sleep(30)
print("[1/3] v10 완료 확인!", flush=True)

# v11 실행
print("\n[2/3] v11 실행 중...", flush=True)
result = subprocess.run([sys.executable, 'train_v11.py'], capture_output=False)
if result.returncode != 0:
    print(f"[2/3] v11 실패 (exit code {result.returncode})", flush=True)
else:
    print("[2/3] v11 완료!", flush=True)

# v12 실행
print("\n[3/3] v12 실행 중...", flush=True)
result = subprocess.run([sys.executable, 'train_v12.py'], capture_output=False)
if result.returncode != 0:
    print(f"[3/3] v12 실패 (exit code {result.returncode})", flush=True)
else:
    print("[3/3] v12 완료!", flush=True)

print("\n" + "=" * 60, flush=True)
print("체이닝 완료! submission_v10~v12.csv 확인하세요.", flush=True)
print("=" * 60, flush=True)
