"""
v12 완료 대기 → v13 → v14 → v15 자동 체이닝
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import time
import subprocess

print("=" * 60, flush=True)
print("체이닝2 시작: v12 완료 대기 → v13 → v14 → v15", flush=True)
print("=" * 60, flush=True)

# v12 완료 대기
print("\n[1/4] v12 완료 대기 중...", flush=True)
while not os.path.exists('./submission_v12.csv'):
    time.sleep(30)
print("[1/4] v12 완료 확인!", flush=True)

# v13
print("\n[2/4] v13 실행 중...", flush=True)
result = subprocess.run([sys.executable, 'train_v13.py'], capture_output=False)
print(f"[2/4] v13 {'완료' if result.returncode == 0 else '실패'}!", flush=True)

# v14
print("\n[3/4] v14 실행 중...", flush=True)
result = subprocess.run([sys.executable, 'train_v14.py'], capture_output=False)
print(f"[3/4] v14 {'완료' if result.returncode == 0 else '실패'}!", flush=True)

# v15
print("\n[4/4] v15 실행 중...", flush=True)
result = subprocess.run([sys.executable, 'train_v15.py'], capture_output=False)
print(f"[4/4] v15 {'완료' if result.returncode == 0 else '실패'}!", flush=True)

print("\n" + "=" * 60, flush=True)
print("체이닝2 완료! submission_v13~v15.csv 확인하세요.", flush=True)
print("=" * 60, flush=True)
