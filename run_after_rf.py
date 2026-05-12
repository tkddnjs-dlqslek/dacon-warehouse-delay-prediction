"""
Chain script: runs analyze_with_rf.py, then starts oracle-ET training.
Run this after train_oracle_rf.py completes.
"""
import sys, os, subprocess
sys.stdout.reconfigure(encoding='utf-8')

print("=== Running RF blend analysis ===", flush=True)
ret = subprocess.run([sys.executable, 'analyze_with_rf.py'], capture_output=False)
print(f"analyze_with_rf.py exit code: {ret.returncode}", flush=True)

print("\n=== Starting oracle-ET training ===", flush=True)
subprocess.run([sys.executable, 'train_oracle_et.py'], capture_output=False)
