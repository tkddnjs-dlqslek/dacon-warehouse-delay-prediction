"""
Waits for all oracle OOF files to exist, then runs final_oracle_blend_all.py.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

REQUIRED = [
    'results/oracle_seq/oof_seqC_rf.npy',
    'results/oracle_seq/oof_seqC_et.npy',
    'results/oracle_seq/oof_seqC_log_v3.npy',
    'results/oracle_seq/oof_seqC_xgb_lag3.npy',
]

print("Waiting for all oracle models to complete...", flush=True)
while True:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if not missing:
        break
    print(f"  Still waiting for: {missing}", flush=True)
    time.sleep(60)

print("All oracle models ready! Running final blend analysis...", flush=True)
subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)
print("Final analysis complete!", flush=True)
