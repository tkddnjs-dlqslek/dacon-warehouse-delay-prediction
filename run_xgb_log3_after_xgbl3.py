"""
Waits for oracle-XGB-lag3 to complete, then trains oracle-XGB-log-lag3.
After xgb_log3 completes, also re-runs final_oracle_blend_all.py for comprehensive result.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

XGB_L3_OOF = 'results/oracle_seq/oof_seqC_xgb_lag3.npy'
print("Waiting for XGB-lag3 to complete...", flush=True)
while not os.path.exists(XGB_L3_OOF):
    time.sleep(30)
print("XGB-lag3 done! Starting XGB-log-lag3...", flush=True)
subprocess.run([sys.executable, 'train_oracle_xgb_log_lag3.py'], capture_output=False)

# Also wait for the rest of the chain (ET + lv3) before final analysis
REQUIRED_ALL = [
    'results/oracle_seq/oof_seqC_rf.npy',
    'results/oracle_seq/oof_seqC_et.npy',
    'results/oracle_seq/oof_seqC_log_v3.npy',
    'results/oracle_seq/oof_seqC_xgb_lag3.npy',
    'results/oracle_seq/oof_seqC_xgb_log3.npy',
]
print("Waiting for ALL oracle models (including xgb_log3)...", flush=True)
while True:
    missing = [p for p in REQUIRED_ALL if not os.path.exists(p)]
    if not missing:
        break
    print(f"  Waiting: {missing}", flush=True)
    time.sleep(60)

print("All done! Running final comprehensive blend analysis...", flush=True)
subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)
print("Complete!", flush=True)
