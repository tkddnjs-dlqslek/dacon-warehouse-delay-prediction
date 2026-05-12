"""
Clean final watcher: waits for RF (from bg) + ET + log_v3 (from chain).
Then starts oracle-XGB-fixedproxy and runs comprehensive final analysis.
Oracle models that WORK: XGB (2 raw), lv2 (LGB log 3), RF (2 raw), ET (2 raw), lv3 (LGB log 5).
Oracle models that FAIL: XGB+3raw, XGB+3log (exposure bias amplification).
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

REQUIRED = [
    'results/oracle_seq/oof_seqC_rf.npy',
    'results/oracle_seq/oof_seqC_et.npy',
    'results/oracle_seq/oof_seqC_log_v3.npy',
]

print("Waiting for RF + ET + log_v3...", flush=True)
while True:
    missing = [os.path.basename(p) for p in REQUIRED if not os.path.exists(p)]
    if not missing:
        break
    print(f"  Waiting: {missing}", flush=True)
    time.sleep(60)
print("RF + ET + log_v3 ready!", flush=True)

# Quick first analysis (without fixedproxy)
print("\n=== First analysis (RF+ET+lv3+XGB+lv2) ===", flush=True)
subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)

# Now train oracle-XGB-fixedproxy if not done
XFP_OOF = 'results/oracle_seq/oof_seqC_xgb_fixedproxy.npy'
if not os.path.exists(XFP_OOF):
    print("\n=== Starting oracle-XGB-fixedproxy ===", flush=True)
    subprocess.run([sys.executable, 'train_oracle_xgb_fixedproxy.py'], capture_output=False)

# Also train oracle-RF-log if not done
RFLOG_OOF = 'results/oracle_seq/oof_seqC_rf_log.npy'
if not os.path.exists(RFLOG_OOF):
    print("\n=== Starting oracle-RF-log ===", flush=True)
    subprocess.run([sys.executable, 'train_oracle_rf_log.py'], capture_output=False)

# Final comprehensive analysis
print("\n=== Final comprehensive analysis ===", flush=True)
subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)
print("\nAll experiments complete!", flush=True)
