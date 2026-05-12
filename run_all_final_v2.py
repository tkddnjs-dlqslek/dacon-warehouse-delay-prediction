"""
Corrected final watcher: waits for RF, ET, log_v3 (from chain) + xgb_log3 + xgb_fixedproxy.
Then runs final_oracle_blend_all.py for the comprehensive analysis.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

REQUIRED = [
    'results/oracle_seq/oof_seqC_rf.npy',
    'results/oracle_seq/oof_seqC_et.npy',
    'results/oracle_seq/oof_seqC_log_v3.npy',
    'results/oracle_seq/oof_seqC_xgb_log3.npy',
]
OPTIONAL = [
    'results/oracle_seq/oof_seqC_xgb_fixedproxy.npy',
]

print("Waiting for required oracle models...", flush=True)
while True:
    missing = [p for p in REQUIRED if not os.path.exists(p)]
    if not missing:
        break
    print(f"  Missing: {[os.path.basename(p) for p in missing]}", flush=True)
    time.sleep(60)

# Check if fixedproxy is also ready (optional)
for p in OPTIONAL:
    if os.path.exists(p):
        print(f"  Bonus: {os.path.basename(p)} also ready!", flush=True)
    else:
        print(f"  Optional {os.path.basename(p)} not ready — will run without it", flush=True)

print("Required models ready! Running comprehensive final blend...", flush=True)
subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)

# Also start oracle-xgb-fixedproxy if not done yet
XFP_OOF = 'results/oracle_seq/oof_seqC_xgb_fixedproxy.npy'
if not os.path.exists(XFP_OOF):
    print("\nStarting oracle-XGB-fixedproxy now (xgb_log3 and chain complete)...", flush=True)
    subprocess.run([sys.executable, 'train_oracle_xgb_fixedproxy.py'], capture_output=False)
    print("Running final analysis WITH fixedproxy...", flush=True)
    subprocess.run([sys.executable, 'final_oracle_blend_all.py'], capture_output=False)
else:
    print("\nfixedproxy already done — final_oracle_blend_all.py already included it.", flush=True)

print("All done!", flush=True)
