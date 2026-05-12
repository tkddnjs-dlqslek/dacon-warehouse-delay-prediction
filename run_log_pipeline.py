"""
New pipeline: raw-lag RF/ET failed (exposure bias too strong).
Log-lag approach: log(y+1) target + log lags → distribution compression helps RF survive.

Order:
  1. oracle-log_v3     (LGB, 5 log lags)
  2. oracle-XGB-fixedproxy  (XGB, 2 raw lags, FIXED proxy)
  3. oracle-RF-log     (RF, log target, 3 log lags)
  4. oracle-lgb_cumstats    (LGB, log + cumulative stats)
  5. final_oracle_blend_all.py
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

BASE = r'C:\Users\user\Desktop\데이콘 4월'
PYTHON = r'C:\Python313\python.exe'

def wait_for_file(path, label, interval=30, timeout=7200):
    print(f"[pipeline] Waiting for {label}: {os.path.basename(path)}", flush=True)
    t0 = time.time()
    while not os.path.exists(path):
        elapsed = time.time() - t0
        if elapsed > timeout:
            print(f"[pipeline] TIMEOUT waiting for {label}", flush=True)
            return False
        time.sleep(interval)
    print(f"[pipeline] {label} ready ({time.time()-t0:.0f}s)", flush=True)
    return True

def run_step(script, label):
    print(f"\n[pipeline] === Starting {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON, script], cwd=BASE, capture_output=False)
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(f"[pipeline] {label} DONE ({elapsed:.0f}s)", flush=True)
        return True
    else:
        print(f"[pipeline] {label} FAILED (rc={result.returncode}, {elapsed:.0f}s)", flush=True)
        return False

steps = [
    ('train_oracle_log_v3.py',         'oracle-log_v3',        'results/oracle_seq/oof_seqC_log_v3.npy'),
    ('train_oracle_xgb_fixedproxy.py', 'oracle-XGB-fixedproxy','results/oracle_seq/oof_seqC_xgb_fixedproxy.npy'),
    ('train_oracle_rf_log.py',         'oracle-RF-log',        'results/oracle_seq/oof_seqC_rf_log.npy'),
    ('train_oracle_lgb_cumstats.py',   'oracle-lgb_cumstats',  'results/oracle_seq/oof_seqC_cumstats.npy'),
]

for script, label, oof_path in steps:
    if os.path.exists(oof_path):
        print(f"[pipeline] {label} already done, skipping", flush=True)
        continue
    ok = run_step(script, label)
    if not ok:
        print(f"[pipeline] Skipping rest of chain after {label} failure", flush=True)

print("\n[pipeline] === Running final blend analysis ===", flush=True)
run_step('final_oracle_blend_all.py', 'final_blend_all')

print("[pipeline] ALL DONE", flush=True)
