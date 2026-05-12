"""
Follow-up pipeline: runs after run_log_pipeline.py completes.
Adds oracle-XGB-log2 (XGB + log target + 2 log lags) then re-runs final blend.
"""
import sys, os, subprocess, time
sys.stdout.reconfigure(encoding='utf-8')

BASE   = r'C:\Users\user\Desktop\데이콘 4월'
PYTHON = r'C:\Python313\python.exe'
LOG    = os.path.join(BASE, 'logs', 'run_log_pipeline.log')

def wait_for_pipeline_done(timeout=10800):
    """Wait for run_log_pipeline.py to finish."""
    print("[followup] Waiting for main pipeline to finish...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        if os.path.exists(LOG):
            with open(LOG, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            if 'ALL DONE' in content:
                print(f"[followup] Main pipeline done ({time.time()-t0:.0f}s)", flush=True)
                return True
        time.sleep(30)
    print("[followup] Timeout waiting for pipeline", flush=True)
    return False

def run_step(script, label):
    print(f"\n[followup] === Starting {label} ===", flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON, script], cwd=BASE, capture_output=False)
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = 'DONE' if ok else f'FAILED(rc={result.returncode})'
    print(f"[followup] {label} {status} ({elapsed:.0f}s)", flush=True)
    return ok

# Wait for main pipeline
wait_for_pipeline_done()

extra_steps = [
    ('train_oracle_xgb_log2.py',     'oracle-XGB-log2',   'results/oracle_seq/oof_seqC_xgb_log2.npy'),
    ('train_oracle_lgb_latepos.py',  'oracle-LGB-latepos', 'results/oracle_seq/oof_seqC_lgb_latepos.npy'),
    ('train_oracle_lgb_log4.py',     'oracle-LGB-log4',    'results/oracle_seq/oof_seqC_lgb_log4.npy'),
    ('train_oracle_xgb_lag1only.py', 'oracle-XGB-lag1',   'results/oracle_seq/oof_seqC_xgb_lag1.npy'),
]

for script, label, oof_path in extra_steps:
    if os.path.exists(oof_path):
        print(f"[followup] {label} already done, skipping", flush=True)
        continue
    run_step(script, label)

print("\n[followup] === Running final blend (all models) ===", flush=True)
run_step('final_oracle_blend_all.py', 'final_blend_all')

print("[followup] ALL DONE", flush=True)
