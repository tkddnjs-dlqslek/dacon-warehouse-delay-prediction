"""DD → EE → FF → GG → HH → II 연속 실행 (CC는 이미 실행 중)."""
import subprocess, sys, os, time
os.chdir(r'C:\Users\user\Desktop\데이콘 4월')

batches = ['build_submissions_dd.py', 'build_submissions_ee.py',
           'build_submissions_ff.py', 'build_submissions_gg.py',
           'build_submissions_hh.py', 'build_submissions_ii.py']

for script in batches:
    print(f"\n{'='*60}\nStarting {script} at {time.strftime('%H:%M:%S')}\n{'='*60}")
    result = subprocess.run([sys.executable, script], capture_output=False, text=True)
    print(f"\n{script} finished (exit={result.returncode}) at {time.strftime('%H:%M:%S')}")
