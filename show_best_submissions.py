"""
Shows all oracle submissions ranked by OOF MAE with status indicators.
Run to see what to submit.
"""
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

submissions = []
for f in glob.glob('submission_oracle*.csv') + glob.glob('submission_final*.csv'):
    size = os.path.getsize(f)
    if size < 1000: continue  # skip empty
    try:
        oofstr = ''
        for part in f.replace('.csv','').split('_'):
            if part.startswith('OOF') or part.startswith('oof'):
                oofstr = part.replace('OOF','').replace('oof','')
        if oofstr:
            oofval = float(oofstr)
        else:
            oofval = 99.0
    except:
        oofval = 99.0
    submissions.append((oofval, f, size))

submissions.sort(key=lambda x: x[0])
print(f"\n{'OOF MAE':<10} {'File':<70} {'Size':>8}")
print('-'*90)
for mae, fname, sz in submissions:
    # Highlight the best ones
    marker = ''
    if mae < 8.3801: marker = ' <<< NEW BEST'
    elif mae < 8.3832: marker = ' <<< RECOMMEND'
    print(f"{mae:<10.4f} {fname:<70} {sz//1024:>5}K{marker}")

print(f"\nCurrent LB best: 9.7711 (FIXED, OOF 8.3935)")
print(f"Best oracle OOF so far: {submissions[0][0]:.4f} ({submissions[0][1]})")
