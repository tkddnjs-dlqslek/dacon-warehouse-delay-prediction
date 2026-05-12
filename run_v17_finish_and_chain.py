"""
v17 5개 모델로 앙상블 마무리 → v18 → v19 체이닝
- v17 checkpoint 5개 활용
- slope를 빠른 방식(단순 차분)으로 대체
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from scipy.optimize import minimize

TARGET = 'avg_delay_minutes_next_30m'
V17_CKPT = './results/v17_ckpt'
SEED = 42

print("=== v17 마무리 (5개 모델 앙상블) ===", flush=True)
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
y = train[TARGET]

# v17 checkpoint 로드 (5개)
names = ['raw_LGB_MAE', 'raw_LGB_Huber', 'raw_CatBoost', 'log_LGB_MAE', 'log_LGB_Huber']
oofs, tests = [], []
for name in names:
    oof = np.load(f'{V17_CKPT}/{name}_s{SEED}_oof.npy')
    tpred = np.load(f'{V17_CKPT}/{name}_s{SEED}_test.npy')
    oofs.append(oof)
    tests.append(tpred)
    print(f"  {name}: OOF {mean_absolute_error(y, oof):.4f}", flush=True)

def ens_mae(w):
    w = np.array(w); w = np.maximum(w, 0); w = w / w.sum()
    return mean_absolute_error(y, sum(wi*p for wi,p in zip(w, oofs)))

res = minimize(ens_mae, x0=[0.2]*5, method='Nelder-Mead', options={'maxiter': 50000})
bw = np.array(res.x); bw = np.maximum(bw, 0); bw = bw / bw.sum()
best_mae = res.fun

print(f"\n  v17 앙상블 OOF MAE: {best_mae:.4f}")
for n, w in zip(names, bw):
    print(f"    {n}: {w:.3f}")

final_pred = np.clip(sum(wi*p for wi,p in zip(bw, tests)), 0, None)
pd.DataFrame({'ID': test['ID'], TARGET: final_pred}).to_csv('./submission_v17.csv', index=False)
print(f"  submission_v17.csv 저장\n", flush=True)

# v18, v19 실행
import subprocess
print("=== v18 실행 ===", flush=True)
r = subprocess.run([sys.executable, 'train_v18.py'], capture_output=False)
print(f"v18 {'완료' if r.returncode==0 else '실패'}", flush=True)

print("\n=== v19 실행 ===", flush=True)
r = subprocess.run([sys.executable, 'train_v19.py'], capture_output=False)
print(f"v19 {'완료' if r.returncode==0 else '실패'}", flush=True)

print("\n전체 체이닝 완료!", flush=True)
