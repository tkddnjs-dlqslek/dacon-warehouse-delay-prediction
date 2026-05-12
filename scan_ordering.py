import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, glob
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values
test_ls = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_rid_to_ls = np.array([te_ls_pos[i] for i in id_order])
train_layouts = set(pd.read_csv('train.csv')['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

print("%-55s %10s %10s %6s  true" % ("File", "rid_unsn", "ls_unsn", "ls_ord"))
print("-"*95)

results = []
for fpath in sorted(glob.glob('results/**/*test*.npy', recursive=True)):
    try:
        t = np.clip(np.load(fpath), 0, None)
        if t.ndim > 1: t = t.mean(axis=1)
        if len(t) != 50000: continue
        rid_u = float(t[unseen_mask].mean())
        ls_u = float(t[te_rid_to_ls][unseen_mask].mean())
        in_ls = abs(rid_u - ls_u) > 0.5
        true_u = ls_u if in_ls else rid_u
        results.append((true_u, rid_u, ls_u, in_ls, fpath))
    except:
        pass

for true_u, rid_u, ls_u, in_ls, fpath in sorted(results, reverse=True)[:35]:
    fname = os.path.relpath(fpath, 'results')
    print("%-55s %10.3f %10.3f %6s  %6.3f" % (fname, rid_u, ls_u, str(in_ls), true_u))
