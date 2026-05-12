"""Check stacking_final, neural_army, mlp_army structures and MAEs."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np, pandas as pd

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

# Stacking final
with open('results/stacking_final.pkl','rb') as f: d = pickle.load(f)
print("stacking_final keys:", list(d.keys()))
print("best_stack_name:", d.get('best_stack_name','N/A'))
res = d.get('results',{})
print("results sub-keys:", list(res.keys())[:10])
for k in list(res.keys())[:5]:
    v = res[k]
    if isinstance(v, dict): print(f"  {k}: {list(v.keys())}")

# Neural army
with open('results/neural_army.pkl','rb') as f: d_na = pickle.load(f)
print("\nneural_army keys:", list(d_na.keys()))
for name in d_na:
    d = d_na[name]
    if isinstance(d, dict):
        oof_key = [k for k in d if 'oof' in k.lower()][0] if any('oof' in k.lower() for k in d) else None
        if oof_key:
            oof = d[oof_key]
            if hasattr(oof,'__len__') and len(oof)==250000:
                mae = np.mean(np.abs(oof[id2] - y_true))
                print(f"  {name}: id2_mae={mae:.5f}")

# MLP army
with open('results/mlp_army.pkl','rb') as f: d_ma = pickle.load(f)
print("\nmlp_army keys:", list(d_ma.keys()))
for name in d_ma:
    d = d_ma[name]
    if isinstance(d, dict):
        oof_key = [k for k in d if 'oof' in k.lower()][0] if any('oof' in k.lower() for k in d) else None
        if oof_key:
            oof = d[oof_key]
            if hasattr(oof,'__len__') and len(oof)==250000:
                mae = np.mean(np.abs(oof[id2] - y_true))
                print(f"  {name}: id2_mae={mae:.5f}")
