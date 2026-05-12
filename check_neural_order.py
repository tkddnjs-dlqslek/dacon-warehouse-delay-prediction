"""Check ordering and MAEs of neural model predictions."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np, pandas as pd

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

# Check bilstm ordering
with open('results/bilstm_final.pkl','rb') as f: d = pickle.load(f)
oof = d['oof']
mae_raw  = np.mean(np.abs(oof - y_true))
mae_id2  = np.mean(np.abs(oof[id2] - y_true))
print(f"bilstm:  no-idx MAE={mae_raw:.5f}  id2-idx MAE={mae_id2:.5f}  stored_mae={d['mae']:.5f}")

with open('results/cnn_final.pkl','rb') as f: d = pickle.load(f)
oof = d['cnn_oof']
mae_raw  = np.mean(np.abs(oof - y_true))
mae_id2  = np.mean(np.abs(oof[id2] - y_true))
print(f"cnn:     no-idx MAE={mae_raw:.5f}  id2-idx MAE={mae_id2:.5f}  stored_mae={d['cnn_oof_mae']:.5f}")

with open('results/mlp_final.pkl','rb') as f: d = pickle.load(f)
oof = d['mlp_oof']
mae_raw  = np.mean(np.abs(oof - y_true))
mae_id2  = np.mean(np.abs(oof[id2] - y_true))
print(f"mlp:     no-idx MAE={mae_raw:.5f}  id2-idx MAE={mae_id2:.5f}  stored_mae={d['mlp_oof_mae']:.5f}")

# Check all neural models with correct ordering
print(f"\n=== All neural model MAEs ===")
neural_configs = [
    ('bilstm',    'results/bilstm_final.pkl',           'oof',          'test'),
    ('cnn',       'results/cnn_final.pkl',               'cnn_oof',      'cnn_test'),
    ('tcn',       'results/tcn_final.pkl',               'oof',          'test'),
    ('transformer','results/transformer_enc_final.pkl',  'oof',          'test'),
    ('mlp',       'results/mlp_final.pkl',               'mlp_oof',      'mlp_test'),
    ('mlp2',      'results/mlp2_final.pkl',              'mlp2_oof',     'mlp2_test'),
    ('mlp_deep',  'results/mlp_deep_final.pkl',          'oof',          'test'),
    ('mlp_resnet','results/mlp_resnet_final.pkl',        'oof',          'test'),
    ('mlp_wide',  'results/mlp_wide_final.pkl',          'oof',          'test'),
    ('mlp_aug',   'results/mlp_aug_final.pkl',           'mlp_aug_oof',  'mlp_aug_test'),
    ('mlp_gelu',  'results/mlp_deep_gelu_final.pkl',     'oof',          'test'),
    ('mlp_s2',    'results/mlp_deep_s2_final.pkl',       'oof',          'test'),
    ('mlp_s3',    'results/mlp_deep_s3_final.pkl',       'oof',          'test'),
    ('deepcnn',   'results/deepcnn_final.pkl',           'oof',          'test'),
    ('mega33_v31','results/mega33_v31_final.pkl',        'meta_avg_oof', 'meta_avg_test'),
]

for name, fp, oof_key, test_key in neural_configs:
    with open(fp,'rb') as f: d = pickle.load(f)
    oof = d[oof_key]
    mae_raw = np.mean(np.abs(oof - y_true))
    mae_id2 = np.mean(np.abs(oof[id2] - y_true))
    # Stored MAE clue
    stored_mae = d.get('mae', d.get(oof_key+'_mae', 'N/A'))
    corr_raw  = np.corrcoef(oof, y_true)[0,1]
    print(f"  {name:12s}: no-idx={mae_raw:.5f}  id2={mae_id2:.5f}  stored={stored_mae if isinstance(stored_mae,str) else f'{stored_mae:.5f}':s}")
