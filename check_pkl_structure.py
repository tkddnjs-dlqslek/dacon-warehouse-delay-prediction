"""Quick check of pkl file structures to find OOF/test predictions."""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import pickle, numpy as np

files = [
    'results/bilstm_final.pkl',
    'results/cnn_final.pkl',
    'results/tcn_final.pkl',
    'results/transformer_enc_final.pkl',
    'results/mlp_final.pkl',
    'results/mlp2_final.pkl',
    'results/mlp_deep_final.pkl',
    'results/mlp_resnet_final.pkl',
    'results/mlp_wide_final.pkl',
    'results/mlp_aug_final.pkl',
    'results/mlp_deep_gelu_final.pkl',
    'results/mlp_deep_s2_final.pkl',
    'results/mlp_deep_s3_final.pkl',
    'results/deepcnn_final.pkl',
    'results/neural_army.pkl',
    'results/mlp_army.pkl',
    'results/stacking_final.pkl',
    'results/mega33_v31_final.pkl',
]

for fp in files:
    try:
        with open(fp,'rb') as f: d = pickle.load(f)
        if isinstance(d, dict):
            keys = list(d.keys())[:8]
            shapes = {k: (d[k].shape if hasattr(d[k],'shape') else type(d[k]).__name__) for k in keys}
            print(f"{fp}: keys={keys[:5]}  shapes={shapes}")
        elif isinstance(d, np.ndarray):
            print(f"{fp}: array {d.shape}")
        else:
            print(f"{fp}: type={type(d)}")
    except Exception as e:
        print(f"{fp}: ERROR {e}")
