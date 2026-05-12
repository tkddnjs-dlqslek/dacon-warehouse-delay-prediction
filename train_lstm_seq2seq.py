"""
Seq2Seq LSTM: 시나리오 전체 25 timestep을 한번에 예측.
- No oracle / no proxy lag issue
- Bidirectional LSTM (test time에도 전체 sequence input 가능)
- v30 feat_cols 사용 (149 features × 25 timesteps)
"""
import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupKFold

ROOT = r"C:\Users\user\Desktop\데이콘 4월"
DEVICE = 'cpu'
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

# ── Load v30 FE ──
print("Loading v30 FE...", flush=True)
with open(os.path.join(ROOT, "results", "eda_v30", "v30_fe_cache.pkl"), "rb") as f:
    v30 = pickle.load(f)
with open(os.path.join(ROOT, "results", "eda_v30", "v30_test_fe_cache.pkl"), "rb") as f:
    te_raw = pickle.load(f)

feat_cols = v30['feat_cols']
train_fe  = v30['train_fe'].sort_values(['layout_id','scenario_id']).reset_index(drop=True)
test_fe   = pd.DataFrame(te_raw).sort_values(['layout_id','scenario_id']).reset_index(drop=True)

TARGET = "avg_delay_minutes_next_30m"
SEQ_LEN = 25
N_FEAT  = len(feat_cols)

print(f"feat_cols: {N_FEAT}", flush=True)

# ── Reshape to sequences ──
def to_sequences(df, cols, target_col=None):
    """(n_scenarios, SEQ_LEN, n_features) and optionally targets."""
    # each group of SEQ_LEN rows = one scenario
    X = df[cols].values.astype(np.float32)
    n = len(X) // SEQ_LEN
    X_seq = X.reshape(n, SEQ_LEN, len(cols))
    if target_col and target_col in df.columns:
        y = df[target_col].values.astype(np.float32).reshape(n, SEQ_LEN)
        return X_seq, y
    return X_seq

print("Reshaping to sequences...", flush=True)
X_seq, y_seq = to_sequences(train_fe, feat_cols, TARGET)
X_test_seq   = to_sequences(test_fe, feat_cols)
n_train = len(X_seq)
print(f"Train sequences: {n_train}, Test sequences: {len(X_test_seq)}", flush=True)

# Group IDs for CV (one group per scenario, grouped by layout_id)
layout_ids = train_fe.groupby(['layout_id','scenario_id'])['layout_id'].first().values  # (n_train,)

# ── Model ──
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)           # (B, T, H*2)
        return self.head(out).squeeze(-1)  # (B, T)

# ── Training ──
def train_fold(X_tr, y_tr, X_val, y_val, epochs=80, batch_size=256, lr=1e-3):
    X_tr_t  = torch.tensor(X_tr)
    y_tr_t  = torch.tensor(y_tr)
    X_val_t = torch.tensor(X_val)
    y_val_t = torch.tensor(y_val)

    ds  = TensorDataset(X_tr_t, y_tr_t)
    dl  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model = BiLSTM(N_FEAT).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit  = nn.L1Loss()

    best_val = float('inf'); best_state = None; patience = 15; no_improve = 0
    for ep in range(epochs):
        model.train()
        for Xb, yb in dl:
            opt.zero_grad()
            pred = torch.clamp(model(Xb), min=0)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            val_pred = torch.clamp(model(X_val_t), min=0).numpy()
        val_mae = np.mean(np.abs(val_pred - y_val))
        if val_mae < best_val:
            best_val = val_mae; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break
    model.load_state_dict(best_state)
    return model, best_val

gkf    = GroupKFold(n_splits=5)
oof    = np.zeros((n_train, SEQ_LEN))
tests  = []

print("Training BiLSTM (5-fold)...", flush=True)
for fold_i, (tr_idx, val_idx) in enumerate(gkf.split(X_seq, groups=layout_ids)):
    t0 = time.time()
    model, val_mae = train_fold(X_seq[tr_idx], y_seq[tr_idx],
                                X_seq[val_idx], y_seq[val_idx])
    model.eval()
    with torch.no_grad():
        val_pred  = torch.clamp(model(torch.tensor(X_seq[val_idx])), min=0).numpy()
        test_pred = torch.clamp(model(torch.tensor(X_test_seq)), min=0).numpy()
    oof[val_idx] = val_pred
    tests.append(test_pred)
    fold_mae = np.mean(np.abs(val_pred - y_seq[val_idx]))
    print(f"Fold {fold_i+1}: BiLSTM={fold_mae:.4f}  ({time.time()-t0:.0f}s)", flush=True)

# Flatten OOF back to row-level
oof_flat  = oof.reshape(-1)
y_flat    = y_seq.reshape(-1)
test_avg  = np.mean(tests, axis=0).reshape(-1)

oof_mae = np.mean(np.abs(oof_flat - y_flat))
print(f"\nBiLSTM OOF MAE: {oof_mae:.4f}", flush=True)

os.makedirs(os.path.join(ROOT, "results", "lstm"), exist_ok=True)
np.save(os.path.join(ROOT, "results", "lstm", "oof_lstm.npy"), oof_flat)
np.save(os.path.join(ROOT, "results", "lstm", "test_lstm.npy"), test_avg)
print("Saved.", flush=True)

# Blend eval vs FIXED
train_raw = pd.read_csv(os.path.join(ROOT, "train.csv")).sort_values(['layout_id','scenario_id']).reset_index(drop=True)
y_true = train_raw[TARGET].values
with open(os.path.join(ROOT, "results", "mega33_final.pkl"), "rb") as f:
    d = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
fixed = (fw['mega33']*d['meta_avg_oof']
       + fw['rank_adj']*np.load(os.path.join(ROOT,"results","ranking","rank_adj_oof.npy"))
       + fw['iter_r1']*np.load(os.path.join(ROOT,"results","iter_pseudo","round1_oof.npy"))
       + fw['iter_r2']*np.load(os.path.join(ROOT,"results","iter_pseudo","round2_oof.npy"))
       + fw['iter_r3']*np.load(os.path.join(ROOT,"results","iter_pseudo","round3_oof.npy")))

# oof_flat is in layout_id,scenario_id order = same as train_raw
fixed_mae = np.mean(np.abs(fixed - y_true))
lstm_corr = np.corrcoef(fixed, oof_flat)[0,1]
print(f"\nFIXED MAE: {fixed_mae:.4f}")
print(f"LSTM corr w/ FIXED: {lstm_corr:.4f}")

best_m = fixed_mae; best_w = 0
for w in np.arange(0.02, 0.51, 0.02):
    mm = np.mean(np.abs((1-w)*fixed + w*oof_flat - y_true))
    if mm < best_m: best_m = mm; best_w = w
print(f"FIXED+LSTM best: w={best_w:.2f}  MAE={best_m:.4f}  delta={best_m-fixed_mae:+.4f}")
print("Done.", flush=True)
