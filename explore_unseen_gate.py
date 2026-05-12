import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/user/Desktop/데이콘 4월")

# ── exact setup from finish_calibration.py ───────────────────────────────────
train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

train_layouts = set(train_raw['layout_id'].unique())
unseen_mask   = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask     = ~unseen_mask
print(f"Unseen test rows: {unseen_mask.sum()} / {len(test_raw)}")

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)

fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

mega33_oof  = d33['meta_avg_oof'][id2]
mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]
mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]
r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]

xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')

slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
slh_t  = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]
rh_t   = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]
slhm_t = np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

mae = lambda p: float(np.mean(np.abs(np.clip(p, 0, None) - y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega   = (1-w34)*mega33_oof  + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33']-dr2-dr3; w2 = fw['iter_r2']+dr2; w3 = fw['iter_r3']+dr3
    fx  = wm*mega   + fw['rank_adj']*rank_oof  + fw['iter_r1']*r1_oof  + w2*r2_oof  + w3*r3_oof
    fxt = wm*mega_t + fw['rank_adj']*rank_test + fw['iter_r1']*r1_test + w2*r2_test + w3*r3_test
    w_rem = 1.0-wf; wxgb=0.12*w_rem/0.36; wlv2=0.16*w_rem/0.36; wrem=0.08*w_rem/0.36
    oo = np.clip(wf*fx  + wxgb*xgb_o + wlv2*lv2_o + wrem*rem_o, 0, None)
    ot = np.clip(wf*fxt + wxgb*xgb_t + wlv2*lv2_t + wrem*rem_t, 0, None)
    if w_cb > 0:
        oo = np.clip((1-w_cb)*oo + w_cb*cb_oof,  0, None)
        ot = np.clip((1-w_cb)*ot + w_cb*cb_test, 0, None)
    return oo, ot

bb_o, bb_t = make_pred(0.25, -0.04, -0.02, 0.72, 0.12)
print(f"best_base: OOF={mae(bb_o):.5f}  test={bb_t.mean():.3f}")

fw4_o = np.clip(0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
fw4_t = np.clip(0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)
print(f"4way:      OOF={mae(fw4_o):.5f}  test={fw4_t.mean():.3f}  unseen={fw4_t[unseen_mask].mean():.3f}")

# dual gate baseline (best known)
n1, w1, n2, w2 = 2000, 0.15, 5500, 0.08
sfw = np.sort(fw4_o)
m1_o = fw4_o >= sfw[-n1]; m2_o = fw4_o >= sfw[-n2]
dual_o = fw4_o.copy()
dual_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
dual_o[m2_o] = (1-w2)*dual_o[m2_o] + w2*slhm_o[m2_o]
dual_mae = mae(dual_o)

sft = np.sort(fw4_t)
m1_t = fw4_t >= sft[-n1]; m2_t = fw4_t >= sft[-n2]
dual_t = fw4_t.copy()
dual_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
dual_t[m2_t] = (1-w2)*dual_t[m2_t] + w2*slhm_t[m2_t]
dual_t = np.clip(dual_t, 0, None)
print(f"dual_gate: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

# fold analysis
groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true), dtype=int)
for fi, (_, vi) in enumerate(gkf.split(train_raw, y_true, groups)):
    fold_ids[vi] = fi
fold_maes = [float(np.mean(np.abs(dual_o[fold_ids==fi] - y_true[fold_ids==fi]))) for fi in range(5)]
print(f"dual folds: {[f'{x:.4f}' for x in fold_maes]}")

print("\n" + "="*70)
print("Part 1: Unseen layout boost -- apply stronger gate weights to unseen rows only")
print("(OOF unchanged since all OOF rows are seen layouts)")
print("="*70)

def make_test_asym(n1_s, w1_s, n1_u, w1_u, n2_s, w2_s, n2_u, w2_u):
    sft_ = np.sort(fw4_t)
    m1_s_ = (fw4_t >= sft_[-n1_s]) & seen_mask
    m1_u_ = (fw4_t >= sft_[-n1_u]) & unseen_mask
    m2_s_ = (fw4_t >= sft_[-n2_s]) & seen_mask
    m2_u_ = (fw4_t >= sft_[-n2_u]) & unseen_mask
    m1f = m1_s_ | m1_u_; m2f = m2_s_ | m2_u_
    w1_arr = np.where(unseen_mask, w1_u, w1_s)
    w2_arr = np.where(unseen_mask, w2_u, w2_s)
    t = fw4_t.copy()
    t[m1f] = (1-w1_arr[m1f])*fw4_t[m1f] + w1_arr[m1f]*rh_t[m1f]
    t[m2f] = (1-w2_arr[m2f])*t[m2f]     + w2_arr[m2f]*slhm_t[m2f]
    return np.clip(t, 0, None)

# scan: same n, boost w for unseen
print("\n-- A) Same gate size, boost w for unseen --")
rows_a = []
for w1u in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    for w2u in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25]:
        t = make_test_asym(n1, w1, n1, w1u, n2, w2, n2, w2u)
        rows_a.append({'w1u':w1u,'w2u':w2u,'test':t.mean(),'seen':t[seen_mask].mean(),'unseen':t[unseen_mask].mean()})
df_a = pd.DataFrame(rows_a).sort_values('unseen', ascending=False)
print("Top 10 by unseen (same n, boost w):")
print(df_a.head(10).to_string(index=False))

# scan: extend gate n for unseen
print("\n-- B) Extend gate size (n) for unseen rows --")
rows_b = []
for n1u in [2000, 3000, 4000, 5000, 7000, 10000]:
    for n2u in [5500, 8000, 11000, 15000, 20000]:
        t = make_test_asym(n1, w1, n1u, w1, n2, w2, n2u, w2)
        rows_b.append({'n1u':n1u,'n2u':n2u,'test':t.mean(),'seen':t[seen_mask].mean(),'unseen':t[unseen_mask].mean()})
df_b = pd.DataFrame(rows_b).sort_values('unseen', ascending=False)
print("Top 10 by unseen (extend n):")
print(df_b.head(10).to_string(index=False))

# scan: combined boost (n + w)
print("\n-- C) Combined: wider n AND higher w for unseen --")
rows_c = []
best_unseen_b = df_b.iloc[0]
for n1u in [2000, 3000, 4000, 5000, 7000, 10000, 20000]:
    for w1u in [0.15, 0.20, 0.25, 0.30]:
        for n2u in [5500, 8000, 11000, 15000, 20000]:
            for w2u in [0.08, 0.12, 0.15, 0.18, 0.25]:
                t = make_test_asym(n1, w1, n1u, w1u, n2, w2, n2u, w2u)
                rows_c.append({'n1u':n1u,'w1u':w1u,'n2u':n2u,'w2u':w2u,
                               'test':t.mean(),'seen':t[seen_mask].mean(),'unseen':t[unseen_mask].mean()})
df_c = pd.DataFrame(rows_c).sort_values('unseen', ascending=False)
print("Top 15 by unseen (combined):")
print(df_c.head(15).to_string(index=False))

# pure seen-only gate: ignore unseen rows entirely
print("\n-- D) Gate on seen rows only (n_seen=2000, w1=0.15, n2=5500, w2=0.08 but unseen untouched) --")
t_seen_only = fw4_t.copy()
m1_s_only = (fw4_t >= sft[-n1]) & seen_mask
m2_s_only = (fw4_t >= sft[-n2]) & seen_mask
t_seen_only[m1_s_only] = (1-w1)*fw4_t[m1_s_only] + w1*rh_t[m1_s_only]
t_seen_only[m2_s_only] = (1-w2)*t_seen_only[m2_s_only] + w2*slhm_t[m2_s_only]
t_seen_only = np.clip(t_seen_only, 0, None)
print(f"Seen-only gate: test={t_seen_only.mean():.3f}  seen={t_seen_only[seen_mask].mean():.3f}  unseen={t_seen_only[unseen_mask].mean():.3f}")

# pure unseen-only gate variations
print("\n-- E) Gate on unseen rows only (apply full rh+slhm blend to ALL unseen) --")
rows_e = []
for w_rh in [0.10, 0.15, 0.20, 0.25, 0.30]:
    for w_slhm in [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        t = fw4_t.copy()
        # Apply full rh + slhm to ALL unseen rows (no gate threshold)
        t[unseen_mask] = np.clip(
            (1-w_rh)*fw4_t[unseen_mask] + w_rh*rh_t[unseen_mask], 0, None)
        t[unseen_mask] = np.clip(
            (1-w_slhm)*t[unseen_mask] + w_slhm*slhm_t[unseen_mask], 0, None)
        # Also apply standard seen gate
        t[m1_s_only] = (1-w1)*fw4_t[m1_s_only] + w1*rh_t[m1_s_only]
        t[m2_s_only] = (1-w2)*t[m2_s_only] + w2*slhm_t[m2_s_only]
        t = np.clip(t, 0, None)
        rows_e.append({'w_rh':w_rh,'w_slhm':w_slhm,
                       'test':t.mean(),'seen':t[seen_mask].mean(),'unseen':t[unseen_mask].mean()})
df_e = pd.DataFrame(rows_e).sort_values('unseen', ascending=False)
print("Top 10 (apply rh+slhm to ALL unseen + standard seen gate):")
print(df_e.head(10).to_string(index=False))

print("\n" + "="*70)
print("SUMMARY")
print("="*70)
best_c = df_c.iloc[0]
best_a = df_a.iloc[0]
best_e = df_e.iloc[0]
print(f"dual_gate (sym):    test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}  OOF={dual_mae:.5f}")
print(f"best A (w boost):   test={best_a.test:.3f}  seen={best_a.seen:.3f}  unseen={best_a.unseen:.3f}  OOF={dual_mae:.5f}")
print(f"best B (n extend):  test={df_b.iloc[0].test:.3f}  seen={df_b.iloc[0].seen:.3f}  unseen={df_b.iloc[0].unseen:.3f}  OOF={dual_mae:.5f}")
print(f"best C (n+w combo): test={best_c.test:.3f}  seen={best_c.seen:.3f}  unseen={best_c.unseen:.3f}  OOF={dual_mae:.5f}")
print(f"best E (full uns.): test={best_e.test:.3f}  seen={best_e.seen:.3f}  unseen={best_e.unseen:.3f}  OOF={dual_mae:.5f}")

# Save best submission
best_row = df_c.iloc[0]
best_test = make_test_asym(n1, w1, int(best_row.n1u), best_row.w1u,
                           n2, w2, int(best_row.n2u), best_row.w2u)
sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = best_test
fname = f"submission_unseenBoost_n1u{int(best_row.n1u)}_w1u{best_row.w1u}_n2u{int(best_row.n2u)}_w2u{best_row.w2u}_OOF{dual_mae:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"\nSaved: {fname}")
print(f"test={best_test.mean():.3f}  seen={best_test[seen_mask].mean():.3f}  unseen={best_test[unseen_mask].mean():.3f}")
