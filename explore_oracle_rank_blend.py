import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os

os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
test_raw  = pd.read_csv('test.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
test_raw['_row_id']  = test_raw['ID'].str.replace('TEST_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
test_raw  = test_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values
train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values
seen_mask = ~unseen_mask
mae_fn = lambda p: float(np.mean(np.abs(np.clip(p,0,None)-y_true)))
id_order = test_raw['ID'].values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos   = {row['ID']:i for i,row in train_ls.iterrows()}
id2      = [ls_pos[i] for i in train_raw['ID'].values]
test_ls  = pd.read_csv('test.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
te_ls_pos = {row['ID']:i for i,row in test_ls.iterrows()}
te_id2    = [te_ls_pos[i] for i in test_raw['ID'].values]

xgb_o=np.load('results/oracle_seq/oof_seqC_xgb.npy'); xgb_t=np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o=np.load('results/oracle_seq/oof_seqC_log_v2.npy'); lv2_t=np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o=np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t=np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o=np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t=np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o=np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t=np.load('results/oracle_seq/test_C_xgb_monotone.npy')
rank_oof=np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test=np.load('results/ranking/rank_adj_test.npy')[te_id2]
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof=d33['meta_avg_oof'][id2]; mega34_oof=d34['meta_avg_oof'][id2]
mega33_test=d33['meta_avg_test'][te_id2]; mega34_test=d34['meta_avg_test'][te_id2]
cb_oof=np.clip(d33['meta_oofs']['cb'][id2],0,None); cb_test=np.clip(d33['meta_tests']['cb'][te_id2],0,None)
r1_oof=np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test=np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof=np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test=np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof=np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test=np.load('results/iter_pseudo/round3_test.npy')[te_id2]
slh_o=np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t=np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o=np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t=np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o=np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2]; slhm_t=np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]

w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof=(1-w34)*mega33_oof+w34*mega34_oof; mega_t=(1-w34)*mega33_test+w34*mega34_test
wm=fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
fx_o=wm*mega_oof+fw['rank_adj']*rank_oof+fw['iter_r1']*r1_oof+w2*r2_oof+w3*r3_oof
fxt=wm*mega_t+fw['rank_adj']*rank_test+fw['iter_r1']*r1_test+w2*r2_test+w3*r3_test

# oracle_NEW
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t  = oracle_new_df['avg_delay_minutes_next_30m'].values

# oracle_5way
oracle_t_eq = np.clip((xgb_t+lv2_t+rem_t+xgbc_t+mono_t)/5, 0, None)
oracle_o_eq = np.clip((xgb_o+lv2_o+rem_o+xgbc_o+mono_o)/5, 0, None)

sub_tmpl = pd.read_csv('sample_submission.csv')

# ============================================================
print("="*70)
print("Part 1: rank_adj model characteristics")
print("="*70)
rank_oof_c = np.clip(rank_oof, 0, None)
rank_test_c = np.clip(rank_test, 0, None)
print(f"rank_adj OOF: {mae_fn(rank_oof_c):.5f}")
print(f"rank_adj test: mean={rank_test_c.mean():.3f}  seen={rank_test_c[seen_mask].mean():.3f}  unseen={rank_test_c[unseen_mask].mean():.3f}")

from scipy.stats import pearsonr
r_rank_oracle, _ = pearsonr(rank_test_c, oracle_new_t)
r_rank_5way, _ = pearsonr(rank_test_c, oracle_t_eq)
print(f"\nr(rank_test, oracle_NEW) = {r_rank_oracle:.4f}")
print(f"r(rank_test, oracle_5way) = {r_rank_5way:.4f}")

# ============================================================
print("\n" + "="*70)
print("Part 2: Oracle + rank_adj blends")
print("="*70)
print(f"\n{'Config':40s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")
print("-"*70)

# rank_adj OOF
r_oof = mae_fn(rank_oof_c)
oracle5_oof = mae_fn(oracle_o_eq)

for w_rank in [0.05, 0.10, 0.15, 0.20]:
    # oracle_NEW + rank_adj
    ct_on = np.clip((1-w_rank)*oracle_new_t + w_rank*rank_test_c, 0, None)
    print(f"  oracleNEW+rank{w_rank:.2f}:                 {'   ---  ':>9}  {ct_on[seen_mask].mean():8.3f}  {ct_on[unseen_mask].mean():8.3f}")

for w_rank in [0.05, 0.10, 0.15, 0.20]:
    # oracle_5way + rank_adj
    oo_r = np.clip((1-w_rank)*oracle_o_eq + w_rank*rank_oof_c, 0, None)
    ct_5r = np.clip((1-w_rank)*oracle_t_eq + w_rank*rank_test_c, 0, None)
    # OOF for oracle_5way + rank_adj: we can compute
    oof_r = mae_fn(oo_r)
    print(f"  oracle5way+rank{w_rank:.2f}:                 {oof_r:9.5f}  {ct_5r[seen_mask].mean():8.3f}  {ct_5r[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 3: oracle_5way + rank_adj OOF grid")
print("="*70)
print(f"\n{'w_rank':8s}  {'OOF':>9}  {'seen':>8}  {'unseen':>8}")
for w_rank in np.arange(0, 0.31, 0.02):
    oo_r = np.clip((1-w_rank)*oracle_o_eq + w_rank*rank_oof_c, 0, None)
    oof_r = mae_fn(oo_r)
    ct_5r = np.clip((1-w_rank)*oracle_t_eq + w_rank*rank_test_c, 0, None)
    print(f"  {w_rank:.2f}      {oof_r:9.5f}  {ct_5r[seen_mask].mean():8.3f}  {ct_5r[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 4: Three-way blends: oracle_NEW + oracle_5way + rank_adj")
print("="*70)
# oracle_NEW (best LB), oracle_5way (lower seen/unseen), rank_adj (ranking-based)
# Can three-way blend find a better sweet spot?

print(f"\noracle_NEW: seen=17.046, unseen=22.716, OOF=8.38247")
print(f"oracle_5way: seen=16.789, unseen=22.358, OOF=8.40935")
print(f"rank_adj: seen={rank_test_c[seen_mask].mean():.3f}, unseen={rank_test_c[unseen_mask].mean():.3f}, OOF={r_oof:.5f}")
print()

print(f"{'w_oracle':8s}  {'w_5way':6s}  {'w_rank':6s}  {'seen':>8}  {'unseen':>8}")
for w_oracle, w_5way, w_rank in [
    (0.8, 0.1, 0.1),
    (0.7, 0.2, 0.1),
    (0.7, 0.1, 0.2),
    (0.6, 0.2, 0.2),
    (0.5, 0.3, 0.2),
    (0.5, 0.4, 0.1),
    (0.6, 0.3, 0.1),
    (0.9, 0.1, 0.0),
    (0.8, 0.2, 0.0),
]:
    ct = np.clip(w_oracle*oracle_new_t + w_5way*oracle_t_eq + w_rank*rank_test_c, 0, None)
    print(f"  {w_oracle:.1f}       {w_5way:.1f}     {w_rank:.1f}     {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 5: iter_pseudo models + oracle blend")
print("="*70)
# iter_pseudo rounds 1-3 have OOF 8.37-8.38, different distribution
r1_t_c = np.clip(r1_test, 0, None); r2_t_c = np.clip(r2_test, 0, None); r3_t_c = np.clip(r3_test, 0, None)
r1_o_c = np.clip(r1_oof, 0, None); r2_o_c = np.clip(r2_oof, 0, None); r3_o_c = np.clip(r3_oof, 0, None)

print(f"iter_r1 OOF: {mae_fn(r1_o_c):.5f}  seen={r1_t_c[seen_mask].mean():.3f}  unseen={r1_t_c[unseen_mask].mean():.3f}")
print(f"iter_r2 OOF: {mae_fn(r2_o_c):.5f}  seen={r2_t_c[seen_mask].mean():.3f}  unseen={r2_t_c[unseen_mask].mean():.3f}")
print(f"iter_r3 OOF: {mae_fn(r3_o_c):.5f}  seen={r3_t_c[seen_mask].mean():.3f}  unseen={r3_t_c[unseen_mask].mean():.3f}")
print(f"r(r1_test, oracle_NEW) = {pearsonr(r1_t_c, oracle_new_t)[0]:.4f}")
print(f"r(r2_test, oracle_NEW) = {pearsonr(r2_t_c, oracle_new_t)[0]:.4f}")

# Oracle_NEW + iter_pseudo
for w_iter in [0.05, 0.10, 0.15]:
    ct = np.clip((1-w_iter)*oracle_new_t + w_iter*r1_t_c, 0, None)
    print(f"  oracleNEW+r1_{w_iter:.2f}: seen={ct[seen_mask].mean():.3f}  unseen={ct[unseen_mask].mean():.3f}")

# ============================================================
print("\n" + "="*70)
print("Part 6: ALL models seen/unseen sorted — find clusters")
print("="*70)

all_model_preds = {
    'oracle_NEW':  oracle_new_t,
    'oracle_5way': oracle_t_eq,
    'xgb_only':    np.clip(xgb_t,0,None),
    'lv2_only':    np.clip(lv2_t,0,None),
    'rem_only':    np.clip(rem_t,0,None),
    'xgbc_only':   np.clip(xgbc_t,0,None),
    'mono_only':   np.clip(mono_t,0,None),
    'rank_adj':    rank_test_c,
    'r1_pseudo':   r1_t_c,
    'r2_pseudo':   r2_t_c,
}

print(f"\n{'Model':20s}  {'seen':>8}  {'unseen':>8}  Notes")
items = sorted(all_model_preds.items(), key=lambda x: x[1][seen_mask].mean())
for name, pred in items:
    flag = " *** BEST" if name == 'oracle_NEW' else ""
    print(f"  {name:20s}  {pred[seen_mask].mean():8.3f}  {pred[unseen_mask].mean():8.3f}{flag}")

# ============================================================
print("\n" + "="*70)
print("Part 7: Save final oracle variant candidates")
print("="*70)

to_save = []

# oracle_NEW + oracle_5way blends
for w5, label in [(0.1,'oN90o5w10'), (0.2,'oN80o5w20'), (0.3,'oN70o5w30'), (0.5,'oN50o5w50')]:
    ct = np.clip((1-w5)*oracle_new_t + w5*oracle_t_eq, 0, None)
    to_save.append((label, ct))

# oracle_5way (pure)
to_save.append(('oracle5way_pure', oracle_t_eq))

# rem_xgb_lv2 equal
rem_xgb_lv2 = np.clip((rem_t+xgb_t+lv2_t)/3, 0, None)
to_save.append(('rem_xgb_lv2_3way', rem_xgb_lv2))

# Three-way: oracle_NEW 70 + oracle_5way 20 + rank 10
ct_3way = np.clip(0.7*oracle_new_t + 0.2*oracle_t_eq + 0.1*rank_test_c, 0, None)
to_save.append(('oN70_5w20_rank10', ct_3way))

# oracle_5way + rank_adj 0.10
ct_5r10 = np.clip(0.9*oracle_t_eq + 0.1*rank_test_c, 0, None)
to_save.append(('oracle5w90_rank10', ct_5r10))

# Check OOFs for reference
# oracle_5way + rank 10% OOF:
oo_5r = np.clip(0.9*oracle_o_eq + 0.1*rank_oof_c, 0, None)
oof_5r = mae_fn(oo_5r)
print(f"\noracle5way+rank10 OOF: {oof_5r:.5f}")
print(f"oracle5way pure OOF: {mae_fn(oracle_o_eq):.5f}")

print(f"\n{'Filename':65s}  {'seen':>8}  {'unseen':>8}")
for label, ct in to_save:
    fname = f"FINAL_NEW_{label}_OOF_oracle.csv"
    sub = sub_tmpl.copy(); sub['avg_delay_minutes_next_30m'] = ct
    sub.to_csv(fname, index=False)
    print(f"  SAVED  {fname[:63]:63s}  {ct[seen_mask].mean():8.3f}  {ct[unseen_mask].mean():8.3f}")

print("\nDone.")
