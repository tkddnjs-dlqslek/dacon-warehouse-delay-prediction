import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle, os
from sklearn.model_selection import GroupKFold

os.chdir("C:/Users/user/Desktop/데이콘 4월")

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

with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)

mega33_oof  = d33['meta_avg_oof'][id2]; mega33_test = d33['meta_avg_test'][te_id2]
mega34_oof  = d34['meta_avg_oof'][id2]; mega34_test = d34['meta_avg_test'][te_id2]
cb_oof  = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
cb_test = np.clip(d33['meta_tests']['cb'][te_id2], 0, None)
rank_oof  = np.load('results/ranking/rank_adj_oof.npy')[id2]
rank_test = np.load('results/ranking/rank_adj_test.npy')[te_id2]
r1_oof  = np.load('results/iter_pseudo/round1_oof.npy')[id2]; r1_test = np.load('results/iter_pseudo/round1_test.npy')[te_id2]
r2_oof  = np.load('results/iter_pseudo/round2_oof.npy')[id2]; r2_test = np.load('results/iter_pseudo/round2_test.npy')[te_id2]
r3_oof  = np.load('results/iter_pseudo/round3_oof.npy')[id2]; r3_test = np.load('results/iter_pseudo/round3_test.npy')[te_id2]
xgb_o = np.load('results/oracle_seq/oof_seqC_xgb.npy');       xgb_t = np.load('results/oracle_seq/test_C_xgb.npy')
lv2_o = np.load('results/oracle_seq/oof_seqC_log_v2.npy');    lv2_t = np.load('results/oracle_seq/test_C_log_v2.npy')
rem_o = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy'); rem_t = np.load('results/oracle_seq/test_C_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy'); xgbc_t = np.load('results/oracle_seq/test_C_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy'); mono_t = np.load('results/oracle_seq/test_C_xgb_monotone.npy')
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]; slh_t  = np.load('results/cascade/spec_lgb_w30_huber_test.npy')[te_id2]
rh_o   = np.load('results/cascade/spec_lgb_raw_huber_oof.npy')[id2]; rh_t   = np.load('results/cascade/spec_lgb_raw_huber_test.npy')[te_id2]
slhm_o = np.load('results/cascade/spec_lgb_w30_mae_oof.npy')[id2];   slhm_t = np.load('results/cascade/spec_lgb_w30_mae_test.npy')[te_id2]
savg_o = np.load('results/cascade/spec_avg_oof.npy')[id2];            savg_t = np.load('results/cascade/spec_avg_test.npy')[te_id2]

mae_fn = lambda p: float(np.mean(np.abs(np.clip(p, 0, None) - y_true)))

def make_pred(w34=0.0, dr2=-0.04, dr3=-0.02, wf=0.64, w_cb=0.0):
    mega = (1-w34)*mega33_oof + w34*mega34_oof
    mega_t = (1-w34)*mega33_test + w34*mega34_test
    wm = fw['mega33']-dr2-dr3; w2=fw['iter_r2']+dr2; w3=fw['iter_r3']+dr3
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
fw4_o = np.clip(0.74*bb_o + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
fw4_t = np.clip(0.74*bb_t + 0.08*slh_t + 0.10*xgbc_t + 0.08*mono_t, 0, None)

n1, w1, n2, w2 = 2000, 0.15, 5500, 0.08
sfw = np.sort(fw4_o); sft = np.sort(fw4_t)
m1_o = fw4_o >= sfw[-n1]; m2_o = fw4_o >= sfw[-n2]
dual_o = fw4_o.copy()
dual_o[m1_o] = (1-w1)*fw4_o[m1_o] + w1*rh_o[m1_o]
dual_o[m2_o] = (1-w2)*dual_o[m2_o] + w2*slhm_o[m2_o]
dual_mae = mae_fn(dual_o)

m1_t = fw4_t >= sft[-n1]; m2_t = fw4_t >= sft[-n2]
dual_t = fw4_t.copy()
dual_t[m1_t] = (1-w1)*fw4_t[m1_t] + w1*rh_t[m1_t]
dual_t[m2_t] = (1-w2)*dual_t[m2_t] + w2*slhm_t[m2_t]
dual_t = np.clip(dual_t, 0, None)
print(f"dual_gate: OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  seen={dual_t[seen_mask].mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")

groups = train_raw['layout_id'].values
gkf = GroupKFold(n_splits=5)
fold_ids = np.zeros(len(y_true), dtype=int)
for fi, (_, vi) in enumerate(gkf.split(train_raw, y_true, groups)):
    fold_ids[vi] = fi

print("\n" + "="*70)
print("Part 1: Proxy-OOF for unseen boost using high-inflow seen layouts")
print("(Find seen train layouts most similar to unseen test by inflow)")
print("="*70)

# Analyze inflow distribution
inflow_col = 'order_inflow_15m'
train_inflow_by_layout = train_raw.groupby('layout_id')[inflow_col].mean()
test_inflow_by_layout  = test_raw.groupby('layout_id')[inflow_col].mean()

unseen_test_layouts = [l for l in test_raw['layout_id'].unique() if l not in train_layouts]
seen_test_layouts   = [l for l in test_raw['layout_id'].unique() if l in train_layouts]

unseen_inflow_vals = test_inflow_by_layout.loc[unseen_test_layouts]
seen_inflow_vals   = train_inflow_by_layout

print(f"Unseen test layouts inflow: mean={unseen_inflow_vals.mean():.1f}  median={unseen_inflow_vals.median():.1f}")
print(f"Seen train layouts inflow:  mean={seen_inflow_vals.mean():.1f}  median={seen_inflow_vals.median():.1f}")
p_threshold = unseen_inflow_vals.quantile(0.25)  # 25th percentile of unseen inflow
print(f"25th pct of unseen inflow: {p_threshold:.1f}")

# Find train layouts with inflow >= 25th pct of unseen inflow
high_inflow_layouts = seen_inflow_vals[seen_inflow_vals >= p_threshold].index.tolist()
hi_mask = train_raw['layout_id'].isin(high_inflow_layouts)
print(f"Train layouts with inflow >= {p_threshold:.1f}: {len(high_inflow_layouts)} / {len(seen_inflow_vals)}")
print(f"Train rows in high-inflow seen layouts: {hi_mask.sum()} / {len(train_raw)}")

# Proxy OOF: apply dual gate on OOF predictions, evaluate only on high-inflow rows
# Standard dual gate on OOF
dual_mae_hi = float(np.mean(np.abs(np.clip(dual_o[hi_mask], 0, None) - y_true[hi_mask])))
fw4_mae_hi  = float(np.mean(np.abs(np.clip(fw4_o[hi_mask], 0, None) - y_true[hi_mask])))
print(f"\nHigh-inflow OOF: fw4={fw4_mae_hi:.5f}  dual={dual_mae_hi:.5f}  delta={dual_mae_hi-fw4_mae_hi:+.5f}")

# spec_avg on high-inflow rows
savg_mae_hi = float(np.mean(np.abs(np.clip(savg_o[hi_mask], 0, None) - y_true[hi_mask])))
print(f"spec_avg on high-inflow OOF: {savg_mae_hi:.5f}")

# Now proxy-test the unseen boost by applying it to high-inflow OOF rows
# The idea: pretend high-inflow OOF rows are "proxy unseen" rows and measure improvement
print("\n-- Proxy unseen boost on high-inflow OOF rows --")
m1_s_oof = (fw4_o >= sfw[-n1]) & ~hi_mask
m2_s_oof = (fw4_o >= sfw[-n2]) & ~hi_mask

rows = []
for n1u in [2000, 4000, 7000, 10000, 20000]:
    for w1u in [0.15, 0.25, 0.35]:
        for n2u in [5500, 8000, 15000]:
            for w2u in [0.08, 0.15, 0.25]:
                # seen rows: standard dual gate
                m1_u_oof = (fw4_o >= sfw[-n1u]) & hi_mask if n1u < len(fw4_o) else hi_mask
                m2_u_oof = (fw4_o >= sfw[-n2u]) & hi_mask if n2u < len(fw4_o) else hi_mask
                m1f_oof = m1_s_oof | m1_u_oof
                m2f_oof = m2_s_oof | m2_u_oof
                w1a_oof = np.where(hi_mask, w1u, w1)
                w2a_oof = np.where(hi_mask, w2u, w2)
                o = fw4_o.copy()
                o[m1f_oof] = (1-w1a_oof[m1f_oof])*fw4_o[m1f_oof] + w1a_oof[m1f_oof]*rh_o[m1f_oof]
                o[m2f_oof] = (1-w2a_oof[m2f_oof])*o[m2f_oof]     + w2a_oof[m2f_oof]*slhm_o[m2f_oof]
                o = np.clip(o, 0, None)
                all_mae = mae_fn(o)
                hi_mae  = float(np.mean(np.abs(o[hi_mask] - y_true[hi_mask])))
                rows.append({'n1u':n1u,'w1u':w1u,'n2u':n2u,'w2u':w2u,
                             'all_mae':all_mae,'hi_mae':hi_mae,
                             'delta_all':all_mae-dual_mae,'delta_hi':hi_mae-dual_mae_hi})

df = pd.DataFrame(rows)
df_best_hi = df.sort_values('hi_mae')
print("\nTop 15 by hi_mae improvement (proxy unseen OOF):")
print(df_best_hi.head(15)[['n1u','w1u','n2u','w2u','all_mae','hi_mae','delta_all','delta_hi']].to_string(index=False))

df_best_all = df.sort_values('all_mae')
print("\nTop 10 by all_mae (standard OOF):")
print(df_best_all.head(10)[['n1u','w1u','n2u','w2u','all_mae','hi_mae','delta_all','delta_hi']].to_string(index=False))

print("\n" + "="*70)
print("Part 2: spec_avg as unseen specialist")
print("(spec_avg has highest unseen=25.964 and solo_OOF=8.909)")
print("="*70)

print(f"spec_avg: OOF={mae_fn(savg_o):.5f}  test={savg_t.mean():.3f}  unseen={savg_t[unseen_mask].mean():.3f}")
print(f"Corr(spec_avg, dual_o): {np.corrcoef(savg_o, dual_o)[0,1]:.4f}")

m1_s_t = (fw4_t >= sft[-n1]) & seen_mask
m2_s_t = (fw4_t >= sft[-n2]) & seen_mask

rows2 = []
for w_sa in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    # Apply spec_avg blend to unseen test rows
    t = dual_t.copy()
    t[unseen_mask] = np.clip((1-w_sa)*dual_t[unseen_mask] + w_sa*savg_t[unseen_mask], 0, None)
    rows2.append({'w_sa':w_sa,'test':t.mean(),'seen':t[seen_mask].mean(),'unseen':t[unseen_mask].mean()})

    # Also compute proxy OOF improvement on high-inflow rows
    o2 = dual_o.copy()
    o2[hi_mask] = np.clip((1-w_sa)*dual_o[hi_mask] + w_sa*savg_o[hi_mask], 0, None)
    proxy_mae = float(np.mean(np.abs(o2[hi_mask] - y_true[hi_mask])))
    print(f"  w_sa={w_sa:.2f}: unseen={t[unseen_mask].mean():.3f}  test={t.mean():.3f}  proxy_hi_mae={proxy_mae:.5f} (delta={proxy_mae-dual_mae_hi:+.5f})")

print("\n" + "="*70)
print("Part 3: Best proxy-OOF-validated unseen boost configs")
print("="*70)

best_hi_row = df_best_hi.iloc[0]
n1u_b = int(best_hi_row.n1u); w1u_b = best_hi_row.w1u
n2u_b = int(best_hi_row.n2u); w2u_b = best_hi_row.w2u
print(f"Best proxy: n1u={n1u_b}, w1u={w1u_b}, n2u={n2u_b}, w2u={w2u_b}")
print(f"  all_mae={best_hi_row.all_mae:.5f}  hi_mae={best_hi_row.hi_mae:.5f}")
print(f"  delta_all={best_hi_row.delta_all:+.5f}  delta_hi={best_hi_row.delta_hi:+.5f}")

# Build test version
m1_u_t = (fw4_t >= sft[-n1u_b]) & unseen_mask if n1u_b < len(fw4_t) else unseen_mask
m2_u_t = (fw4_t >= sft[-n2u_b]) & unseen_mask if n2u_b < len(fw4_t) else unseen_mask
m1f_t = m1_s_t | m1_u_t; m2f_t = m2_s_t | m2_u_t
w1a_t = np.where(unseen_mask, w1u_b, w1); w2a_t = np.where(unseen_mask, w2u_b, w2)
best_proxy_t = fw4_t.copy()
best_proxy_t[m1f_t] = (1-w1a_t[m1f_t])*fw4_t[m1f_t] + w1a_t[m1f_t]*rh_t[m1f_t]
best_proxy_t[m2f_t] = (1-w2a_t[m2f_t])*best_proxy_t[m2f_t] + w2a_t[m2f_t]*slhm_t[m2f_t]
best_proxy_t = np.clip(best_proxy_t, 0, None)
print(f"  test={best_proxy_t.mean():.3f}  seen={best_proxy_t[seen_mask].mean():.3f}  unseen={best_proxy_t[unseen_mask].mean():.3f}")

sub = pd.read_csv('sample_submission.csv')
sub['avg_delay_minutes_next_30m'] = best_proxy_t
fname = f"submission_proxyOOF_n1u{n1u_b}_w1u{w1u_b}_n2u{n2u_b}_w2u{w2u_b}_OOF{dual_mae:.5f}.csv"
sub.to_csv(fname, index=False)
print(f"Saved: {fname}")

print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"dual_gate (baseline):  OOF={dual_mae:.5f}  test=19.939  unseen=23.479")
print(f"best proxy-OOF boost:  OOF={best_hi_row.all_mae:.5f}  test={best_proxy_t.mean():.3f}  unseen={best_proxy_t[unseen_mask].mean():.3f}")
