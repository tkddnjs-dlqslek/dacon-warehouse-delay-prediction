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
fold_mae_fn = lambda p: [float(np.mean(np.abs(p[fold_ids==fi] - y_true[fold_ids==fi]))) for fi in range(5)]

print("\n" + "="*70)
print("Part 1: Cascade models at extreme OOF predictions (top-500/1000)")
print("(To find best model for a 3rd gate tier below rh at n=2000)")
print("="*70)

# After dual gate, what are the actual residuals for top-500 predictions?
sfw2 = np.sort(dual_o)
t500_mask = dual_o >= sfw2[-500]
t1000_mask = dual_o >= sfw2[-1000]

y_top500 = y_true[t500_mask]
y_top1000 = y_true[t1000_mask]
dual_top500 = dual_o[t500_mask]
dual_top1000 = dual_o[t1000_mask]
print(f"Top-500 OOF predictions: dual_mae={np.mean(np.abs(dual_top500-y_top500)):.3f}  y_mean={y_top500.mean():.2f}  pred_mean={dual_top500.mean():.2f}")
print(f"Top-1000 OOF predictions: dual_mae={np.mean(np.abs(dual_top1000-y_top1000)):.3f}  y_mean={y_top1000.mean():.2f}  pred_mean={dual_top1000.mean():.2f}")

cascade_models = {
    'rh':   (rh_o, rh_t),
    'slhm': (slhm_o, slhm_t),
    'slh':  (slh_o, slh_t),
    'savg': (savg_o, savg_t),
}
print(f"\n{'Model':8s}  {'top500_MAE':>10s}  {'top1000_MAE':>11s}  {'top500_y':>9s}  {'top500_pred':>11s}")
for name, (oof_a, test_a) in cascade_models.items():
    m500 = np.mean(np.abs(oof_a[t500_mask] - y_top500))
    m1000 = np.mean(np.abs(oof_a[t1000_mask] - y_top1000))
    print(f"  {name:8s}  {m500:>10.3f}  {m1000:>11.3f}  {y_top500.mean():>9.3f}  {oof_a[t500_mask].mean():>11.3f}")

# Also check what dual_o predicts vs y at very top
print(f"\n  dual_gate  {np.mean(np.abs(dual_top500-y_top500)):>10.3f}  {np.mean(np.abs(dual_top1000-y_top1000)):>11.3f}  {y_top500.mean():>9.3f}  {dual_top500.mean():>11.3f}")

# Load all available cascade models
import glob
cascade_dir = 'results/cascade/'
all_casc = {}
for fp in sorted(glob.glob(os.path.join(cascade_dir, '*_oof.npy'))):
    fn = os.path.basename(fp).replace('_oof.npy','')
    test_fp = fp.replace('_oof.npy','_test.npy')
    if not os.path.exists(test_fp): continue
    oo = np.load(fp)[id2]; tt = np.load(test_fp)[te_id2]
    if len(oo) != len(y_true) or len(tt) != len(test_raw): continue
    m500_mae = np.mean(np.abs(oo[t500_mask] - y_top500))
    m1000_mae = np.mean(np.abs(oo[t1000_mask] - y_top1000))
    all_casc[fn] = (oo, tt, m500_mae, m1000_mae)
    print(f"  {fn:35s}  top500={m500_mae:8.3f}  top1000={m1000_mae:8.3f}  pred500={oo[t500_mask].mean():.2f}")

print("\n" + "="*70)
print("Part 2: Third tier gate -- add tier 0 at top-500/700 rows")
print("Apply tier 0 model after dual gate")
print("="*70)

# Sort cascade models by top-500 MAE
sorted_casc = sorted(all_casc.items(), key=lambda x: x[1][2])
best_casc_name, (best_casc_o, best_casc_t, best_m500, best_m1000) = sorted_casc[0]
print(f"Best cascade for top-500: {best_casc_name} (MAE={best_m500:.3f})")

# Build triple gate: dual_gate + tier0
rows = []
for n0, model_name in [(300,best_casc_name),(500,best_casc_name),(700,best_casc_name),(1000,best_casc_name)]:
    for w0 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        casc_o_here, casc_t_here = all_casc[model_name][0], all_casc[model_name][1]
        sfw_d = np.sort(dual_o)
        sft_d = np.sort(dual_t)
        m0_o = dual_o >= sfw_d[-n0]
        m0_t = dual_t >= sft_d[-n0]
        triple_o = dual_o.copy()
        triple_o[m0_o] = (1-w0)*dual_o[m0_o] + w0*casc_o_here[m0_o]
        triple_o = np.clip(triple_o, 0, None)
        triple_t = dual_t.copy()
        triple_t[m0_t] = (1-w0)*dual_t[m0_t] + w0*casc_t_here[m0_t]
        triple_t = np.clip(triple_t, 0, None)
        all_mae = mae_fn(triple_o)
        folds = fold_mae_fn(triple_o)
        delta = all_mae - dual_mae
        rows.append({'n0':n0,'w0':w0,'model':model_name,'OOF':all_mae,'delta':delta,
                     'test':triple_t.mean(),'unseen':triple_t[unseen_mask].mean(),
                     'folds':folds,'n_improved':sum(1 for fi,fd in zip(folds,fold_mae_fn(dual_o)) if fi<fd)})

print("\nTriple gate results (model={}):".format(best_casc_name))
rows_df = pd.DataFrame(rows).sort_values('OOF')
print(rows_df[['n0','w0','OOF','delta','test','unseen','n_improved']].head(20).to_string(index=False))

# Try rh as tier 0 (since rh is best at p99)
print("\nTriple gate with rh as tier 0:")
rows_rh = []
for n0 in [200, 300, 500, 700, 1000]:
    for w0 in [0.10, 0.15, 0.20, 0.25, 0.30]:
        sfw_d = np.sort(dual_o); sft_d = np.sort(dual_t)
        m0_o = dual_o >= sfw_d[-n0]; m0_t = dual_t >= sft_d[-n0]
        triple_o = dual_o.copy()
        triple_o[m0_o] = (1-w0)*dual_o[m0_o] + w0*rh_o[m0_o]
        triple_o = np.clip(triple_o, 0, None)
        triple_t = dual_t.copy()
        triple_t[m0_t] = (1-w0)*dual_t[m0_t] + w0*rh_t[m0_t]
        triple_t = np.clip(triple_t, 0, None)
        all_mae = mae_fn(triple_o)
        delta = all_mae - dual_mae
        rows_rh.append({'n0':n0,'w0':w0,'OOF':all_mae,'delta':delta,
                        'test':triple_t.mean(),'unseen':triple_t[unseen_mask].mean()})

df_rh = pd.DataFrame(rows_rh).sort_values('OOF')
print(df_rh.head(15).to_string(index=False))

print("\n" + "="*70)
print("Part 3: Best triple gate -- save top candidates")
print("="*70)

all_results = []
for casc_name, (casc_o_v, casc_t_v, _, _) in all_casc.items():
    for n0 in [300, 500, 700, 1000]:
        for w0 in [0.10, 0.15, 0.20, 0.25]:
            sfw_d = np.sort(dual_o); sft_d = np.sort(dual_t)
            m0_o = dual_o >= sfw_d[-n0]; m0_t = dual_t >= sft_d[-n0]
            triple_o = dual_o.copy()
            triple_o[m0_o] = (1-w0)*dual_o[m0_o] + w0*casc_o_v[m0_o]
            triple_o = np.clip(triple_o, 0, None)
            triple_t = dual_t.copy()
            triple_t[m0_t] = (1-w0)*dual_t[m0_t] + w0*casc_t_v[m0_t]
            triple_t = np.clip(triple_t, 0, None)
            oof = mae_fn(triple_o)
            if oof < dual_mae:
                folds = fold_mae_fn(triple_o)
                all_results.append({
                    'name':casc_name,'n0':n0,'w0':w0,'OOF':oof,'delta':oof-dual_mae,
                    'test':triple_t.mean(),'unseen':triple_t[unseen_mask].mean(),
                    'n_improved':sum(1 for fi,fd in zip(folds,fold_mae_fn(dual_o)) if fi<fd)
                })

if all_results:
    df_all = pd.DataFrame(all_results).sort_values('OOF')
    print(f"Triple gate configs that improve OOF ({len(all_results)} found):")
    print(df_all.head(20).to_string(index=False))

    best = df_all.iloc[0]
    casc_o_b, casc_t_b = all_casc[best.name][0], all_casc[best.name][1]
    sfw_d = np.sort(dual_o); sft_d = np.sort(dual_t)
    m0_o = dual_o >= sfw_d[-int(best.n0)]
    m0_t = dual_t >= sft_d[-int(best.n0)]
    best_triple_o = dual_o.copy()
    best_triple_o[m0_o] = (1-best.w0)*dual_o[m0_o] + best.w0*casc_o_b[m0_o]
    best_triple_o = np.clip(best_triple_o, 0, None)
    best_triple_t = dual_t.copy()
    best_triple_t[m0_t] = (1-best.w0)*dual_t[m0_t] + best.w0*casc_t_b[m0_t]
    best_triple_t = np.clip(best_triple_t, 0, None)

    folds_best = fold_mae_fn(best_triple_o)
    print(f"\nBEST triple gate: model={best.name}, n0={int(best.n0)}, w0={best.w0}")
    print(f"  OOF={mae_fn(best_triple_o):.5f} ({mae_fn(best_triple_o)-dual_mae:+.6f})")
    print(f"  test={best_triple_t.mean():.3f}  unseen={best_triple_t[unseen_mask].mean():.3f}")
    print(f"  folds={[f'{x:.4f}' for x in folds_best]}")
    print(f"  n_improved={best.n_improved}/5")

    sub = pd.read_csv('sample_submission.csv')
    sub['avg_delay_minutes_next_30m'] = best_triple_t
    fname = f"submission_tripleGate_{best.name}_n0{int(best.n0)}_w0{best.w0}_OOF{mae_fn(best_triple_o):.5f}.csv"
    sub.to_csv(fname, index=False)
    print(f"Saved: {fname}")
else:
    print("No triple gate config improved OOF.")
    print("Dual gate remains best.")

print("\n" + "="*70)
print("FINAL COMPARISON")
print("="*70)
print(f"4way base:   OOF=8.37624  test=19.413  unseen=22.802")
print(f"dual_gate:   OOF={dual_mae:.5f}  test={dual_t.mean():.3f}  unseen={dual_t[unseen_mask].mean():.3f}")
if all_results:
    print(f"triple_gate: OOF={df_all.iloc[0].OOF:.5f}  test={df_all.iloc[0].test:.3f}  unseen={df_all.iloc[0].unseen:.3f}")
