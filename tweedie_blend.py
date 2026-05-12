import sys; sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, pickle
import warnings; warnings.filterwarnings('ignore')
import os
os.chdir("C:/Users/user/Desktop/데이콘 4월")

train_raw = pd.read_csv('train.csv')
train_raw['_row_id'] = train_raw['ID'].str.replace('TRAIN_','').astype(int)
train_raw = train_raw.sort_values('_row_id').reset_index(drop=True)
y_true = train_raw['avg_delay_minutes_next_30m'].values

test_raw = pd.read_csv('test.csv')
test_raw['_row_id'] = test_raw['ID'].str.replace('TEST_','').astype(int)
test_raw = test_raw.sort_values('_row_id').reset_index(drop=True)
id_order = test_raw['ID'].values

train_layouts = set(train_raw['layout_id'].unique())
unseen_mask = ~test_raw['layout_id'].isin(train_layouts).values

train_ls = pd.read_csv('train.csv').sort_values(['layout_id','scenario_id']).reset_index(drop=True)
ls_pos = {row['ID']:i for i,row in train_ls.iterrows()}
id2 = [ls_pos[i] for i in train_raw['ID'].values]

# Reconstruct oracle_NEW OOF and test
with open('results/mega33_final.pkl','rb') as f: d33 = pickle.load(f)
with open('results/mega34_final.pkl','rb') as f: d34 = pickle.load(f)
xgb_o  = np.load('results/oracle_seq/oof_seqC_xgb.npy')
lv2_o  = np.load('results/oracle_seq/oof_seqC_log_v2.npy')
rem_o  = np.load('results/oracle_seq/oof_seqC_xgb_remaining.npy')
xgbc_o = np.load('results/oracle_seq/oof_seqC_xgb_combined.npy')
mono_o = np.load('results/oracle_seq/oof_seqC_xgb_monotone.npy')
fw = dict(mega33=0.7636614598089654, rank_adj=0.1588758398901156,
          iter_r1=0.011855567572749024, iter_r2=0.034568307, iter_r3=0.031038826)
mega33_oof = d33['meta_avg_oof'][id2]; mega34_oof = d34['meta_avg_oof'][id2]
cb_oof_mega = np.clip(d33['meta_oofs']['cb'][id2], 0, None)
rank_oof = np.load('results/ranking/rank_adj_oof.npy')[id2]
r1_oof = np.load('results/iter_pseudo/round1_oof.npy')[id2]
r2_oof = np.load('results/iter_pseudo/round2_oof.npy')[id2]
r3_oof = np.load('results/iter_pseudo/round3_oof.npy')[id2]
slh_o  = np.load('results/cascade/spec_lgb_w30_huber_oof.npy')[id2]
w34=0.25; dr2=-0.04; dr3=-0.02; wf=0.72; w_cb=0.12
mega_oof = (1-w34)*mega33_oof + w34*mega34_oof
wm = fw['mega33']-dr2-dr3; w2_ = fw['iter_r2']+dr2; w3_ = fw['iter_r3']+dr3
fx_o = wm*mega_oof + fw['rank_adj']*rank_oof + fw['iter_r1']*r1_oof + w2_*r2_oof + w3_*r3_oof
w_rem2 = 1-wf; wxgb = 0.12*w_rem2/0.36; wlv2 = 0.16*w_rem2/0.36; wrem2 = 0.08*w_rem2/0.36
bb_oo = np.clip(wf*fx_o + wxgb*xgb_o + wlv2*lv2_o + wrem2*rem_o, 0, None)
bb_oo = np.clip((1-w_cb)*bb_oo + w_cb*cb_oof_mega, 0, None)
fw4_oo = np.clip(0.74*bb_oo + 0.08*slh_o + 0.10*xgbc_o + 0.08*mono_o, 0, None)
oracle_mae = np.mean(np.abs(y_true - fw4_oo))

# oracle_NEW test predictions
oracle_new_df = pd.read_csv('submission_oracle_NEW_OOF8.3825.csv')
oracle_new_df = oracle_new_df.set_index('ID').reindex(id_order).reset_index()
oracle_new_t = oracle_new_df['avg_delay_minutes_next_30m'].values

print(f"oracle_NEW: OOF={oracle_mae:.4f}  seen={oracle_new_t[~unseen_mask].mean():.3f}  unseen={oracle_new_t[unseen_mask].mean():.3f}")
print()

# === Analyze all components: Tweedie variants + cascade/mae ===
components = [
    ('tweedie/tw11', 'results/tweedie/tw11_oof.npy', 'results/tweedie/tw11_test.npy'),
    ('tweedie/tw13', 'results/tweedie/tw13_oof.npy', 'results/tweedie/tw13_test.npy'),
    ('tweedie/tw15', 'results/tweedie/tw15_oof.npy', 'results/tweedie/tw15_test.npy'),
    ('tweedie/tw17', 'results/tweedie/tw17_oof.npy', 'results/tweedie/tw17_test.npy'),
    ('tweedie/tw19', 'results/tweedie/tw19_oof.npy', 'results/tweedie/tw19_test.npy'),
    ('cascade/w30_mae', 'results/cascade/spec_lgb_w30_mae_oof.npy', 'results/cascade/spec_lgb_w30_mae_test.npy'),
    ('cascade/raw_mae', 'results/cascade/spec_lgb_raw_mae_oof.npy', 'results/cascade/spec_lgb_raw_mae_test.npy'),
    ('cascade/raw_huber', 'results/cascade/spec_lgb_raw_huber_oof.npy', 'results/cascade/spec_lgb_raw_huber_test.npy'),
]

print(f"{'Name':<22} {'OOF_MAE':>8} {'corr':>7} {'t_seen':>8} {'t_unseen':>9} {'Δu':>7}")
print("-"*75)

best_blends = []

for name, oof_path, test_path in components:
    try:
        o = np.load(oof_path)
        t = np.load(test_path)
        if o.ndim > 1: o = o.mean(axis=1)
        if t.ndim > 1: t = t.mean(axis=1)

        # Align OOF to y_true order
        if len(o) == len(id2): o = o[id2]
        if len(o) != len(y_true):
            print(f"  {name}: OOF shape mismatch {o.shape}")
            continue
        if len(t) != len(id_order):
            print(f"  {name}: test shape mismatch {t.shape}")
            continue

        o = np.clip(o, 0, None)
        t = np.clip(t, 0, None)

        oof_mae = np.mean(np.abs(y_true - o))
        corr = np.corrcoef(fw4_oo, o)[0,1]
        t_seen = t[~unseen_mask].mean()
        t_unseen = t[unseen_mask].mean()
        delta_u = t_unseen - oracle_new_t[unseen_mask].mean()

        print(f"  {name:<20} {oof_mae:8.4f}  {corr:7.4f}  {t_seen:8.3f}  {t_unseen:9.3f}  {delta_u:+7.3f}")

        # Blend sweep: oracle(1-w) + component(w)
        blend_results = []
        for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
            oof_bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
            test_bl = np.clip((1-w)*oracle_new_t + w*t, 0, None)
            bl_mae = np.mean(np.abs(y_true - oof_bl))
            bl_unseen = test_bl[unseen_mask].mean()
            bl_seen = test_bl[~unseen_mask].mean()
            blend_results.append((w, bl_mae, bl_seen, bl_unseen))

        # Find best weight by OOF
        best = min(blend_results, key=lambda x: x[1])
        best_blends.append({
            'name': name, 'oof_solo': oof_mae, 'corr': corr,
            't_unseen': t_unseen, 'delta_u': delta_u,
            'best_w': best[0], 'best_oof': best[1],
            'best_seen': best[2], 'best_unseen': best[3],
            'oof_gain': oracle_mae - best[1],
            'oof_arr': o, 'test_arr': t
        })

    except Exception as e:
        print(f"  {name}: ERROR {e}")

print()
print("=== Blend Analysis: does OOF improve AND test_unseen stay >= oracle? ===")
print(f"oracle_NEW baseline: OOF={oracle_mae:.4f}  test_unseen={oracle_new_t[unseen_mask].mean():.3f}")
print()
print(f"{'Name':<22} {'best_w':>7} {'blend_OOF':>10} {'OOF_gain':>10} {'blend_unseen':>13} {'Δu_after':>10}")
print("-"*80)

for b in sorted(best_blends, key=lambda x: -x['oof_gain']):
    delta_u_after = b['best_unseen'] - oracle_new_t[unseen_mask].mean()
    marker = ""
    if b['oof_gain'] > 0.001 and delta_u_after >= -0.1:
        marker = " *** CANDIDATE ***"
    elif b['oof_gain'] > 0.001:
        marker = " (OOF+, but loses unseen)"
    print(f"  {b['name']:<20} {b['best_w']:7.2f}  {b['best_oof']:10.4f}  {b['oof_gain']:+10.4f}  {b['best_unseen']:13.3f}  {delta_u_after:+10.3f}{marker}")

print()
print("=== Detailed sweep for top candidates (by OOF gain) ===")
for b in sorted(best_blends, key=lambda x: -x['oof_gain'])[:3]:
    print(f"\n  --- {b['name']} (solo OOF={b['oof_solo']:.4f}, corr={b['corr']:.4f}) ---")
    print(f"  {'w':>5} {'blend_OOF':>10} {'OOF_gain':>10} {'t_seen':>8} {'t_unseen':>9} {'Δu':>7}")
    o = b['oof_arr']; t = b['test_arr']
    for w in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        oof_bl = np.clip((1-w)*fw4_oo + w*o, 0, None)
        test_bl = np.clip((1-w)*oracle_new_t + w*t, 0, None)
        bl_mae = np.mean(np.abs(y_true - oof_bl))
        bl_unseen = test_bl[unseen_mask].mean()
        bl_seen = test_bl[~unseen_mask].mean()
        du = bl_unseen - oracle_new_t[unseen_mask].mean()
        marker = " <-- OOF+" if bl_mae < oracle_mae else ""
        print(f"  {w:5.2f}  {bl_mae:10.4f}  {bl_mae-oracle_mae:+10.4f}  {bl_seen:8.3f}  {bl_unseen:9.3f}  {du:+7.3f}{marker}")

print()
print("=== Saving best candidate if it improves OOF ===")
# Find the single best blend that: 1) improves OOF, 2) maintains unseen >= oracle_unseen - 0.3
oracle_unseen_val = oracle_new_t[unseen_mask].mean()
candidates = [b for b in best_blends if b['oof_gain'] > 0.001 and (b['best_unseen'] >= oracle_unseen_val - 0.3)]

if candidates:
    best_c = max(candidates, key=lambda x: x['oof_gain'])
    print(f"  Best candidate: {best_c['name']}, w={best_c['best_w']:.2f}, OOF={best_c['best_oof']:.4f} ({best_c['oof_gain']:+.4f}), unseen={best_c['best_unseen']:.3f}")

    # Save submission
    o = best_c['oof_arr']; t = best_c['test_arr']
    w = best_c['best_w']
    test_blend = np.clip((1-w)*oracle_new_t + w*t, 0, None)
    sub_tmpl = pd.read_csv('sample_submission.csv')
    sub = sub_tmpl.copy()
    sub['avg_delay_minutes_next_30m'] = test_blend
    fname = f"FINAL_tw_blend_{best_c['name'].replace('/','_')}_w{int(w*100)}_OOF{best_c['best_oof']:.4f}.csv"
    sub.to_csv(fname, index=False)
    print(f"  Saved: {fname}")
    print(f"  test_seen={test_blend[~unseen_mask].mean():.3f}  test_unseen={test_blend[unseen_mask].mean():.3f}")
else:
    print("  No candidate found with OOF improvement + maintaining unseen level.")
    print("  Showing all blend OOFs:")
    for b in sorted(best_blends, key=lambda x: x['best_oof']):
        print(f"    {b['name']}: best_OOF={b['best_oof']:.4f} ({b['oof_gain']:+.4f}), unseen={b['best_unseen']:.3f}")
