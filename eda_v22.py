"""
EDA v22: v14 (LB 10.17) 개선을 위한 심층 분석
- Section 0: 데이터 로딩 + v14 OOF 재생산
- Section 1: GAP 진단 (OOF vs LB)
- Section 2: 미사용 컬럼 분석
- Section 3: 극단값 분석
- Section 4: Train vs Test 분포 비교
- Section 5: Layout 타입별 성능
- Section 6: Lag/Rolling 확장 후보 ★
- Section 7: 잔차 분석 + 2-Stage
- Section 8: 앙상블/후처리 기회
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, roc_auc_score
import lightgbm as lgb
from scipy.stats import ks_2samp, spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import warnings
import time
import os

warnings.filterwarnings('ignore')

# 한글 폰트
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

TARGET = 'avg_delay_minutes_next_30m'
N_SPLITS = 5
SEED = 42
OUT_DIR = './results/eda_v22'
os.makedirs(OUT_DIR, exist_ok=True)

def section_timer(name):
    """섹션 시작/종료 타이머"""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n{'='*60}")
            print(f"  {name}")
            print(f"{'='*60}", flush=True)
            return self
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"  [{name}] 완료: {elapsed:.1f}초", flush=True)
    return Timer()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 0: 데이터 로딩 + v14 OOF 재생산
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 0: 데이터 로딩 + v14 OOF"):
    train = pd.read_csv('./train.csv')
    test = pd.read_csv('./test.csv')
    layout = pd.read_csv('./layout_info.csv')
    print(f"  train: {train.shape}, test: {test.shape}, layout: {layout.shape}")

    # y, y_log는 train_fe 생성 후 정렬된 순서에서 추출 (아래에서 재할당)

    # 컬럼 분류
    meta_cols = ['ID', 'layout_id', 'scenario_id', TARGET]
    operational_cols = [c for c in train.columns if c not in meta_cols]
    print(f"  운영 피처: {len(operational_cols)}개")

    key_cols = ['order_inflow_15m', 'congestion_score', 'robot_utilization',
                'battery_mean', 'fault_count_15m', 'blocked_path_15m',
                'pack_utilization', 'charge_queue_length']
    non_key_cols = [c for c in operational_cols if c not in key_cols]
    print(f"  key_cols (lag/rolling 적용): {len(key_cols)}개")
    print(f"  non_key_cols (미적용): {len(non_key_cols)}개")

    # layout 정보
    test_layouts = set(test['layout_id'].unique())
    train_layouts = set(train['layout_id'].unique())
    shared_layouts = test_layouts & train_layouts
    unseen_test_layouts = test_layouts - train_layouts
    print(f"  train layouts: {len(train_layouts)}, test layouts: {len(test_layouts)}")
    print(f"  shared: {len(shared_layouts)}, unseen in test: {len(unseen_test_layouts)}")

    # v14 engineer_features (복사)
    def engineer_features_v14(df, layout_df):
        df = df.merge(layout_df, on='layout_id', how='left')
        df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
        df['timeslot_sq'] = df['timeslot'] ** 2
        df['timeslot_norm'] = df['timeslot'] / 24.0
        df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
        group = df.groupby(['layout_id', 'scenario_id'])
        for col in key_cols:
            if col not in df.columns: continue
            g = group[col]
            df[f'{col}_lag1'] = g.shift(1)
            df[f'{col}_lag2'] = g.shift(2)
            df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
            df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
            df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
            df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
            df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())
        # interactions
        df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
        rta = df['robot_active'] + df['robot_idle'] + df['robot_charging']
        df['robot_available_ratio'] = df['robot_idle'] / (rta + 1)
        df['robot_charging_ratio'] = df['robot_charging'] / (rta + 1)
        df['battery_risk'] = df['low_battery_ratio'] * df['charge_queue_length']
        df['congestion_x_utilization'] = df['congestion_score'] * df['robot_utilization']
        df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']
        df['order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
        df['urgent_order_volume'] = df['order_inflow_15m'] * df['urgent_order_ratio']
        df['dock_pressure'] = df['loading_dock_util'] * df['outbound_truck_wait_min']
        df['staff_per_order'] = df['staff_on_floor'] / (df['order_inflow_15m'] + 1)
        df['total_utilization'] = (df['pack_utilization'] + df['staging_area_util'] + df['loading_dock_util']) / 3
        df['fault_x_congestion'] = df['fault_count_15m'] * df['congestion_score']
        df['battery_charge_pressure'] = df['low_battery_ratio'] * df['avg_charge_wait']
        df['congestion_per_robot'] = df['congestion_score'] / (df['robot_active'] + 1)
        df['order_per_staff'] = df['order_inflow_15m'] / (df['staff_on_floor'] + 1)
        # layout ratios
        df['order_per_area'] = df['order_inflow_15m'] / (df['floor_area_sqm'] + 1) * 1000
        df['congestion_per_area'] = df['congestion_score'] / (df['floor_area_sqm'] + 1) * 1000
        df['fault_per_robot_total'] = df['fault_count_15m'] / (df['robot_total'] + 1)
        df['blocked_per_robot_total'] = df['blocked_path_15m'] / (df['robot_total'] + 1)
        df['collision_per_robot_total'] = df['near_collision_15m'] / (df['robot_total'] + 1)
        df['pack_util_per_station'] = df['pack_utilization'] / (df['pack_station_count'] + 1)
        df['charge_queue_per_charger'] = df['charge_queue_length'] / (df['charger_count'] + 1)
        df['order_per_pack_station'] = df['order_inflow_15m'] / (df['pack_station_count'] + 1)
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['robot_density'] = df['robot_total'] / (df['floor_area_sqm'] + 1) * 1000
        df['active_vs_total'] = df['robot_active'] / (df['robot_total'] + 1)
        df['congestion_x_aisle_width'] = df['congestion_score'] * df['aisle_width_avg']
        df['congestion_x_compactness'] = df['congestion_score'] * df['layout_compactness']
        df['blocked_x_one_way'] = df['blocked_path_15m'] * df['one_way_ratio']
        df['utilization_x_compactness'] = df['robot_utilization'] * df['layout_compactness']
        # remove layout static
        layout_static = ['layout_type', 'aisle_width_avg', 'intersection_count', 'one_way_ratio',
                         'pack_station_count', 'charger_count', 'layout_compactness', 'zone_dispersion',
                         'robot_total', 'building_age_years', 'floor_area_sqm', 'ceiling_height_m',
                         'fire_sprinkler_count', 'emergency_exit_count']
        df = df.drop(columns=[c for c in layout_static if c in df.columns], errors='ignore')
        # remove high corr
        corr_remove = [
            'battery_mean_rmean3', 'charge_queue_length_rmean3',
            'battery_mean_rmean5', 'charge_queue_length_rmean5',
            'pack_utilization_rmean5', 'battery_mean_lag1',
            'charge_queue_length_lag1', 'congestion_score_rmean3',
            'order_inflow_15m_cummean', 'robot_utilization_rmean5',
            'robot_utilization_rmean3', 'order_inflow_15m_rmean5',
            'battery_risk', 'congestion_score_rmean5',
            'pack_utilization_rmean3', 'order_inflow_15m_rmean3',
            'charge_queue_length_lag2', 'blocked_path_15m_rmean5',
        ]
        df = df.drop(columns=[c for c in corr_remove if c in df.columns], errors='ignore')
        return df

    # v14 피처 생성
    test_raw = test.copy()

    train_fe = engineer_features_v14(train.copy(), layout)
    # train_fe는 sort_values + reset_index 되어 있음 → y도 여기서 추출
    y = train_fe[TARGET].values
    y_log = np.log1p(y)

    exclude = ['ID', 'layout_id', 'scenario_id', TARGET]
    feature_cols = [c for c in train_fe.columns if c not in exclude]
    print(f"  v14 피처 수: {len(feature_cols)}")

    X = train_fe[feature_cols]
    X_test_fe = engineer_features_v14(test.copy(), layout)[feature_cols]
    groups = train_fe['layout_id']

    gkf = GroupKFold(n_splits=N_SPLITS)
    folds = list(gkf.split(X, y, groups=groups))

    # train_raw도 동일 순서로 정렬 (train_fe와 1:1 대응)
    train_raw = train.copy()
    train_raw = train_raw.sort_values(['layout_id', 'scenario_id']).reset_index(drop=True)
    train_raw['timeslot'] = train_raw.groupby(['layout_id', 'scenario_id']).cumcount()
    train_raw = train_raw.merge(layout[['layout_id', 'layout_type']], on='layout_id', how='left')

    # v14 OOF 생성 (LGB_MAE + log target, 5-fold)
    print("  v14 OOF 생성 (LGB_MAE, log target, 5-fold)...")
    v14_oof = np.zeros(len(train))
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        model = lgb.LGBMRegressor(
            objective='mae', n_estimators=5000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=SEED, verbose=-1, n_jobs=-1)
        model.fit(X.iloc[tr_idx], y_log[tr_idx],
                  eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        v14_oof[val_idx] = np.expm1(model.predict(X.iloc[val_idx]))
        print(f"    Fold {fold_idx}: MAE={mean_absolute_error(y[val_idx], v14_oof[val_idx]):.4f}")

    v14_oof = np.clip(v14_oof, 0, None)
    v14_residual = y - v14_oof
    oof_mae = mean_absolute_error(y, v14_oof)
    print(f"  ▶ v14 OOF MAE (LGB only): {oof_mae:.4f}")

    # OOF 저장
    pd.DataFrame({'actual': y, 'predicted': v14_oof, 'residual': v14_residual}).to_csv(
        f'{OUT_DIR}/v14_oof_predictions.csv', index=False)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 1: GAP 진단
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 1: GAP 진단 (OOF vs LB)"):
    report = []
    report.append("=== Section 1: GAP 진단 ===\n")

    # 1.1 Fold별 MAE
    report.append("1.1 Fold별 MAE 분해:")
    fold_maes = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        fold_mae = mean_absolute_error(y[val_idx], v14_oof[val_idx])
        fold_layouts = train_fe['layout_id'].iloc[val_idx].nunique()
        fold_n = len(val_idx)
        report.append(f"  Fold {fold_idx}: MAE={fold_mae:.4f}, layouts={fold_layouts}, n={fold_n}")
        fold_maes.append(fold_mae)
        print(f"  Fold {fold_idx}: MAE={fold_mae:.4f}, layouts={fold_layouts}")
    report.append(f"  Fold MAE std: {np.std(fold_maes):.4f}")
    report.append(f"  Fold MAE range: {max(fold_maes)-min(fold_maes):.4f}\n")

    # 1.2 Seen vs Unseen layout
    report.append("1.2 Seen vs Unseen layout OOF MAE:")
    seen_mask = train_fe['layout_id'].isin(shared_layouts).values
    unseen_mask = ~seen_mask
    seen_mae = mean_absolute_error(y[seen_mask], v14_oof[seen_mask])
    unseen_mae = mean_absolute_error(y[unseen_mask], v14_oof[unseen_mask])
    report.append(f"  Seen layouts ({shared_layouts.__len__()}개): MAE={seen_mae:.4f}, n={seen_mask.sum()}")
    report.append(f"  Unseen layouts ({len(train_layouts - shared_layouts)}개): MAE={unseen_mae:.4f}, n={unseen_mask.sum()}")
    report.append(f"  Seen vs Unseen 차이: {unseen_mae - seen_mae:.4f}\n")
    print(f"  Seen MAE: {seen_mae:.4f}, Unseen MAE: {unseen_mae:.4f}, 차이: {unseen_mae-seen_mae:.4f}")

    # 1.3 OOF vs submission 분포 비교
    report.append("1.3 OOF vs Submission 예측 분포:")
    try:
        sub_v14 = pd.read_csv('./submission_v14.csv')
        for pct in [5, 25, 50, 75, 95, 99]:
            oof_pct = np.percentile(v14_oof, pct)
            sub_pct = np.percentile(sub_v14[TARGET], pct)
            report.append(f"  P{pct}: OOF={oof_pct:.2f}, Sub={sub_pct:.2f}, diff={sub_pct-oof_pct:.2f}")
        report.append(f"  OOF mean={v14_oof.mean():.2f}, Sub mean={sub_v14[TARGET].mean():.2f}")

        # 분포 시각화
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        ax[0].hist(v14_oof, bins=100, alpha=0.7, label='OOF', density=True)
        ax[0].hist(sub_v14[TARGET], bins=100, alpha=0.7, label='Submission', density=True)
        ax[0].set_title('OOF vs Submission 전체 분포')
        ax[0].legend()
        ax[0].set_xlim(0, 100)
        ax[1].hist(v14_oof[v14_oof < 50], bins=50, alpha=0.7, label='OOF', density=True)
        ax[1].hist(sub_v14[TARGET][sub_v14[TARGET] < 50], bins=50, alpha=0.7, label='Submission', density=True)
        ax[1].set_title('OOF vs Submission (< 50min)')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(f'{OUT_DIR}/gap_distribution.png', dpi=100)
        plt.close()
    except FileNotFoundError:
        report.append("  submission_v14.csv 없음 — 스킵")

    with open(f'{OUT_DIR}/gap_diagnosis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print('\n'.join(report[-10:]))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 2: 미사용 컬럼 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 2: 미사용 컬럼 분석"):
    # 2.1 Target과 Spearman 상관
    print("  2.1 Target 상관계수...")
    corr_results = []
    for col in operational_cols:
        valid = train_raw[col].notna()
        if valid.sum() < 1000: continue
        rho, _ = spearmanr(train_raw.loc[valid, col], y[valid])
        corr_results.append({
            'column': col,
            'spearman_corr': rho,
            'abs_corr': abs(rho),
            'is_key_col': col in key_cols,
        })
    corr_df = pd.DataFrame(corr_results).sort_values('abs_corr', ascending=False)
    print(f"  상위 20개 (target 상관):")
    for _, row in corr_df.head(20).iterrows():
        flag = "★" if row['is_key_col'] else " "
        print(f"    {flag} {row['column']:35s}  r={row['spearman_corr']:+.4f}")

    # 2.2 Within-scenario 분산
    print("\n  2.2 Within-scenario 분산...")
    within_var = {}
    scenario_groups = train_raw.groupby(['layout_id', 'scenario_id'])
    for col in operational_cols:
        try:
            wv = scenario_groups[col].std().mean()
            within_var[col] = wv if not np.isnan(wv) else 0
        except:
            within_var[col] = 0
    var_df = pd.DataFrame([{'column': k, 'within_scenario_std': v} for k, v in within_var.items()])
    var_df = var_df.sort_values('within_scenario_std', ascending=False)

    # 시간 변동이 큰 상위 20개
    print(f"  시간 변동 상위 20개:")
    for _, row in var_df.head(20).iterrows():
        flag = "★" if row['column'] in key_cols else " "
        print(f"    {flag} {row['column']:35s}  std={row['within_scenario_std']:.4f}")

    # 시간 변동 없는 컬럼 (정적)
    static_ops = var_df[var_df['within_scenario_std'] < 0.001]['column'].tolist()
    print(f"  시간 변동 없는 컬럼 ({len(static_ops)}개): {static_ops[:10]}...")

    # 2.3 잔차와의 상관
    print("\n  2.3 v14 잔차와의 상관...")
    # v14_residual은 train_fe 순서 — train_raw도 동일 순서
    resid_corrs = []
    for col in operational_cols:
        valid = train_raw[col].notna()
        if valid.sum() < 1000: continue
        rho, _ = spearmanr(train_raw.loc[valid, col], v14_residual[valid])
        resid_corrs.append({'column': col, 'resid_corr': rho, 'abs_resid_corr': abs(rho),
                            'is_key_col': col in key_cols})
    resid_df = pd.DataFrame(resid_corrs).sort_values('abs_resid_corr', ascending=False)
    print(f"  잔차 상관 상위 20개:")
    for _, row in resid_df.head(20).iterrows():
        flag = "★" if row['is_key_col'] else " "
        print(f"    {flag} {row['column']:35s}  r={row['resid_corr']:+.4f}")

    # 통합 테이블 저장
    merged = corr_df.merge(var_df, on='column').merge(resid_df[['column', 'resid_corr', 'abs_resid_corr']], on='column')
    merged = merged.sort_values('abs_corr', ascending=False)
    merged.to_csv(f'{OUT_DIR}/unused_column_analysis.csv', index=False)

    # 시각화: 미사용 컬럼 중 target 상관 top 20
    non_key_corr = corr_df[~corr_df['is_key_col']].head(20)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].barh(non_key_corr['column'][::-1], non_key_corr['abs_corr'][::-1])
    axes[0].set_title('미사용 컬럼: Target 상관 Top 20')
    axes[0].set_xlabel('|Spearman r|')

    non_key_resid = resid_df[~resid_df['is_key_col']].head(20)
    axes[1].barh(non_key_resid['column'][::-1], non_key_resid['abs_resid_corr'][::-1])
    axes[1].set_title('미사용 컬럼: 잔차 상관 Top 20')
    axes[1].set_xlabel('|Spearman r|')
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/unused_column_ranking.png', dpi=100)
    plt.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 3: 극단값 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 3: 극단값 분석"):
    report = []
    report.append("=== Section 3: 극단값 분석 ===\n")

    extreme_mask = y > 100
    report.append(f"극단값 (>100min) 비율: {extreme_mask.mean():.4f} ({extreme_mask.sum()}개)")
    report.append(f"극단값 target 통계: mean={y[extreme_mask].mean():.1f}, "
                  f"median={np.median(y[extreme_mask]):.1f}, "
                  f"max={y[extreme_mask].max():.1f}\n")

    # 3.1 timeslot/layout_type 분포
    report.append("3.1 극단값 프로파일:")
    report.append("  Timeslot 분포:")
    for ts in [0, 5, 10, 15, 20, 24]:
        ts_mask = train_raw['timeslot'] == ts
        ts_extreme = extreme_mask & ts_mask
        pct = ts_extreme.sum() / ts_mask.sum() * 100 if ts_mask.sum() > 0 else 0
        report.append(f"    ts={ts}: 극단값 비율={pct:.2f}%")

    report.append("  Layout type 분포:")
    for lt in train_raw['layout_type'].unique():
        lt_mask = train_raw['layout_type'] == lt
        lt_extreme = extreme_mask & lt_mask.values
        pct = lt_extreme.sum() / lt_mask.sum() * 100 if lt_mask.sum() > 0 else 0
        report.append(f"    {lt}: 극단값 비율={pct:.2f}%")

    # 3.2 극단값 이진 분류기
    report.append("\n3.2 극단값 이진 분류기:")
    X_cls = X.copy()
    y_cls = (y > 100).astype(int)
    cls_oof = np.zeros(len(train))
    cls_aucs = []
    for fold_idx, (tr_idx, val_idx) in enumerate(folds):
        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=6, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            random_state=SEED, verbose=-1, n_jobs=-1)
        clf.fit(X_cls.iloc[tr_idx], y_cls[tr_idx],
                eval_set=[(X_cls.iloc[val_idx], y_cls[val_idx])],
                callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        cls_oof[val_idx] = clf.predict_proba(X_cls.iloc[val_idx])[:, 1]
        fold_auc = roc_auc_score(y_cls[val_idx], cls_oof[val_idx])
        cls_aucs.append(fold_auc)
    overall_auc = roc_auc_score(y_cls, cls_oof)
    report.append(f"  OOF AUC: {overall_auc:.4f}")
    report.append(f"  Fold AUCs: {[f'{a:.4f}' for a in cls_aucs]}")
    report.append(f"  → {'2-stage 가능성 높음' if overall_auc > 0.85 else '2-stage 효과 제한적'}\n")
    print(f"  극단값 분류 AUC: {overall_auc:.4f}")

    # 3.3 극단값 MAE 기여도
    report.append("3.3 극단값 MAE 기여도:")
    # v14_oof와 y 순서: 둘 다 train_fe 순서 (= train_raw 순서)
    normal_mae = mean_absolute_error(y[~extreme_mask], v14_oof[~extreme_mask])
    extreme_mae_val = mean_absolute_error(y[extreme_mask], v14_oof[extreme_mask])
    total_mae_check = mean_absolute_error(y, v14_oof)
    extreme_contrib = extreme_mask.sum() * extreme_mae_val / (len(y) * total_mae_check) * 100
    report.append(f"  정상값 MAE: {normal_mae:.4f} (n={(~extreme_mask).sum()})")
    report.append(f"  극단값 MAE: {extreme_mae_val:.4f} (n={extreme_mask.sum()})")
    report.append(f"  전체 MAE: {total_mae_check:.4f}")
    report.append(f"  극단값 MAE 기여도: {extreme_contrib:.1f}%\n")
    print(f"  정상 MAE: {normal_mae:.4f}, 극단 MAE: {extreme_mae_val:.4f}, 기여도: {extreme_contrib:.1f}%")

    # 3.4 Clipping 민감도
    report.append("3.4 Clipping 민감도:")
    best_clip_mae = total_mae_check
    best_clip_val = None
    for cap in [50, 75, 100, 150, 200, 300, 500, None]:
        clipped = np.clip(v14_oof, 0, cap)
        clip_mae = mean_absolute_error(y, clipped)
        report.append(f"  Cap={cap}: MAE={clip_mae:.4f}")
        if clip_mae < best_clip_mae:
            best_clip_mae = clip_mae
            best_clip_val = cap
    report.append(f"  → 최적 Cap: {best_clip_val} (MAE {best_clip_mae:.4f})")
    print(f"  최적 Cap: {best_clip_val} (MAE {best_clip_mae:.4f})")

    # 시각화
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(y[extreme_mask], bins=50, color='red', alpha=0.7)
    axes[0].set_title(f'극단값 (>100min) 분포 (n={extreme_mask.sum()})')
    axes[0].set_xlabel('target')

    axes[1].scatter(v14_oof[extreme_mask], y[extreme_mask], alpha=0.3, s=5, c='red')
    axes[1].plot([0, 700], [0, 700], 'k--', alpha=0.5)
    axes[1].set_title('극단값: 예측 vs 실제')
    axes[1].set_xlabel('predicted'); axes[1].set_ylabel('actual')

    # clipping curve
    caps = list(range(50, 501, 10))
    clip_maes = [mean_absolute_error(y, np.clip(v14_oof, 0, c)) for c in caps]
    axes[2].plot(caps, clip_maes)
    axes[2].axhline(total_mae_check, color='r', linestyle='--', label=f'No clip: {total_mae_check:.4f}')
    axes[2].set_title('Clipping 민감도')
    axes[2].set_xlabel('Cap'); axes[2].set_ylabel('MAE'); axes[2].legend()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/extreme_profile.png', dpi=100)
    plt.close()

    with open(f'{OUT_DIR}/extreme_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 4: Train vs Test 분포 비교
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 4: Train vs Test 분포 비교"):
    report = []
    report.append("=== Section 4: Train vs Test 분포 비교 ===\n")

    # 4.1 KS Test 전체
    report.append("4.1 KS Test (전체 90개 운영 컬럼):")
    ks_results = []
    for col in operational_cols:
        tr_vals = train_raw[col].dropna()
        te_vals = test_raw[col].dropna()
        if len(tr_vals) < 100 or len(te_vals) < 100: continue
        stat, pval = ks_2samp(tr_vals, te_vals)
        ks_results.append({'column': col, 'ks_stat': stat, 'p_value': pval,
                           'is_key_col': col in key_cols})
    ks_df = pd.DataFrame(ks_results).sort_values('ks_stat', ascending=False)
    report.append(f"  KS stat > 0.1인 컬럼 ({(ks_df['ks_stat'] > 0.1).sum()}개):")
    for _, row in ks_df[ks_df['ks_stat'] > 0.1].iterrows():
        flag = "★" if row['is_key_col'] else " "
        report.append(f"    {flag} {row['column']:35s}  KS={row['ks_stat']:.4f}")
    print(f"  KS > 0.1: {(ks_df['ks_stat'] > 0.1).sum()}개 컬럼")
    for _, row in ks_df.head(10).iterrows():
        print(f"    {row['column']:35s} KS={row['ks_stat']:.4f}")

    # 4.2 sku_concentration 심층
    report.append("\n4.2 sku_concentration 심층 분석:")
    sku_ks_per_layout = []
    for lid in shared_layouts:
        tr_vals = train_raw.loc[train_raw['layout_id'] == lid, 'sku_concentration'].dropna()
        te_vals = test_raw.loc[test_raw['layout_id'] == lid, 'sku_concentration'].dropna()
        if len(tr_vals) > 10 and len(te_vals) > 10:
            stat, pval = ks_2samp(tr_vals, te_vals)
            sku_ks_per_layout.append({'layout_id': lid, 'ks_stat': stat, 'p_value': pval,
                                       'tr_mean': tr_vals.mean(), 'te_mean': te_vals.mean()})
    sku_ks_df = pd.DataFrame(sku_ks_per_layout).sort_values('ks_stat', ascending=False)
    n_significant = (sku_ks_df['p_value'] < 0.05).sum()
    report.append(f"  공유 {len(sku_ks_df)}개 layout 중 유의미한 분포 차이: {n_significant}개")
    report.append(f"  평균 KS stat: {sku_ks_df['ks_stat'].mean():.4f}")
    if len(sku_ks_df) > 0:
        report.append(f"  최대 KS layout: {sku_ks_df.iloc[0]['layout_id']} (KS={sku_ks_df.iloc[0]['ks_stat']:.4f})")
    print(f"  sku_concentration: 공유 layout 중 유의미 차이 {n_significant}/{len(sku_ks_df)}개")

    # 4.3 공유 layout 운영 피처 비교
    report.append("\n4.3 공유 layout 운영 피처 분포 비교:")
    shared_train = train_raw[train_raw['layout_id'].isin(shared_layouts)]
    shared_test = test_raw[test_raw['layout_id'].isin(shared_layouts)]
    shared_ks = []
    for col in operational_cols:
        tr_vals = shared_train[col].dropna()
        te_vals = shared_test[col].dropna()
        if len(tr_vals) < 100 or len(te_vals) < 100: continue
        stat, pval = ks_2samp(tr_vals, te_vals)
        shared_ks.append({'column': col, 'ks_stat': stat, 'p_value': pval})
    shared_ks_df = pd.DataFrame(shared_ks).sort_values('ks_stat', ascending=False)
    report.append(f"  공유 layout에서 KS > 0.1: {(shared_ks_df['ks_stat'] > 0.1).sum()}개")
    for _, row in shared_ks_df.head(10).iterrows():
        report.append(f"    {row['column']:35s}  KS={row['ks_stat']:.4f}")
    print(f"  공유 layout KS > 0.1: {(shared_ks_df['ks_stat'] > 0.1).sum()}개")

    # 4.4 day_of_week, shift_hour
    report.append("\n4.4 시간 분포:")
    for col in ['day_of_week', 'shift_hour']:
        if col in train_raw.columns and col in test_raw.columns:
            tr_dist = train_raw[col].value_counts(normalize=True).sort_index()
            te_dist = test_raw[col].value_counts(normalize=True).sort_index()
            report.append(f"  {col}:")
            for val in sorted(set(tr_dist.index) | set(te_dist.index)):
                tr_pct = tr_dist.get(val, 0) * 100
                te_pct = te_dist.get(val, 0) * 100
                report.append(f"    {val}: train={tr_pct:.1f}%, test={te_pct:.1f}%")

    ks_df.to_csv(f'{OUT_DIR}/distribution_comparison.csv', index=False)

    # 시각화
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    top10 = ks_df.head(10)
    axes[0].barh(top10['column'][::-1], top10['ks_stat'][::-1])
    axes[0].set_title('전체: KS Statistic Top 10')
    axes[0].axvline(0.1, color='r', linestyle='--', alpha=0.5)

    top10_shared = shared_ks_df.head(10)
    axes[1].barh(top10_shared['column'][::-1], top10_shared['ks_stat'][::-1])
    axes[1].set_title('공유 Layout: KS Statistic Top 10')
    axes[1].axvline(0.1, color='r', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/ks_test_results.png', dpi=100)
    plt.close()

    with open(f'{OUT_DIR}/distribution_comparison.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 5: Layout 타입별 성능
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 5: Layout 타입별 성능"):
    report = []
    report.append("=== Section 5: Layout 타입별 분석 ===\n")

    # 5.1 분포
    report.append("5.1 Layout type 분포:")
    # train
    tr_lt = train_raw.merge(layout[['layout_id', 'layout_type']], on='layout_id', how='left', suffixes=('', '_dup'))
    if 'layout_type_dup' in tr_lt.columns:
        tr_lt = tr_lt.drop(columns='layout_type_dup')
    tr_lt_dist = tr_lt['layout_type'].value_counts(normalize=True)
    # test seen/unseen
    te_with_lt = test_raw.merge(layout[['layout_id', 'layout_type']], on='layout_id', how='left')
    te_seen = te_with_lt[te_with_lt['layout_id'].isin(shared_layouts)]
    te_unseen = te_with_lt[~te_with_lt['layout_id'].isin(shared_layouts)]
    te_seen_dist = te_seen['layout_type'].value_counts(normalize=True) if len(te_seen) > 0 else pd.Series()
    te_unseen_dist = te_unseen['layout_type'].value_counts(normalize=True) if len(te_unseen) > 0 else pd.Series()

    for lt in ['narrow', 'grid', 'hybrid', 'hub_spoke']:
        tr_pct = tr_lt_dist.get(lt, 0) * 100
        ts_pct = te_seen_dist.get(lt, 0) * 100
        tu_pct = te_unseen_dist.get(lt, 0) * 100
        report.append(f"  {lt:12s}: train={tr_pct:.1f}%, test-seen={ts_pct:.1f}%, test-unseen={tu_pct:.1f}%")
        print(f"  {lt:12s}: train={tr_pct:.1f}%, test-seen={ts_pct:.1f}%, test-unseen={tu_pct:.1f}%")

    # 5.2 Layout type별 OOF MAE
    report.append("\n5.2 Layout type별 OOF MAE:")
    lt_col = train_raw['layout_type'].values
    for lt in ['narrow', 'grid', 'hybrid', 'hub_spoke']:
        mask = lt_col == lt
        if mask.sum() == 0: continue
        lt_mae = mean_absolute_error(y[mask], v14_oof[mask])
        report.append(f"  {lt:12s}: MAE={lt_mae:.4f}, n={mask.sum()}")
        print(f"  {lt:12s}: MAE={lt_mae:.4f}")

    # 5.3 Target 통계
    report.append("\n5.3 Target 통계 by layout_type:")
    for lt in ['narrow', 'grid', 'hybrid', 'hub_spoke']:
        mask = lt_col == lt
        if mask.sum() == 0: continue
        vals = y[mask]
        extreme_pct = (vals > 100).mean() * 100
        report.append(f"  {lt:12s}: mean={vals.mean():.2f}, median={np.median(vals):.2f}, "
                      f"std={vals.std():.2f}, extreme%={extreme_pct:.2f}%")

    with open(f'{OUT_DIR}/layout_type_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 6: Lag/Rolling 확장 후보 ★ 최우선
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 6: Lag/Rolling 확장 후보 ★"):
    report = []
    report.append("=== Section 6: Lag/Rolling 확장 후보 ===\n")

    # 6.1 Lag 추가 정보량 계산
    print("  6.1 Lag 추가 정보량 계산...")
    # train_raw은 이미 sorted by layout_id, scenario_id
    scenario_group = train_raw.groupby(['layout_id', 'scenario_id'])

    lag_info = []
    dynamic_cols = [c for c in non_key_cols if within_var.get(c, 0) > 0.001]
    print(f"  동적 컬럼 수: {len(dynamic_cols)}개 (정적 제외)")

    for col in dynamic_cols:
        try:
            raw_vals = train_raw[col].values
            lag1_vals = scenario_group[col].shift(1).values
            diff1_vals = raw_vals - lag1_vals

            # valid mask (not NaN)
            valid_raw = ~np.isnan(raw_vals) & ~np.isnan(y)
            valid_lag = valid_raw & ~np.isnan(lag1_vals)
            valid_diff = valid_raw & ~np.isnan(diff1_vals)

            raw_corr = abs(spearmanr(raw_vals[valid_raw], y[valid_raw])[0]) if valid_raw.sum() > 100 else 0
            lag_corr = abs(spearmanr(lag1_vals[valid_lag], y[valid_lag])[0]) if valid_lag.sum() > 100 else 0
            diff_corr = abs(spearmanr(diff1_vals[valid_diff], y[valid_diff])[0]) if valid_diff.sum() > 100 else 0

            # 잔차 상관도
            resid_raw_corr = abs(spearmanr(raw_vals[valid_raw], v14_residual[valid_raw])[0]) if valid_raw.sum() > 100 else 0
            resid_lag_corr = abs(spearmanr(lag1_vals[valid_lag], v14_residual[valid_lag])[0]) if valid_lag.sum() > 100 else 0
            resid_diff_corr = abs(spearmanr(diff1_vals[valid_diff], v14_residual[valid_diff])[0]) if valid_diff.sum() > 100 else 0

            lag_info.append({
                'column': col,
                'raw_target_corr': raw_corr,
                'lag1_target_corr': lag_corr,
                'diff1_target_corr': diff_corr,
                'lag_gain': lag_corr - raw_corr,  # lag가 raw 대비 추가 정보
                'diff_gain': diff_corr - raw_corr,
                'resid_raw_corr': resid_raw_corr,
                'resid_lag_corr': resid_lag_corr,
                'resid_diff_corr': resid_diff_corr,
                'max_new_info': max(lag_corr, diff_corr, resid_lag_corr, resid_diff_corr),
                'within_std': within_var.get(col, 0),
            })
        except Exception as e:
            continue

    lag_df = pd.DataFrame(lag_info).sort_values('max_new_info', ascending=False)
    lag_df.to_csv(f'{OUT_DIR}/lag_expansion_candidates.csv', index=False)

    # 6.2 상위 후보
    report.append("6.1-6.2 Lag 확장 후보 순위 (max_new_info 기준):")
    print(f"\n  Lag 확장 후보 Top 15:")
    for i, (_, row) in enumerate(lag_df.head(15).iterrows()):
        line = (f"    {i+1:2d}. {row['column']:30s}  "
                f"raw={row['raw_target_corr']:.3f} lag1={row['lag1_target_corr']:.3f} "
                f"diff1={row['diff1_target_corr']:.3f} "
                f"resid_lag={row['resid_lag_corr']:.3f} resid_diff={row['resid_diff_corr']:.3f}")
        report.append(line)
        print(line)

    # 6.3 상위 5개 후보 1-fold 빠른 검증
    print("\n  6.3 상위 5개 후보 1-fold 검증...")
    top5_candidates = lag_df.head(5)['column'].tolist()
    report.append(f"\n6.3 1-fold 검증 후보: {top5_candidates}")

    # v14 baseline (fold 0만)
    tr_idx, val_idx = folds[0]
    baseline_model = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    baseline_model.fit(X.iloc[tr_idx], y_log[tr_idx],
                       eval_set=[(X.iloc[val_idx], y_log[val_idx])],
                       callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    baseline_pred = np.expm1(baseline_model.predict(X.iloc[val_idx]))
    baseline_pred = np.clip(baseline_pred, 0, None)
    baseline_fold_mae = mean_absolute_error(y[val_idx], baseline_pred)
    report.append(f"  v14 baseline Fold0 MAE: {baseline_fold_mae:.4f}")
    print(f"  v14 baseline Fold0 MAE: {baseline_fold_mae:.4f}")

    # lag 후보 추가 피처 생성
    train_aug = train_fe.copy()
    for col in top5_candidates:
        if col in train_raw.columns:
            lag1 = scenario_group[col].shift(1).values
            diff1 = train_raw[col].values - lag1
            train_aug[f'{col}_lag1'] = lag1
            train_aug[f'{col}_diff1'] = diff1

    aug_feature_cols = [c for c in train_aug.columns if c not in exclude]
    X_aug = train_aug[aug_feature_cols]

    aug_model = lgb.LGBMRegressor(
        objective='mae', n_estimators=3000, learning_rate=0.03,
        num_leaves=63, max_depth=8, min_child_samples=50,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
        random_state=SEED, verbose=-1, n_jobs=-1)
    aug_model.fit(X_aug.iloc[tr_idx], y_log[tr_idx],
                  eval_set=[(X_aug.iloc[val_idx], y_log[val_idx])],
                  callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
    aug_pred = np.expm1(aug_model.predict(X_aug.iloc[val_idx]))
    aug_pred = np.clip(aug_pred, 0, None)
    aug_fold_mae = mean_absolute_error(y[val_idx], aug_pred)
    report.append(f"  +Lag5후보 Fold0 MAE: {aug_fold_mae:.4f}")
    report.append(f"  → 차이: {baseline_fold_mae - aug_fold_mae:.4f} {'개선' if aug_fold_mae < baseline_fold_mae else '악화'}")
    print(f"  +Lag5후보 Fold0 MAE: {aug_fold_mae:.4f} (차이: {baseline_fold_mae - aug_fold_mae:+.4f})")

    # 개별 후보 검증
    report.append(f"\n  개별 후보 검증:")
    for col in top5_candidates:
        if col not in train_raw.columns: continue
        train_single = train_fe.copy()
        lag1 = scenario_group[col].shift(1).values
        diff1 = train_raw[col].values - lag1
        train_single[f'{col}_lag1'] = lag1
        train_single[f'{col}_diff1'] = diff1
        single_cols = [c for c in train_single.columns if c not in exclude]
        X_single = train_single[single_cols]

        m = lgb.LGBMRegressor(
            objective='mae', n_estimators=3000, learning_rate=0.03,
            num_leaves=63, max_depth=8, min_child_samples=50,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=1.0, reg_lambda=1.0,
            random_state=SEED, verbose=-1, n_jobs=-1)
        m.fit(X_single.iloc[tr_idx], y_log[tr_idx],
              eval_set=[(X_single.iloc[val_idx], y_log[val_idx])],
              callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)])
        p = np.clip(np.expm1(m.predict(X_single.iloc[val_idx])), 0, None)
        single_mae = mean_absolute_error(y[val_idx], p)
        diff_val = baseline_fold_mae - single_mae
        report.append(f"    +{col}: MAE={single_mae:.4f} (차이: {diff_val:+.4f})")
        print(f"    +{col}: MAE={single_mae:.4f} ({diff_val:+.4f})")

    # 6.4 기존 8개 autocorrelation
    report.append("\n6.4 기존 key_cols autocorrelation (lag1~5):")
    for col in key_cols:
        auto_corrs = []
        for lag in [1, 2, 3, 4, 5]:
            shifted = scenario_group[col].shift(lag).values
            valid = ~np.isnan(shifted) & ~np.isnan(train_raw[col].values)
            if valid.sum() > 100:
                ac = np.corrcoef(train_raw[col].values[valid], shifted[valid])[0, 1]
                auto_corrs.append(f"{ac:.3f}")
            else:
                auto_corrs.append("N/A")
        report.append(f"  {col:30s}: lag1={auto_corrs[0]} lag2={auto_corrs[1]} "
                      f"lag3={auto_corrs[2]} lag4={auto_corrs[3]} lag5={auto_corrs[4]}")
    print("  (autocorrelation 결과 → txt 파일 참조)")

    # 시각화
    fig, ax = plt.subplots(figsize=(12, 8))
    top15 = lag_df.head(15)
    x = np.arange(len(top15))
    w = 0.2
    ax.barh(x - w, top15['raw_target_corr'], w, label='raw→target', alpha=0.8)
    ax.barh(x, top15['lag1_target_corr'], w, label='lag1→target', alpha=0.8)
    ax.barh(x + w, top15['diff1_target_corr'], w, label='diff1→target', alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(top15['column'])
    ax.set_title('Lag 확장 후보: 상관계수 비교')
    ax.legend()
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/lag_candidate_ranking.png', dpi=100)
    plt.close()

    with open(f'{OUT_DIR}/lag_expansion_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 7: 잔차 분석
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 7: 잔차 분석"):
    report = []
    report.append("=== Section 7: 잔차 분석 ===\n")

    residual = v14_residual  # y - predicted

    # 7.1 잔차 편향
    report.append("7.1 잔차 편향 분석:")

    # (a) timeslot별
    report.append("  (a) Timeslot별 잔차:")
    ts_vals = train_raw['timeslot'].values
    ts_resid = {}
    for ts in range(25):
        mask = ts_vals == ts
        ts_resid[ts] = residual[mask].mean()
        if ts % 5 == 0:
            report.append(f"    ts={ts:2d}: mean_resid={residual[mask].mean():+.3f}, "
                          f"std={residual[mask].std():.3f}, MAE={mean_absolute_error(y[mask], v14_oof[mask]):.4f}")

    # (b) predicted 구간별
    report.append("  (b) 예측값 구간별 잔차:")
    pred_bins = [(0, 5), (5, 10), (10, 20), (20, 50), (50, 100), (100, np.inf)]
    for lo, hi in pred_bins:
        mask = (v14_oof >= lo) & (v14_oof < hi)
        if mask.sum() == 0: continue
        report.append(f"    pred [{lo:5.0f}-{hi:5.0f}): mean_resid={residual[mask].mean():+.3f}, "
                      f"MAE={mean_absolute_error(y[mask], v14_oof[mask]):.4f}, n={mask.sum()}")

    # (c) layout_type별
    report.append("  (c) Layout type별 잔차:")
    lt_vals = train_raw['layout_type'].values
    for lt in ['narrow', 'grid', 'hybrid', 'hub_spoke']:
        mask = lt_vals == lt
        if mask.sum() == 0: continue
        report.append(f"    {lt:12s}: mean_resid={residual[mask].mean():+.3f}, "
                      f"std={residual[mask].std():.3f}")

    # (d) congestion_score 구간별
    report.append("  (d) Congestion 구간별 잔차:")
    cong = train_raw['congestion_score'].values
    for lo, hi in [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, np.inf)]:
        mask = (cong >= lo) & (cong < hi)
        if mask.sum() < 100: continue
        report.append(f"    cong [{lo:.1f}-{hi:.1f}): mean_resid={residual[mask].mean():+.3f}, "
                      f"MAE={mean_absolute_error(y[mask], v14_oof[mask]):.4f}, n={mask.sum()}")

    # 7.2 구간별 MAE
    report.append("\n7.2 Target 구간별 MAE:")
    bins = [(0, 10, 'Low'), (10, 50, 'Mid'), (50, 100, 'High'), (100, np.inf, 'Extreme')]
    for lo, hi, name in bins:
        mask = (y >= lo) & (y < hi)
        if mask.sum() == 0: continue
        seg_mae = mean_absolute_error(y[mask], v14_oof[mask])
        contrib = mask.sum() * seg_mae / (len(y) * mean_absolute_error(y, v14_oof)) * 100
        report.append(f"  {name:8s} [{lo:5.0f}-{hi:5.0f}): MAE={seg_mae:.4f}, n={mask.sum()} "
                      f"({mask.mean()*100:.1f}%), 기여도={contrib:.1f}%")
        print(f"  {name:8s}: MAE={seg_mae:.4f}, n={mask.sum()}, 기여도={contrib:.1f}%")

    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) timeslot별 잔차
    ts_means = [residual[ts_vals == ts].mean() for ts in range(25)]
    axes[0, 0].bar(range(25), ts_means)
    axes[0, 0].set_title('Timeslot별 평균 잔차')
    axes[0, 0].set_xlabel('timeslot'); axes[0, 0].set_ylabel('mean residual')
    axes[0, 0].axhline(0, color='r', linestyle='--')

    # (b) predicted vs residual
    sample_idx = np.random.RandomState(42).choice(len(v14_oof), min(10000, len(v14_oof)), replace=False)
    axes[0, 1].scatter(v14_oof[sample_idx], residual[sample_idx], alpha=0.1, s=3)
    axes[0, 1].set_title('예측값 vs 잔차')
    axes[0, 1].set_xlabel('predicted'); axes[0, 1].set_ylabel('residual')
    axes[0, 1].axhline(0, color='r', linestyle='--')
    axes[0, 1].set_xlim(0, 100)

    # (c) 잔차 분포
    axes[1, 0].hist(residual, bins=100, range=(-50, 50))
    axes[1, 0].set_title(f'잔차 분포 (mean={residual.mean():.3f}, std={residual.std():.3f})')
    axes[1, 0].axvline(0, color='r', linestyle='--')

    # (d) layout_type별 MAE
    lt_maes = {}
    for lt in ['narrow', 'grid', 'hybrid', 'hub_spoke']:
        mask = lt_vals == lt
        if mask.sum() > 0:
            lt_maes[lt] = mean_absolute_error(y[mask], v14_oof[mask])
    axes[1, 1].bar(lt_maes.keys(), lt_maes.values())
    axes[1, 1].set_title('Layout Type별 MAE')
    axes[1, 1].set_ylabel('MAE')

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/residual_plots.png', dpi=100)
    plt.close()

    with open(f'{OUT_DIR}/residual_analysis.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Section 8: 후처리 기회
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with section_timer("Section 8: 후처리 기회"):
    report = []
    report.append("=== Section 8: 후처리 기회 ===\n")

    # 8.1 최적 clipping
    report.append("8.1 최적 Clipping (세밀 탐색):")
    best_cap = None
    best_clip_mae = mean_absolute_error(y, v14_oof)
    for cap in range(50, 501, 5):
        clipped = np.clip(v14_oof, 0, cap)
        mae_val = mean_absolute_error(y, clipped)
        if mae_val < best_clip_mae:
            best_clip_mae = mae_val
            best_cap = cap
    report.append(f"  No clip MAE: {mean_absolute_error(y, v14_oof):.4f}")
    report.append(f"  Best clip: Cap={best_cap}, MAE={best_clip_mae:.4f}")
    report.append(f"  → 개선: {mean_absolute_error(y, v14_oof) - best_clip_mae:.4f}")
    print(f"  Best clip: Cap={best_cap}, MAE={best_clip_mae:.4f}")

    # 8.2 공유 layout 보정
    report.append("\n8.2 공유 Layout별 보정:")
    layout_ids = train_raw['layout_id'].values
    corrected_oof = v14_oof.copy()
    layout_bias = {}
    for lid in shared_layouts:
        mask = layout_ids == lid
        if mask.sum() < 10: continue
        bias = y[mask].mean() - v14_oof[mask].mean()
        layout_bias[lid] = bias
        corrected_oof[mask] = v14_oof[mask] + bias

    corrected_mae = mean_absolute_error(y, corrected_oof)
    report.append(f"  보정 전 MAE (shared layouts): {mean_absolute_error(y[layout_ids == list(shared_layouts)[0]], v14_oof[layout_ids == list(shared_layouts)[0]]):.4f}")
    report.append(f"  보정 후 전체 MAE: {corrected_mae:.4f}")
    report.append(f"  → 개선: {mean_absolute_error(y, v14_oof) - corrected_mae:.4f}")
    bias_vals = list(layout_bias.values())
    report.append(f"  Layout bias 통계: mean={np.mean(bias_vals):.3f}, std={np.std(bias_vals):.3f}, "
                  f"min={np.min(bias_vals):.3f}, max={np.max(bias_vals):.3f}")
    print(f"  Layout 보정 후 전체 MAE: {corrected_mae:.4f} (개선: {mean_absolute_error(y, v14_oof) - corrected_mae:.4f})")

    # 8.3 Timeslot별 보정
    report.append("\n8.3 Timeslot별 보정:")
    ts_vals = train_raw['timeslot'].values
    ts_corrected = v14_oof.copy()
    ts_bias = {}
    for ts in range(25):
        mask = ts_vals == ts
        bias = y[mask].mean() - v14_oof[mask].mean()
        ts_bias[ts] = bias
        ts_corrected[mask] = v14_oof[mask] + bias

    ts_corrected_mae = mean_absolute_error(y, ts_corrected)
    report.append(f"  보정 전 MAE: {mean_absolute_error(y, v14_oof):.4f}")
    report.append(f"  Timeslot 보정 후 MAE: {ts_corrected_mae:.4f}")
    report.append(f"  → 개선: {mean_absolute_error(y, v14_oof) - ts_corrected_mae:.4f}")
    print(f"  Timeslot 보정 후 MAE: {ts_corrected_mae:.4f}")

    # 8.4 복합 보정 (layout + timeslot)
    report.append("\n8.4 복합 보정 (Layout + Timeslot):")
    combo_corrected = v14_oof.copy()
    for lid in shared_layouts:
        mask = layout_ids == lid
        combo_corrected[mask] += layout_bias.get(lid, 0)
    for ts in range(25):
        mask = ts_vals == ts
        combo_corrected[mask] += ts_bias.get(ts, 0)
    combo_mae = mean_absolute_error(y, combo_corrected)
    report.append(f"  복합 보정 후 MAE: {combo_mae:.4f}")
    print(f"  복합 보정 후 MAE: {combo_mae:.4f}")

    with open(f'{OUT_DIR}/postprocessing_opportunities.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 최종 요약
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print(f"\n{'='*60}")
print(f"  EDA v22 완료!")
print(f"{'='*60}")
print(f"  결과 저장 위치: {OUT_DIR}/")
print(f"  생성된 파일:")
for f in sorted(os.listdir(OUT_DIR)):
    fpath = os.path.join(OUT_DIR, f)
    size = os.path.getsize(fpath)
    print(f"    {f} ({size:,} bytes)")
print(f"{'='*60}", flush=True)
