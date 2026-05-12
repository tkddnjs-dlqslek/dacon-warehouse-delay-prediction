"""
피처 엔지니어링 파이프라인
- layout 병합, 타임슬롯 파생, lag/rolling, 상호작용 피처
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def load_data(base_path='.'):
    train = pd.read_csv(f'{base_path}/train.csv')
    test = pd.read_csv(f'{base_path}/test.csv')
    layout = pd.read_csv(f'{base_path}/layout_info.csv')
    return train, test, layout


def add_timeslot(df):
    """시나리오 내 타임슬롯 (0~24) 파생"""
    df['timeslot'] = df.groupby(['layout_id', 'scenario_id']).cumcount()
    return df


def merge_layout(df, layout):
    """layout_info.csv 병합 (+14 정적 피처)"""
    df = df.merge(layout, on='layout_id', how='left')
    return df


def encode_categoricals(train, test):
    """layout_type label encoding"""
    le = LabelEncoder()
    combined = pd.concat([train['layout_type'], test['layout_type']], axis=0)
    le.fit(combined)
    train['layout_type_enc'] = le.transform(train['layout_type'])
    test['layout_type_enc'] = le.transform(test['layout_type'])
    return train, test


# lag/rolling 대상 핵심 동적 컬럼
KEY_DYNAMIC_COLS = [
    'order_inflow_15m', 'congestion_score', 'robot_utilization',
    'battery_mean', 'fault_count_15m', 'blocked_path_15m',
    'pack_utilization', 'charge_queue_length', 'near_collision_15m',
    'avg_trip_distance',
]


def add_lag_rolling_features(df):
    """시나리오 내 lag, diff, rolling 피처 생성"""
    df = df.sort_values(['layout_id', 'scenario_id', 'timeslot']).reset_index(drop=True)
    group = df.groupby(['layout_id', 'scenario_id'])

    for col in KEY_DYNAMIC_COLS:
        if col not in df.columns:
            continue
        g = group[col]
        # Lag
        df[f'{col}_lag1'] = g.shift(1)
        df[f'{col}_lag2'] = g.shift(2)
        # Diff
        df[f'{col}_diff1'] = df[col] - df[f'{col}_lag1']
        # Rolling
        df[f'{col}_rmean3'] = g.transform(lambda x: x.rolling(3, min_periods=1).mean())
        df[f'{col}_rstd3'] = g.transform(lambda x: x.rolling(3, min_periods=1).std())
        df[f'{col}_rmax3'] = g.transform(lambda x: x.rolling(3, min_periods=1).max())
        df[f'{col}_rmean5'] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
        # Cumulative
        df[f'{col}_cummean'] = g.transform(lambda x: x.expanding().mean())

    return df


def add_interaction_features(df):
    """상호작용 및 비율 피처"""
    # Robot 관련
    df['order_per_robot'] = df['order_inflow_15m'] / (df['robot_active'] + 1)
    robot_total_active = df['robot_active'] + df['robot_idle'] + df['robot_charging']
    df['robot_available_ratio'] = df['robot_idle'] / (robot_total_active + 1)
    df['robot_charging_ratio'] = df['robot_charging'] / (robot_total_active + 1)

    # Battery 관련
    df['battery_risk'] = df['low_battery_ratio'] * df['charge_queue_length']

    # Congestion 관련
    df['congestion_x_utilization'] = df['congestion_score'] * df['robot_utilization']
    df['congestion_x_order'] = df['congestion_score'] * df['order_inflow_15m']

    # Order 관련
    df['order_complexity'] = df['unique_sku_15m'] * df['avg_items_per_order']
    df['urgent_order_volume'] = df['order_inflow_15m'] * df['urgent_order_ratio']

    # Dock/Loading 관련
    df['dock_pressure'] = df['loading_dock_util'] * df['outbound_truck_wait_min']

    # Layout 기반 (layout 병합 후)
    if 'floor_area_sqm' in df.columns and 'robot_total' in df.columns:
        df['floor_area_per_robot'] = df['floor_area_sqm'] / (df['robot_total'] + 1)
        df['charger_ratio'] = df['charger_count'] / (df['robot_total'] + 1)
        df['pack_station_density'] = df['pack_station_count'] / (df['floor_area_sqm'] + 1) * 1000

    # 시간 관련
    df['timeslot_norm'] = df['timeslot'] / 24.0
    df['is_early'] = (df['timeslot'] < 5).astype(np.int8)
    df['is_late'] = (df['timeslot'] > 19).astype(np.int8)

    # Network/WMS
    df['network_issues'] = df['wms_response_time_ms'] * df['network_latency_ms']

    # Quality composite
    df['quality_composite'] = (
        df['barcode_read_success_rate'] + df['sort_accuracy_pct'] + df['agv_task_success_rate']
    ) / 3

    return df


def add_missing_flags(df, top_n=10):
    """결측 비율 높은 컬럼에 is_missing 플래그"""
    missing_ratio = df.isnull().mean()
    top_missing = missing_ratio.nlargest(top_n).index.tolist()
    for col in top_missing:
        if col in df.columns:
            df[f'is_missing_{col}'] = df[col].isnull().astype(np.int8)
    return df


def get_feature_cols(df, target='avg_delay_minutes_next_30m'):
    """최종 피처 컬럼 리스트 반환"""
    exclude = ['ID', 'layout_id', 'scenario_id', target, 'layout_type']
    return [c for c in df.columns if c not in exclude]


def build_features(train, test, layout):
    """전체 피처 엔지니어링 파이프라인"""
    # 타임슬롯
    train = add_timeslot(train)
    test = add_timeslot(test)

    # Layout 병합
    train = merge_layout(train, layout)
    test = merge_layout(test, layout)

    # 카테고리 인코딩
    train, test = encode_categoricals(train, test)

    # Lag/Rolling
    train = add_lag_rolling_features(train)
    test = add_lag_rolling_features(test)

    # 상호작용
    train = add_interaction_features(train)
    test = add_interaction_features(test)

    # 결측 플래그
    train = add_missing_flags(train)
    test = add_missing_flags(test)

    return train, test


if __name__ == '__main__':
    train, test, layout = load_data('.')
    train, test = build_features(train, test, layout)
    feature_cols = get_feature_cols(train)
    print(f"피처 수: {len(feature_cols)}")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
