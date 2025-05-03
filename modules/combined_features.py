# modules/combined_features.py

import pandas as pd
import numpy as np

def add_session_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['session_step'] = df.groupby('session').cumcount()
    df['time_since_session_open'] = df['session_step'] * 30  # assuming 30m bars
    df['price_vs_session_open'] = df.groupby('session')['open'].transform('first')
    df['price_vs_session_open'] = df['close'] - df['price_vs_session_open']
    df['prev_session'] = df['session'].shift(1)
    df['is_session_transition'] = df['session'] != df['prev_session']
    df['prev_session_return'] = (
        df.groupby('session')['close'].transform('last').shift(1)
        - df.groupby('session')['open'].transform('first').shift(1)
    )
    df['prev_session_efficiency'] = (
        df.groupby('session')['efficiency'].transform('mean').shift(1)
    )
    return df

def add_candle_structure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['abs_return'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['abs_return'] / df['range']
    df['wick_top'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['wick_bottom'] = df[['open', 'close']].min(axis=1) - df['low']
    df['prev_high'] = df['high'].shift(1)
    df['prev_low'] = df['low'].shift(1)
    df['fakeout_flag'] = (df['high'] > df['prev_high']) & (df['close'] < df['prev_high'])
    df['engulfing_flag'] = (
        (df['close'] > df['open'].shift(1)) &
        (df['open'] < df['close'].shift(1))
    )
    df['strong_close'] = (
        ((df['close'] - df['low']) / df['range'] > 0.8) |
        ((df['high'] - df['close']) / df['range'] > 0.8)
    )
    return df

def add_trend_continuation_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['prev_return'] = df['close'].shift(1) - df['open'].shift(1)
    df['price_return'] = df['close'] - df['open']
    df['return_direction_match'] = np.sign(df['price_return']) == np.sign(df['prev_return'])
    df['rolling_return_sum_3'] = df['price_return'].rolling(window=3).sum()
    df['rolling_max_high'] = df['high'].rolling(window=5).max()
    df['rolling_min_low'] = df['low'].rolling(window=5).min()
    df['relative_to_local_extreme'] = df['close'] - df['rolling_max_high'].shift(1)
    return df

def add_structure_zone_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['distance_to_prev_high'] = df['high'] - df['prev_high']
    df['distance_to_prev_low'] = df['low'] - df['prev_low']
    df['touch_prev_session_high'] = (
        (df['high'] >= df['prev_high']) & (df['low'] <= df['prev_high'])
    )
    df['touch_prev_session_low'] = (
        (df['low'] <= df['prev_low']) & (df['high'] >= df['prev_low'])
    )
    df['is_range_boundary_test'] = (
        (df['close'] - df['rolling_max_high']) < (0.01 * df['close'])
    )
    df['equal_highs_lows_flag'] = (
        (df['high'].round(-1) == df['high'].shift(1).round(-1)) &
        (df['low'].round(-1) == df['low'].shift(1).round(-1))
    )
    return df

def add_meta_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features:
      - trend_strength
      - volatility_normalized_return (price_return / ATR_14)
      - body_wick_alignment_score
      - session_bias_score (rolling mean of return)
    """
    df = df.copy()

    # Alias price_return to return if needed for backward compatibility
    if 'price_return' in df.columns and 'return' not in df.columns:
        df['return'] = df['price_return']

    # Composite metrics
    df['trend_strength'] = df['abs_return'] * df['efficiency']
    df['volatility_normalized_return'] = df['price_return'] / df['ATR_14']
    df['body_wick_alignment_score'] = (
        (df['wick_top']    < df['body_ratio']) &
        (df['wick_bottom'] < df['body_ratio'])
    )
    df['session_bias_score'] = (
        df.groupby('session')['return']
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return df
