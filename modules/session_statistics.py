# timeframe_processor.py
import pandas as pd
from session_labeling import label_session
from metrics_calculator import calculate_movement_metrics

def process_timeframe(file_path, timeframe_name):
    """
    Load a timeframe dataset, label sessions, calculate metrics.
    """

    # Load
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)

    # Rename columns properly
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    # Label sessions
    df['session'] = df.index.map(label_session)

    # Calculate movement metrics
    df = calculate_movement_metrics(df)

    # Group and summarize by session
    session_stats = df.groupby('session').agg({
        'range': ['mean', 'max', 'std'],
        'ATR_14': ['mean', 'max', 'std'],
        'return': ['mean', 'std'],
        'abs_return': ['mean', 'std'],
        'efficiency': 'mean'
    })

    session_stats.columns = ['_'.join(col) for col in session_stats.columns]
    session_stats['timeframe'] = timeframe_name

    return session_stats

def calculate_movement_metrics(df):
    """Add range, return, ATR, efficiency columns to df."""
    df['range'] = df['High'] - df['Low']
    df['return'] = df['Close'].diff()
    df['abs_return'] = df['return'].abs()
    df['ATR_14'] = df['range'].rolling(window=14).mean()
    df['efficiency'] = df['range'] / df['ATR_14']
    return df


def label_session(ts):
    """Helper to label trading session (UTC time) including overlaps."""
    if time(0, 0) <= ts.time() < time(7, 0):
        return 'Asia'
    elif time(7, 0) <= ts.time() < time(8, 0):
        return 'Asia + London Overlap'
    elif time(8, 0) <= ts.time() < time(13, 0):
        return 'London'
    elif time(13, 0) <= ts.time() < time(15, 0):
        return 'London + NY Overlap'
    elif time(15, 0) <= ts.time() < time(20, 0):
        return 'New York'
    else:
        return 'Other'


def calculate_session_statistics(df, timeframe_name='Unknown'):
    """
    Given a dataframe with OHLCV, calculates session-wise stats.
    """

    # 1. Prepare Data
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    df['session'] = df.index.map(label_session)

    # Calculate basic movement metrics
    df['range'] = df['high'] - df['low']
    df['return'] = df['close'].diff()
    df['abs_return'] = df['return'].abs()
    df['ATR_14'] = df['range'].rolling(window=14).mean()
    df['efficiency'] = df['range'] / df['ATR_14']

    # 2. Group by Session
    session_stats = df.groupby('session').agg({
        'range': ['mean', 'max', 'std'],
        'ATR_14': ['mean', 'max', 'std'],
        'return': ['mean', 'std'],
        'abs_return': ['mean', 'std'],
        'efficiency': 'mean'
    })

    # 3. Optional: rename columns for clarity
    session_stats.columns = ['_'.join(col) for col in session_stats.columns]

    # 4. Attach timeframe info
    session_stats['timeframe'] = timeframe_name

    return session_stats
