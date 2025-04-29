import pandas as pd
from modules.session_labeling import label_session
from modules.metrics_calculator import calculate_movement_metrics

def process_timeframe(file_path, timeframe_name):
    """
    Load a timeframe dataset, label sessions, calculate metrics, return grouped session stats.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [col.capitalize() for col in df.columns]

    # Set datetime index
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df.set_index('Timestamp', inplace=True)
    elif 'Time' in df.columns:
        df['Time'] = pd.to_datetime(df['Time'])
        df.set_index('Time', inplace=True)
    else:
        raise ValueError("No recognizable time column found (expected 'Timestamp' or 'Time').")

    df = df.sort_index()

    # Label sessions
    df['session'] = df.index.map(label_session)

    # Calculate movement metrics
    df = calculate_movement_metrics(df)

    # Group by session and calculate stats
    session_stats = df.groupby('session').agg({
        'range': ['mean', 'max'],
        'ATR_14': ['mean', 'max'],
        'return': ['mean', 'std'],
        'abs_return': ['mean', 'std'],
        'efficiency': 'mean'
    })

    session_stats.columns = ['_'.join(col) for col in session_stats.columns]
    session_stats['timeframe'] = timeframe_name

    return df, session_stats
