# modules/timeframe_processor.py

import pandas as pd
from typing import Tuple

from modules.session_labeling     import label_session
from modules.metrics_calculator    import calculate_movement_metrics

def process_timeframe(
    file_path: str,
    timeframe_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load a timeframe dataset (CSV), index on 'time', label sessions, compute movement metrics,
    and return both the enriched DataFrame and session‐level summary stats.
    """
    # --- Load & index on 'time' ---
    df = pd.read_csv(
        file_path,
        parse_dates=['time'],
        infer_datetime_format=True
    )
    if 'time' not in df.columns:
        raise KeyError(f"No 'time' column found in {file_path}")
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # --- Session labeling ---
    df['session'] = df.index.map(label_session)

    # --- Movement & volatility features ---
    df = calculate_movement_metrics(df)

    # --- Aggregate by session ---
    session_stats = df.groupby('session').agg({
        'range':        ['mean', 'max'],
        'ATR_14':       ['mean', 'max'],
        'price_return': ['mean', 'std'],
        'abs_return':   ['mean', 'std'],
        'efficiency':   'mean'
    })
    # flatten MultiIndex
    session_stats.columns = ['_'.join(col) for col in session_stats.columns]
    session_stats['timeframe'] = timeframe_name
    session_stats = session_stats.reset_index()

    return df, session_stats
