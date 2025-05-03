# modules/session_statistics.py

import pandas as pd
from typing import Tuple, Optional

from modules.session_labeler    import add_session_labels
from modules.metrics_calculator import calculate_movement_metrics

def process_timeframe(
    file_path: str,
    timeframe_name: str,
    time_col: str = 'time'
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    1) Loads a CSV of OHLCV bars (expects a 'time' column).
    2) Parses & indexes on time_col.
    3) Labels sessions (including overlaps) via add_session_labels.
    4) Calculates movement & volatility metrics via calculate_movement_metrics.
    5) Returns:
         - df_enriched: full bar‐level DataFrame
         - session_stats: summary stats per session
    """
    # --- Load & index ---
    df = pd.read_csv(
        file_path,
        parse_dates=[time_col],
        infer_datetime_format=True
    )
    if time_col not in df.columns:
        raise KeyError(f"Expected a '{time_col}' column in {file_path}")
    df.set_index(time_col, inplace=True)
    df.sort_index(inplace=True)

    # --- Normalize column names to lowercase ---
    df.rename(columns=str.lower, inplace=True)

    # --- Session labeling & metrics enrichment ---
    df = add_session_labels(df, timestamp_col=None)       # adds df['session']
    df = calculate_movement_metrics(df)                   # adds range, price_return, ATR_14, efficiency

    # --- Aggregate session‐level statistics ---
    session_stats = calculate_session_statistics(df, timeframe_name)

    return df, session_stats


def calculate_session_statistics(
    df: pd.DataFrame,
    timeframe_name: str = 'Unknown'
) -> pd.DataFrame:
    """
    Given a DataFrame with columns:
      ['session','range','ATR_14','price_return','abs_return','efficiency'],
    compute summary stats per session.
    """
    agg = {
        'range':        ['mean','max','std'],
        'ATR_14':       ['mean','max','std'],
        'price_return': ['mean','std'],
        'abs_return':   ['mean','std'],
        'efficiency':   'mean'
    }
    stats = df.groupby('session').agg(agg)
    # flatten MultiIndex
    stats.columns = ['_'.join(col) for col in stats.columns]
    stats['timeframe'] = timeframe_name
    stats = stats.reset_index()
    return stats
