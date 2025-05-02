# modules/session_aggregator.py

import pandas as pd
from typing import Optional, Callable

from modules.session_labeler import add_session_labels

def build_session_ohlc(
    df: pd.DataFrame,
    time_col:    str = 'time',
    session_col: str = 'session',
    label_func:  Optional[Callable] = None
) -> pd.DataFrame:
    """
    Convert bar data into session-level OHLCV.

    – Parses and indexes on `time_col` (defaults to 'time').
    – Labels sessions via `label_func` or your central add_session_labels.
    – Groups by date & session, then computes open/high/low/close/volume.
    – Adds derived 'range' and 'price_return'.

    Args:
        df:         Input DataFrame with a 'time' column or datetime index.
        time_col:   Name of the column to parse & index (must exist if index is not datetime).
        session_col:Name of the session label column produced.
        label_func: Optional function(ts)→session; if None, uses add_session_labels.

    Returns:
        session_ohlc: DataFrame with columns
          ['date','session','open','high','low','close','volume','range','price_return'].
    """
    df = df.copy()

    # 1) Parse & set datetime index on `time_col`
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
        df.set_index(time_col, inplace=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise KeyError(f"No '{time_col}' column and index is not datetime.")
    df.sort_index(inplace=True)

    # 2) Label sessions
    if label_func:
        df[session_col] = df.index.map(label_func)
    else:
        # uses your centralized session logic (including overlaps/tz)
        df = add_session_labels(df)
        session_col = 'session'  # ensure consistency

    # 3) Extract calendar date
    df['date'] = df.index.date

    # 4) Ensure volume exists
    if 'volume' not in df.columns:
        df['volume'] = 0

    # 5) Aggregate to session OHLCV
    agg_dict = {
        'open':   'first',
        'high':   'max',
        'low':    'min',
        'close':  'last',
        'volume': 'sum'
    }
    session_ohlc = (
        df.groupby(['date', session_col], sort=True)
          .agg(agg_dict)
          .reset_index()
    )

    # 6) Derived columns
    session_ohlc['range']        = session_ohlc['high'] - session_ohlc['low']
    session_ohlc['price_return'] = session_ohlc['close'] - session_ohlc['open']

    return session_ohlc
