import pandas as pd
from datetime import time

# --- Helper: Label Sessions ---
def label_session(ts):
    h = ts.hour
    if 0 <= h < 7:
        return 'Asia'
    elif 7 <= h < 13:
        return 'London'
    elif 13 <= h < 20:
        return 'New York'
    else:
        return 'Other'

# --- Main Function: Build Session Candles ---
def build_session_ohlc(df, timeframe='30m'):
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Label sessions + date
    df['session'] = df.index.map(label_session)
    df['date'] = df.index.date

    # Aggregate by (date, session)
    grouped = df.groupby(['date', 'session'])

    session_ohlc = grouped.agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum' if 'volume' in df.columns else lambda x: 0
    }).reset_index()

    # Create session timestamp = start of session (midnight)
    session_ohlc['timestamp'] = pd.to_datetime(session_ohlc['date'])

    # Optional: build range, return, etc.
    session_ohlc['range'] = session_ohlc['high'] - session_ohlc['low']
    session_ohlc['return'] = session_ohlc['close'] - session_ohlc['open']

    return session_ohlc
