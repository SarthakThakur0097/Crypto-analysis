# modules/timestamp_features.py

import pandas as pd
from typing import Optional

def add_timestamp_features(
    df: pd.DataFrame,
    time_col: Optional[str] = 'time',
    tz: Optional[str]       = None
) -> pd.DataFrame:
    """
    Ensure the DataFrame is indexed on the 'time' column (or a datetime index),
    then add the following columns:
      - date (YYYY-MM-DD)
      - time_of_day (HH:MM:SS)
      - hour (0–23)
      - minute (0–59)
      - weekday (0=Mon … 6=Sun)
      - month (1–12)

    If tz is provided, timestamps will be localized to UTC then converted to tz.
    """
    df = df.copy()

    # 1) Parse & set index on time_col if present
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], infer_datetime_format=True)
        df.set_index(time_col, inplace=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise KeyError(f"No '{time_col}' column and index is not datetime.")

    # 2) Optional timezone conversion
    if tz:
        df.index = df.index.tz_localize('UTC').tz_convert(tz)

    # 3) Sort by time
    df.sort_index(inplace=True)

    # 4) Add calendar/time features
    df['date']        = df.index.date
    df['time_of_day'] = df.index.time
    df['hour']        = df.index.hour
    df['minute']      = df.index.minute
    df['weekday']     = df.index.weekday
    df['month']       = df.index.month

    return df
