# modules/session_labeler.py

import os
import pandas as pd
from datetime import time
from typing import Optional, Tuple
from modules.session_labeling import label_session  # your overlap-aware helper

def add_session_labels(
    df: pd.DataFrame,
    timestamp_col: str         = 'time',
    tz: Optional[str]         = None
) -> pd.DataFrame:
    """
    Add a 'session' column to df based on UTC time (with overlaps).
    Expects a 'time' datetime column or datetime index.
    """
    df = df.copy()

    # 1) parse & set index on `timestamp_col`
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df.set_index(timestamp_col, inplace=True)
    elif not pd.api.types.is_datetime64_any_dtype(df.index):
        raise KeyError(f"No '{timestamp_col}' column and index is not datetime.")

    # 2) optional timezone conversion
    if tz:
        df.index = df.index.tz_localize('UTC').tz_convert(tz)

    # 3) map timestamps through your overlap-aware helper
    df['session'] = df.index.map(label_session)
    return df

def label_and_save_sessions(
    file_path:     str,
    output_prefix: str,
    timestamp_col: str          = 'time',
    output_base:   str          = './Resampled'
) -> Tuple[str, str]:
    """
    Read CSV, label sessions (with overlaps), and save:
      • raw labeled CSV in output_base/raw/
      • feature-ready CSV in output_base/additional_features/
    Returns (raw_path, features_path).
    """
    df = pd.read_csv(file_path, parse_dates=[timestamp_col])
    df = add_session_labels(df, timestamp_col=timestamp_col)

    raw_dir  = os.path.join(output_base, 'raw')
    feat_dir = os.path.join(output_base, 'additional_features')
    os.makedirs(raw_dir,  exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    raw_path      = os.path.join(raw_dir,  f'{output_prefix}_labeled.csv')
    features_path = os.path.join(feat_dir, f'{output_prefix}_features.csv')

    df.to_csv(raw_path,      index=True, date_format='%Y-%m-%d %H:%M:%S')
    df.to_csv(features_path, index=True, date_format='%Y-%m-%d %H:%M:%S')
    return raw_path, features_path
