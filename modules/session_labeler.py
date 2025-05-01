import pandas as pd
from datetime import time
import os

def label_session(ts):
    t = ts.time()
    if time(0, 0) <= t < time(7, 0):
        return 'Asia'
    elif time(7, 0) <= t < time(13, 0):
        return 'London'
    elif time(13, 0) <= t < time(20, 0):
        return 'New York'
    else:
        return 'Other'

def label_and_save_sessions(file_path, output_prefix, timestamp_col='timestamp'):
    # Load and sort
    df = pd.read_csv(file_path, parse_dates=[timestamp_col])
    df = df.sort_values(by=timestamp_col)
    df.set_index(timestamp_col, inplace=True)

    # Label session
    df['session'] = df.index.map(label_session)

    # Create folders if not exist
    os.makedirs('./Resampled/raw', exist_ok=True)
    os.makedirs('./Resampled/additional_features', exist_ok=True)

    # Save labeled and feature-ready copies
    raw_path = f'./Resampled/raw/{output_prefix}_labeled.csv'
    features_path = f'./Resampled/additional_features/{output_prefix}_features.csv'

    df.to_csv(raw_path)
    df.to_csv(features_path)

    return raw_path, features_path
