import pandas as pd
def add_timestamp_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['date'] = df.index.date
    df['time_of_day'] = df.index.time
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    return df
