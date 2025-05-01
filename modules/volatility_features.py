import pandas as pd

def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Basic movement
    df['abs_return'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']

    # Prior close
    df['prior_close'] = df['close'].shift(1)

    # True range components
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prior_close']).abs()
    df['tr3'] = (df['low'] - df['prior_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # ATR
    df['ATR_14'] = df['true_range'].rolling(window=14).mean()

    # Efficiency
    df['efficiency'] = df['range'] / df['ATR_14']

    # Volatility change dynamics
    df['range_change'] = df['range'] / df['range'].shift(1)
    df['efficiency_change'] = df['efficiency'] / df['efficiency'].shift(1)

    # Volatility spike detection
    df['volatility_spike'] = df['range'] > (2 * df['ATR_14'])
    
    # Rolling volatility std
    df['rolling_volatility_std'] = df['true_range'].rolling(window=10).std()

    return df
