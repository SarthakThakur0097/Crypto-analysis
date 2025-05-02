# modules/volatility_features.py

import pandas as pd
from typing import Optional, Union

def add_volatility_features(
    df: pd.DataFrame,
    atr_window:        int = 14,
    std_window:        int = 10,
    spike_multiplier: float = 2.0,
    high_col:   Optional[str] = None,
    low_col:    Optional[str] = None,
    open_col:   Optional[str] = None,
    close_col:  Optional[str] = None
) -> pd.DataFrame:
    """
    Add volatility features to df:
      - price_return, abs_return, range
      - ATR_<atr_window>, efficiency
      - range_change, efficiency_change
      - volatility_spike (range > spike_multiplier * ATR)
      - rolling_volatility_std (std of true_range over std_window)
    
    Automatically handles columns named High/Low/Open/Close (case-insensitive),
    or you can override via high_col/low_col/open_col/close_col.
    """
    df = df.copy()

    # helper to find column names
    def _find(name, choices):
        if name and name in df.columns:
            return name
        for c in choices:
            if c in df.columns:
                return c
        raise KeyError(f"Could not find any of {choices} in DataFrame")

    hi = _find(high_col,  ['high','High'])
    lo = _find(low_col,   ['low','Low'])
    op = _find(open_col,  ['open','Open'])
    cl = _find(close_col, ['close','Close'])

    # standardize to lowercase
    df = df.rename(columns={hi:'high', lo:'low', op:'open', cl:'close'})

    # basic movement
    df['range']        = df['high'] - df['low']
    df['price_return'] = df['close'] - df['open']
    df['abs_return']   = df['price_return'].abs()

    # true range components
    df['prior_close'] = df['close'].shift(1)
    df['tr1']         = df['high'] - df['low']
    df['tr2']         = (df['high'] - df['prior_close']).abs()
    df['tr3']         = (df['low']  - df['prior_close']).abs()
    df['true_range']  = df[['tr1','tr2','tr3']].max(axis=1)

    # ATR & efficiency
    atr_col = f'ATR_{atr_window}'
    df[atr_col] = df['true_range'].rolling(window=atr_window).mean()
    df['efficiency'] = df['range'] / df[atr_col]

    # drop only rows missing ATR
    df.dropna(subset=[atr_col], inplace=True)

    # volatility change dynamics
    df['range_change'] = df['range'] / df['range'].shift(1)
    df['efficiency_change'] = df['efficiency'] / df['efficiency'].shift(1)

    # spike detection
    df['volatility_spike'] = df['range'] > (spike_multiplier * df[atr_col])

    # rolling std of true_range
    df['rolling_volatility_std'] = df['true_range'].rolling(window=std_window).std()

    # clean up intermediates
    df.drop(columns=['prior_close','tr1','tr2','tr3','true_range'], inplace=True)

    return df
