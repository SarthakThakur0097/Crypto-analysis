# modules/metrics_calculator.py

import pandas as pd
from typing import Union, Optional

def calculate_movement_metrics(
    df: pd.DataFrame,
    high_col:  Optional[str] = None,
    low_col:   Optional[str] = None,
    open_col:  Optional[str] = None,
    close_col: Optional[str] = None,
    atr_window: int = 14
) -> pd.DataFrame:
    """
    Add movement & volatility metrics to a DataFrame:
      - range, price_return, abs_return
      - true_range, ATR_<atr_window>
      - efficiency

    Automatically handles input columns named 'High'/'Low'/'Open'/'Close'
    (any casing), or you can override via high_col/low_col/open_col/close_col.

    Args:
        df:           Input DataFrame.
        high_col:     Override name for the high column.
        low_col:      Override name for the low column.
        open_col:     Override name for the open column.
        close_col:    Override name for the close column.
        atr_window:   Window size for ATR rolling mean.

    Returns:
        A new DataFrame (copy) with these added columns:
          'range', 'price_return', 'abs_return', 'ATR_<n>', 'efficiency'
    """
    df = df.copy()

    # Helper to detect column names
    def _find(provided, choices):
        if provided and provided in df.columns:
            return provided
        for c in choices:
            if c in df.columns:
                return c
        raise KeyError(f"None of {choices} found in DataFrame columns.")

    hi = _find(high_col,  ['high', 'High'])
    lo = _find(low_col,   ['low',  'Low'])
    op = _find(open_col,  ['open', 'Open'])
    cl = _find(close_col, ['close','Close'])

    # Standardize to lowercase
    df = df.rename(columns={hi: 'high', lo: 'low', op: 'open', cl: 'close'})

    # Basic movement metrics
    df['range']        = df['high'] - df['low']
    df['price_return'] = df['close'] - df['open']
    df['abs_return']   = df['price_return'].abs()

    # True range components
    df['prior_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prior_close']).abs()
    df['tr3'] = (df['low']  - df['prior_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

    # ATR and efficiency
    atr_col = f'ATR_{atr_window}'
    df[atr_col] = df['true_range'].rolling(window=atr_window).mean()
    df['efficiency'] = df['range'] / df[atr_col]

    # Drop rows where ATR is NaN (warm-up)
    df.dropna(subset=[atr_col], inplace=True)

    # Clean up helper columns
    df.drop(columns=['prior_close', 'tr1', 'tr2', 'tr3', 'true_range'], inplace=True)

    return df
