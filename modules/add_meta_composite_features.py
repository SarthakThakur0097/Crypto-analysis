# modules/meta_composite_features.py

import pandas as pd

def add_meta_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add composite features:
      - trend_strength
      - volatility_normalized_return  (uses price_return / ATR_14)
      - body_wick_alignment_score
      - session_bias_score
    """
    df = df.copy()

    # alias price_return → return for backward compatibility
    if 'price_return' in df.columns and 'return' not in df.columns:
        df['return'] = df['price_return']

    # now we can safely refer to df['return']
    df['trend_strength']               = df['abs_return'] * df['efficiency']
    df['volatility_normalized_return'] = df['price_return'] / df['ATR_14']
    df['body_wick_alignment_score']    = (
        (df['wick_top']    < df['body_ratio']) &
        (df['wick_bottom'] < df['body_ratio'])
    )
    # use the aliased 'return' here:
    df['session_bias_score'] = (
        df.groupby('session')['return']
          .transform(lambda x: x.rolling(3, min_periods=1).mean())
    )

    return df
