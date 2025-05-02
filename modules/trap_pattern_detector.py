# modules/trap_pattern_detector.py

import pandas as pd
import numpy as np

def flag_bait_and_trap(
    df: pd.DataFrame,
    lookahead: int = 15
) -> pd.DataFrame:
    """
    Flags “bait‐and‐trap” patterns:
      1) fake_break_high: bar’s high > range_top but close ≤ range_top
      2) in next `lookahead` bars:
         a) touched support (low ≤ range_bot)
         b) fake_break_low: low < range_bot & close ≥ range_bot
         c) reversal candle (e.g. engulfing_flag)

    Outputs new columns:
      - fake_break_high (bool)
      - fake_break_low  (bool)
      - bait_trap_pattern (bool)
      - time_to_support   (int bars until first support touch)
    """
    df = df.copy()
    df['fake_break_high']   = (
        (df['high'] > df['range_top']) &
        (df['close'] <= df['range_top'])
    )
    df['fake_break_low']    = False
    df['bait_trap_pattern'] = False
    df['time_to_support']   = np.nan

    for ts in df.index[df['fake_break_high']]:
        pos    = df.index.get_loc(ts)
        window = df.iloc[pos+1 : pos+1+lookahead]

        touched = window['low'] <= df.at[ts, 'range_bot']
        fake_low = (
            (window['low'] < df.at[ts, 'range_bot']) &
            (window['close'] >= df.at[ts, 'range_bot'])
        )
        # if you have candle‐structure flags, include them:
        rev = (
            window.get('engulfing_flag', False) |
            window.get('strong_close', False)
        )

        if touched.any():
            first_loc = touched.idxmax()
            df.at[ts, 'time_to_support'] = window.index.get_loc(first_loc)
        if touched.any() and fake_low.any() and rev.any():
            df.at[ts, 'fake_break_low']    = True
            df.at[ts, 'bait_trap_pattern'] = True

    return df
