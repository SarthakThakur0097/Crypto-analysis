import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def visualize_bait_traps(df, matches, num_charts=5, pre_bars=15, post_bars=30):
    """
    Visualize bait-and-trap setups with additional price context.

    Parameters:
    - df: full DataFrame with OHLCV + bait_trap_pattern
    - matches: subset of df where bait_trap_pattern == True
    - num_charts: number of patterns to visualize
    - pre_bars: candles before the trap to show
    - post_bars: candles after the trap to show
    """
    trap_times = matches.index[:num_charts]

    for i, trap_time in enumerate(trap_times):
        idx = df.index.get_loc(trap_time)

        start_idx = max(0, idx - pre_bars)
        end_idx = min(len(df), idx + post_bars)

        window_df = df.iloc[start_idx:end_idx].copy()

        # Format for mplfinance
        ohlc = window_df[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc.index.name = 'Date'

        # Dashed range lines from trap candle
        range_top = df.loc[trap_time, 'range_top']
        range_bot = df.loc[trap_time, 'range_bot']
        hlines = [range_top, range_bot]

        # Highlight the trap candle with a blue marker at the low
        trap_marker = np.full(len(ohlc), np.nan)
        trap_marker[idx - start_idx] = ohlc['low'].iloc[idx - start_idx]
        highlight_series = pd.Series(trap_marker, index=ohlc.index)

        apds = [
            mpf.make_addplot(highlight_series, type='scatter', markersize=100, marker='*', color='blue')
        ]

        mpf.plot(
            ohlc,
            type='candle',
            style='yahoo',
            volume=True,
            title=f"Bait & Trap Pattern @ {trap_time.strftime('%Y-%m-%d %H:%M')}",
            hlines=dict(hlines=hlines, colors='red', linestyle='dashed'),
            addplot=apds,
            figratio=(10, 6),
            figscale=1.2
        )


