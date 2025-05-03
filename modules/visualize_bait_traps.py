import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf

def visualize_bait_traps(
    df,
    matches,
    num_charts=5,
    pre_bars=15,
    post_bars=30,
    mode="individual",  # "context" is the new mode
    start_date=None,
    end_date=None
):
    """
    Visualize bait-and-trap setups.

    Parameters:
    - df: full DataFrame with OHLCV + trap flags
    - matches: DataFrame with bait_trap_pattern == True
    - num_charts: number of traps to visualize (for 'individual' mode)
    - pre_bars/post_bars: bars before/after each trap (for 'individual' mode)
    - mode: "individual" or "context"
    - start_date, end_date: for context mode, date strings or Timestamps
    """
    if mode == "individual":
        trap_times = matches.index[:num_charts]

        for i, trap_time in enumerate(trap_times):
            idx = df.index.get_loc(trap_time)
            start_idx = max(0, idx - pre_bars)
            end_idx = min(len(df), idx + post_bars)

            window_df = df.iloc[start_idx:end_idx].copy()

            ohlc = window_df[['open', 'high', 'low', 'close', 'volume']].copy()
            ohlc.index.name = 'Date'

            range_top = df.loc[trap_time, 'range_top']
            range_bot = df.loc[trap_time, 'range_bot']
            hlines = [range_top, range_bot]

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

    elif mode == "context":
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date must be provided for context mode")

        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        df_slice = df.loc[start_date:end_date]
        matches_slice = matches.loc[start_date:end_date]

        ohlc = df_slice[['open', 'high', 'low', 'close', 'volume']].copy()
        ohlc.index.name = 'Date'

        # Trap markers on full context
        trap_lows = df_slice['low'].where(df_slice['bait_trap_pattern'], np.nan)
        trap_markers = pd.Series(trap_lows, index=ohlc.index)

        apds = [
            mpf.make_addplot(trap_markers, type='scatter', markersize=80, marker='x', color='red')
        ]

        # Optional range zone overlay (shading not available in mplfinance, but lines are)
        hlines = []
        if 'range_top' in df_slice.columns and 'range_bot' in df_slice.columns:
            unique_range_tops = df_slice['range_top'].dropna().unique()
            unique_range_bots = df_slice['range_bot'].dropna().unique()
            hlines.extend(unique_range_tops)
            hlines.extend(unique_range_bots)

        mpf.plot(
            ohlc,
            type='candle',
            style='yahoo',
            volume=True,
            title=f"ETH 30m Context: {start_date.date()} to {end_date.date()}",
            hlines=dict(hlines=hlines, colors='gray', linestyle='dotted'),
            addplot=apds,
            figratio=(14, 6),
            figscale=1.3
        )
