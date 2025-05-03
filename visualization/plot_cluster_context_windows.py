import pandas as pd
import mplfinance as mpf

def find_cluster_runs(df, cluster_id, min_length=5):
    runs = []
    in_run = False
    start = None

    for i in range(len(df)):
        if df['cluster'].iloc[i] == cluster_id:
            if not in_run:
                in_run = True
                start = i
        else:
            if in_run:
                end = i
                if (end - start) >= min_length:
                    runs.append((start, end))
                in_run = False

    if in_run and (len(df) - start) >= min_length:
        runs.append((start, len(df)))

    return runs

def plot_cluster_context_windows(df, runs, cluster_id, context_size=5, num_plots=10):
    df_plot = df.set_index('time')
    plotted = 0

    for i, (start, end) in enumerate(runs):
        if plotted >= num_plots:
            break

        plot_start = max(0, start - context_size)
        plot_end = min(len(df), end + context_size)
        window_df = df_plot.iloc[plot_start:plot_end][['open', 'high', 'low', 'close', 'volume']]

        # Highlight cluster candles
        highlight_vals = [
            window_df['close'].iloc[j] if (start - plot_start) <= j < (end - plot_start) else None
            for j in range(len(window_df))
        ]
        highlight_series = pd.Series(highlight_vals, index=window_df.index)

        # Add vertical lines for cluster boundaries
        vline_times = [
            window_df.index[start - plot_start],
            window_df.index[end - plot_start - 1]
        ]

        ap = [
            mpf.make_addplot(
                highlight_series,
                type='scatter',
                color='blue',
                markersize=50,
                marker='.'
            )
        ]

        mpf.plot(
            window_df,
            type='candle',
            style='charles',
            title=f"Cluster {cluster_id} Context Run {i+1}",
            volume=True,
            addplot=ap,
            vlines=dict(vlines=vline_times, linestyle='--', linewidths=1.2, colors='red')
        )

        plotted += 1