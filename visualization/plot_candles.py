import mplfinance as mpf
import matplotlib.pyplot as plt
from datetime import time

def plot_candles(df, start_time, end_time, title=None):
    """
    Plot candlestick chart for a given datetime range with session open markers.
    
    Args:
        df: (DataFrame) OHLCV data with DateTime index
        start_time: (str) start datetime, e.g., '2025-04-18 00:00'
        end_time: (str) end datetime, e.g., '2025-04-20 12:00'
        title: (str) optional chart title
    """

    # Make sure index is datetime
    df.index = pd.to_datetime(df.index)

    # Slice the desired time range
    plot_df = df.loc[start_time:end_time]

    if plot_df.empty:
        print(f"No candles found between {start_time} and {end_time}.")
        return

    # Rename columns for mplfinance if needed
    plot_df = plot_df[['Open', 'High', 'Low', 'Close', 'Volume']]

    # Step 1: Create figure and axes first
    fig, axes = mpf.plot(plot_df,
                         type='candle',
                         style='charles',
                         volume=True,
                         title=title if title else f'Candles {start_time} to {end_time}',
                         datetime_format='%b %d %H:%M',
                         xrotation=15,
                         returnfig=True)

    ax = axes[0]  # Main price axis

    # Step 2: Add session open lines
    asia_open = time(0, 0)
    london_open = time(8, 0)
    ny_open = time(13, 0)

    # Find timestamps matching session opens
    asia_lines = plot_df.index[plot_df.index.time == asia_open]
    london_lines = plot_df.index[plot_df.index.time == london_open]
    ny_lines = plot_df.index[plot_df.index.time == ny_open]

    for t in asia_lines:
        ax.axvline(t, color='orange', linestyle='--', linewidth=0.7, label='Asia Open' if t==asia_lines[0] else "")
    for t in london_lines:
        ax.axvline(t, color='green', linestyle='--', linewidth=0.7, label='London Open' if t==london_lines[0] else "")
    for t in ny_lines:
        ax.axvline(t, color='blue', linestyle='--', linewidth=0.7, label='NY Open' if t==ny_lines[0] else "")

    # Step 3: Clean up the legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc='upper left')

    # Step 4: Finally show the plot
    plt.show()
