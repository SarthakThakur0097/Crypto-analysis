# modules/analyze_session_stats.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from modules.session_labeler import add_session_labels  # reuse your session logic

def analyze_session_stats(
    input_csv: str,
    title: str,
    output_folder: str = None
):
    """
    Analyze session-based movement and volatility statistics for a given dataset.
    Uses the 'time' column as the index. Saves CSV and PNGs if output_folder is provided.
    """
    # --- Load & index on 'time' ---
    df = pd.read_csv(
        input_csv,
        parse_dates=['time']
    )
    df.set_index('time', inplace=True)
    df.sort_index(inplace=True)

    # --- Core metrics ---
    df['range']        = df['high'] - df['low']
    df['price_return'] = df['close'] - df['open']
    df['abs_return']   = df['price_return'].abs()

    # --- ATR 14 & efficiency ---
    df['prior_close'] = df['close'].shift(1)
    df['tr1']         = df['high'] - df['low']
    df['tr2']         = (df['high'] - df['prior_close']).abs()
    df['tr3']         = (df['low']  - df['prior_close']).abs()
    df['true_range']  = df[['tr1','tr2','tr3']].max(axis=1)
    df['ATR_14']      = df['true_range'].rolling(window=14).mean()
    df['efficiency']  = df['range'] / df['ATR_14']

    # drop rows before ATR warm-up completes
    df.dropna(subset=['ATR_14'], inplace=True)

    # --- Session labeling (reuse existing module) ---
    df = add_session_labels(df)  # adds df['session']

    # --- Aggregate stats per session ---
    agg = {
        'range':        ['mean','max'],
        'ATR_14':       ['mean','max'],
        'price_return': ['mean','std'],
        'abs_return':   ['mean','std'],
        'efficiency':   'mean',
    }
    session_stats = df.groupby('session').agg(agg)

    print(f"\n=== Session Stats: {title} ===")
    print(session_stats)

    # sanitize title for filenames
    safe_title = title.replace(' ', '_')

    # --- Save CSV if requested ---
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        csv_path = os.path.join(output_folder, f"session_stats_{safe_title}.csv")
        session_stats.to_csv(csv_path)
        print(f"→ saved stats CSV: {csv_path}")

    # --- Plot helper ---
    def plot_dist(series: pd.Series, name: str):
        plt.figure(figsize=(10, 4))
        series.hist(bins=50, alpha=0.7)
        plt.title(f"{name} — {title}")
        plt.xlabel(name)
        plt.ylabel("Frequency")
        plt.grid(True)
        if output_folder:
            png_path = os.path.join(output_folder, f"{name.lower()}_dist_{safe_title}.png")
            plt.savefig(png_path, bbox_inches='tight')
            plt.close()
            print(f"→ saved plot: {png_path}")
        else:
            plt.show()

    # --- Plot distributions ---
    plot_dist(df['range'],        "Candle Range")
    plot_dist(df['price_return'], "Candle Return")
    plot_dist(df['ATR_14'],       "ATR_14")
    plot_dist(df['efficiency'],   "Range Efficiency")
