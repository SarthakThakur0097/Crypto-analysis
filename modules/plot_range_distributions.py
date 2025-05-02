# modules/plot_range_distributions.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional

def plot_range_distributions(
    datasets: List[str],
    titles:   List[str],
    output_folder: Optional[str] = None,
    bins:     int      = 50
) -> None:
    """
    Plot the distribution of candle ranges for multiple CSV datasets.
    Assumes each CSV has:
      - a 'time' column parsable as datetime
      - lowercase 'high' and 'low' columns

    Args:
        datasets:      List of CSV file paths.
        titles:        List of titles (must match datasets in length).
        output_folder: If provided, PNGs are saved there; otherwise, plots show on-screen.
        bins:          Number of histogram bins.
    """
    if len(datasets) != len(titles):
        raise ValueError("`datasets` and `titles` must be the same length.")

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for path, title in zip(datasets, titles):
        # Load with 'time' as datetime index
        df = pd.read_csv(path, parse_dates=['time'], infer_datetime_format=True)
        if 'time' not in df.columns:
            raise ValueError(f"No 'time' column in {path}")
        df.set_index('time', inplace=True)
        df.sort_index(inplace=True)

        # Validate price columns
        if 'high' not in df.columns or 'low' not in df.columns:
            raise ValueError(f"Expected 'high' and 'low' columns in {path}")

        # Compute range
        df['range'] = df['high'] - df['low']

        # Plot histogram
        plt.figure(figsize=(10, 6))
        df['range'].hist(bins=bins, alpha=0.75)
        plt.title(f'Distribution of Candle Ranges — {title}')
        plt.xlabel('Range')
        plt.ylabel('Frequency')
        plt.grid(True)

        if output_folder:
            # Sanitize title for filename
            safe_title = "".join(ch if ch.isalnum() else "_" for ch in title)
            save_path = os.path.join(output_folder, f'range_dist_{safe_title}.png')
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"✅ Saved plot for \"{title}\" → {save_path}")
        else:
            plt.show()
