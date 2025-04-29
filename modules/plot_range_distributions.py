import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_range_distributions(datasets, titles, output_folder=None):
    """
    Plot distribution of candle ranges for multiple datasets side-by-side.

    Args:
        datasets (list of str): Paths to CSV files.
        titles (list of str): Titles corresponding to each dataset.
        output_folder (str, optional): Folder to save plots. If None, plots will just show.
    """

    if output_folder:
        os.makedirs(output_folder, exist_ok=True)

    for i, dataset_path in enumerate(datasets):
        df = pd.read_csv(dataset_path)

        # Detect time column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        else:
            raise ValueError("No recognizable time column found (expected 'timestamp' or 'time').")

        df = df.sort_index()

        # Calculate range
        df['range'] = df['high'] - df['low']

        # Plot
        plt.figure(figsize=(10,6))
        df['range'].hist(bins=50, alpha=0.75)
        plt.title(f'Distribution of Candle Ranges - {titles[i]}')
        plt.xlabel('Range (points)')
        plt.ylabel('Frequency')
        plt.grid(True)

        if output_folder:
            save_path = os.path.join(output_folder, f'range_distribution_{titles[i]}.png')
            plt.savefig(save_path)
            print(f"✅ Saved plot for {titles[i]} to {save_path}")
            plt.close()
        else:
            plt.show()
