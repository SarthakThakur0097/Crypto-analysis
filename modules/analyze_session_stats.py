import pandas as pd
import matplotlib.pyplot as plt
import os

def analyze_session_stats(input_csv, title, output_folder=None):
    """
    Analyze session-based movement and volatility statistics for a given dataset.
    Args:
        input_csv (str): Path to input CSV file.
        title (str): Title for plots and display.
        output_folder (str, optional): If provided, saves the plots there.
    """

    # Load dataset
    df = pd.read_csv(input_csv)

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

    # Calculate core movement columns
    df['range'] = df['high'] - df['low']
    df['return'] = df['close'] - df['open']
    df['abs_return'] = df['return'].abs()

    # Calculate ATR_14
    df['prior_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prior_close']).abs()
    df['tr3'] = (df['low'] - df['prior_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_14'] = df['true_range'].rolling(window=14).mean()

    # Efficiency
    df['efficiency'] = df['range'] / df['ATR_14']

    df.dropna(inplace=True)

    # Label sessions
    def get_session(hour):
    if 0 <= hour < 7:
        return 'Asia'
    elif 7 <= hour < 8:
        return 'Asia + London Overlap'
    elif 8 <= hour < 13:
        return 'London'
    elif 13 <= hour < 15:
        return 'London + NY Overlap'
    elif 15 <= hour < 20:
        return 'New York'
    else:
        return 'Other'


    df['session'] = df.index.hour.map(get_session)

    # Group by session and calculate stats
    session_stats = df.groupby('session').agg({
        'range': ['mean', 'max'],
        'ATR_14': ['mean', 'max'],
        'return': ['mean', 'std'],
        'abs_return': ['mean', 'std'],
        'efficiency': 'mean'
    })

    print(f"\n=== Session Movement and Volatility Stats ({title}) ===")
    print(session_stats)

    # Save session stats if output folder is provided
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        stats_path = os.path.join(output_folder, f'session_stats_{title}.csv')
        session_stats.to_csv(stats_path)
        print(f"✅ Session stats saved to {stats_path}")

    # Plot distributions
    def plot_distribution(series, plot_title, save_name=None):
        plt.figure(figsize=(12,5))
        series.hist(bins=50, alpha=0.75)
        plt.title(plot_title)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.grid(True)
        if output_folder and save_name:
            plt.savefig(os.path.join(output_folder, save_name))
            plt.close()
            print(f"✅ Saved plot: {save_name}")
        else:
            plt.show()

    plot_distribution(df['range'], f'Distribution of Candle Ranges - {title}', f'range_dist_{title}.png')
    plot_distribution(df['return'], f'Distribution of Candle Returns - {title}', f'return_dist_{title}.png')
    plot_distribution(df['ATR_14'], f'Distribution of ATR (14) - {title}', f'atr14_dist_{title}.png')
    plot_distribution(df['efficiency'], f'Distribution of Range Efficiency - {title}', f'efficiency_dist_{title}.png')
