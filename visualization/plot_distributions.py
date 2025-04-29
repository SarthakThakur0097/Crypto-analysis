import matplotlib.pyplot as plt
import os

def plot_distribution(series, title, output_folder=None):
    """
    Plot histogram of a series. Optionally save plot.
    """
    plt.figure(figsize=(12,5))
    series.hist(bins=50, alpha=0.75)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True)
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        save_path = os.path.join(output_folder, title.replace(" ", "_") + '.png')
        plt.savefig(save_path)
        plt.close()
        print(f"✅ Saved plot: {save_path}")
    else:
        plt.show()
