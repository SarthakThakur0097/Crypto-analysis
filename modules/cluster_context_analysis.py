import pandas as pd

def find_cluster_runs(df, cluster_id, min_length=5):
    """
    Find all contiguous runs of a specific cluster ID of at least min_length.
    """
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


def extract_run_contexts(df, runs, context_size=5):
    """
    Extract the context (returns, efficiency) before and after each cluster run.
    """
    context_data = []
    for start, end in runs:
        before = df.iloc[max(0, start - context_size):start]
        after = df.iloc[end:end + context_size]

        context_data.append({
            'start_time': df.iloc[start]['time'],
            'end_time': df.iloc[end - 1]['time'],
            'before_return_mean': before['return'].mean(),
            'before_efficiency_mean': before['efficiency'].mean(),
            'after_return_mean': after['return'].mean(),
            'after_efficiency_mean': after['efficiency'].mean(),
            'run_length': end - start
        })

    return pd.DataFrame(context_data)
