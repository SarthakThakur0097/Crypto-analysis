def calculate_movement_metrics(df):
    """Add range, return, ATR, efficiency to the dataframe."""
    df['range'] = df['High'] - df['Low']
    df['return'] = df['Close'] - df['Open']
    df['abs_return'] = df['return'].abs()
    
    df['prior_close'] = df['Close'].shift(1)
    df['tr1'] = df['High'] - df['Low']
    df['tr2'] = (df['High'] - df['prior_close']).abs()
    df['tr3'] = (df['Low'] - df['prior_close']).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR_14'] = df['true_range'].rolling(window=14).mean()
    
    df['efficiency'] = df['range'] / df['ATR_14']
    
    df.dropna(inplace=True)
    return df
