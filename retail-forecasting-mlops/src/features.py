import pandas as pd

def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag features and rolling mean.
    """
    df['lag_1'] = df['sales'].shift(1)
    df['rolling_mean'] = df['sales'].rolling(7).mean()
    return df
