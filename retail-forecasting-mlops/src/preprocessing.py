import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert date column and sort by date.
    """
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by='date')
    return df
