import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load CSV using pandas and handle missing values.
    """
    df = pd.read_csv(file_path)
    df.fillna(0, inplace=True)
    return df
