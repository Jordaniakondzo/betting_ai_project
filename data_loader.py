import pandas as pd

def load_data(filepath):
    """Load data from a CSV file."""
    return pd.read_csv(filepath)
