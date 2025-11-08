import pandas as pd

EXPECTED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]

def load_data(path: str):
    """Load CSV into pandas DataFrame."""
    return pd.read_csv(path)

def validate_data(df: pd.DataFrame) -> bool:
    """Validate schema and missing values."""
    missing_cols = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing columns: {missing_cols}")
    if df.isnull().any().any():
        raise ValueError("NaN values found in dataset")
    return True
