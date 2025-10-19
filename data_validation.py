import pandas as pd

EXPECTED_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]

def validate_data(df: pd.DataFrame):
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    if df.isnull().any().any():
        raise ValueError("NaN values found in dataset")
    return True
