from src.data_validation import load_data, validate_data

def test_validate_data():
    df = load_data("iris.csv")   # uses your DVC-tracked iris.csv
    assert validate_data(df) is True
