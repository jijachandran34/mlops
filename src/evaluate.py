import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model_path: str, data_path: str):
    """Load model and dataset, compute accuracy & F1 score."""
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]

    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average="macro")
    return {"accuracy": acc, "f1": f1}
