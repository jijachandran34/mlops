import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model_path: str, data_path: str):
    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
    y = df["target"]
    preds = model.predict(X)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "f1": float(f1_score(y, preds, average="macro"))
    }
