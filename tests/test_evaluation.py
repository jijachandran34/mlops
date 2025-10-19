import requests
from src.evaluate import evaluate_model
from google.cloud import storage
import os

MODEL_BUCKET = "ml_ops_wk1_826"
MODEL_BLOB = "my-models/iris-classifier-week-1/run-20251002-071009/model.joblib"


def test_evaluate_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    client = storage.Client()
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(MODEL_BLOB)
    blob.download_to_filename(model_path)

    results = evaluate_model(str(model_path), "iris.csv")
    assert results["accuracy"] >= 0.6
    assert results["f1"] >= 0.6
