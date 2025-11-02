# main.py
from fastapi import FastAPI
from pydantic import BaseModel
# import joblib
import numpy as np
import pandas as pd
import mlflow.sklearn
import os

app = FastAPI(title="🌸 Iris Classifier API")

# Load model
# mlflow.set_tracking_uri("http://34.61.28.47:5000")
# model = joblib.load("model.joblib")
# Get the latest/best model (based on accuracy)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
best_run = mlflow.search_runs(order_by=["metrics.accuracy DESC"]).iloc[0]
best_model_uri = f"runs:/{best_run.run_id}/model"

# Load model from MLflow
model = mlflow.sklearn.load_model(best_model_uri)

# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {
        "predicted_class": prediction
    }
