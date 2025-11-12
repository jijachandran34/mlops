#fast api  app
from fastapi import FastAPI
from pydantic import BaseModel
# import joblib
import mlflow.sklearn
import mlflow
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os

app = FastAPI(title="🌸 Iris Classifier API")

# Load model
try:
    # Load model from MLflow
    print("starting to load")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    print(f"📡 Using MLflow tracking URI: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    runs = mlflow.search_runs(order_by=["metrics.accuracy DESC"])
    if runs.empty:
        raise ValueError("No runs found in MLflow registry.")
    best_run_id = runs.iloc[0].run_id
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"✅ Loaded model from MLflow: {model_uri}")
except Exception as e:
    print(f"⚠️ Warning: Could not load model from MLflow ({e}). Using fallback model.")
    # fallback model to keep API alive
    iris = load_iris()
    X, y = iris.data, iris.target
    model = RandomForestClassifier().fit(X, y)



# Input schema
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Iris Classifier API!!"}

@app.post("/predict/")
def predict_species(data: IrisInput):
    try:
        input_df = pd.DataFrame([data.dict()])
        prediction = model.predict(input_df)[0]
        return {"predicted_class": int(prediction)}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("iris_fastapi:app", host="0.0.0.0", port=8080)
