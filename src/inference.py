import os
import joblib
import pandas as pd
from google.cloud import storage
from sklearn.metrics import accuracy_score, classification_report
import tempfile

# ----------------------------
# CONFIGURATION
# ----------------------------
BUCKET_URI = f"gs://practice-826"
BUCKET_NAME = "practice-826"   # Replace with your bucket name
ARTIFACTS_PATH = "artifacts"           # Folder in GCS where models are stored
DATA_PATH = "test.csv"     # Titanic dataset in GCS
LOCAL_MODEL_PATH = "models/model.joblib"
LOCAL_OUTPUT_PATH = "outputs/predictions.csv"


# ----------------------------
# FUNCTION: Get Latest Model Path
# ----------------------------
def get_latest_model_from_gcs(bucket_name, prefix):
    """Fetch the latest model.pkl file from GCS based on timestamp folder name."""
    client = storage.Client()
    bucket = client.get_bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    # Filter model.pkl paths
    model_paths = [b.name for b in blobs if b.name.endswith("model.joblib")]
    if not model_paths:
        raise FileNotFoundError("No model.pkl found in GCS artifacts directory.")

    # Sort by timestamp (folder naming convention ensures order)
    model_paths.sort(reverse=True)
    latest_model_path = model_paths[0]
    print(f"🟢 Latest model found: {latest_model_path}")
    return latest_model_path


# ----------------------------
# FUNCTION: Download File from GCS
# ----------------------------
def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a file from the bucket."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print(f"✅ Model downloaded to {destination_file_name}")


# ----------------------------
# FUNCTION: Load Dataset from GCS
# ----------------------------
def load_dataset_from_gcs(bucket_name, gcs_file_path):
    """Loads CSV file directly from GCS into Pandas."""
    file_uri = f"gs://{bucket_name}/{gcs_file_path}"
    print(f"📥 Loading dataset from {file_uri}")
    df = pd.read_csv(file_uri)
    return df


# ----------------------------
# FUNCTION: Preprocess Data
# ----------------------------
def preprocess_data(df):
    """Prepare Titanic dataset for inference."""
    df = df.copy()  # avoid SettingWithCopyWarning
    df = df.dropna(subset=["Age", "Fare", "Embarked", "Sex", "Pclass", "SibSp"])
    df.loc[:, "Sex"] = df["Sex"].map({"male": 0, "female": 1})
    X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
    y = df.get("Survived")
    return df, X, y


# ----------------------------
# FUNCTION: Upload Output to GCS
# ----------------------------
def upload_to_gcs(bucket_name, local_file, destination_blob_name):
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file)
    print(f"📤 Uploaded predictions to gs://{bucket_name}/{destination_blob_name}")


# ----------------------------
# MAIN
# ----------------------------
def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # 1️⃣ Get latest model path from GCS
    latest_model_blob = get_latest_model_from_gcs(BUCKET_NAME, ARTIFACTS_PATH)


    # 2️⃣ Download the model locally
    download_blob(BUCKET_NAME, latest_model_blob, LOCAL_MODEL_PATH)

    # 3️⃣ Load dataset from GCS
    df = load_dataset_from_gcs(BUCKET_NAME, DATA_PATH)
    df_clean, X, y = preprocess_data(df)

    # 4️⃣ Load model and predict
    model = joblib.load(LOCAL_MODEL_PATH)
    preds = model.predict(X)
    df_clean["Predicted_Survival"] = preds

    # 5️⃣ Evaluate (if labels exist)
    if y is not None and not y.isnull().any():
        acc = accuracy_score(y, preds)
        print(f"🔹 Accuracy: {acc:.4f}")
        print("Classification Report:")
        print(classification_report(y, preds))
    else:
        print("⚠️ No ground truth labels found — skipping evaluation.")

    # 6️⃣ Save predictions and upload
    df_clean[["PassengerId", "Predicted_Survival"]].to_csv(LOCAL_OUTPUT_PATH, index=False)

    output_blob = f"predictions/{os.path.basename(LOCAL_OUTPUT_PATH)}"
    upload_to_gcs(BUCKET_NAME, LOCAL_OUTPUT_PATH, output_blob)

    print("✅ Inference completed successfully.")


if __name__ == "__main__":
    main()
