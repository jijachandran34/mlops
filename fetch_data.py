"""
Fetch IRIS data from GCS and save locally
"""
import pandas as pd
import os
import subprocess

def fetch_data():
    """Download data from GCS"""
    gcs_path = "gs://ml_ops_wk1_826/v1/data.csv"
    
    try:
        # Method 1: Try using pandas with gcsfs (works in Cloud Shell)
        try:
            df = pd.read_csv(gcs_path)
        except:
            # Method 2: Use gsutil (always available in Cloud Shell)
            print("Using gsutil to download from GCS...")
            os.makedirs("data", exist_ok=True)
            subprocess.run([
                "gsutil", "cp", gcs_path, "data/iris_raw.csv"
            ], check=True)
            df = pd.read_csv("data/iris_raw.csv")
        
        print(f"Data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Save to local data directory
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/iris_raw.csv", index=False)
        print("Data saved to data/iris_raw.csv")
        
        return df
    except Exception as e:
        print(f"Error fetching data from GCS: {e}")
        print("Falling back to sklearn iris dataset...")
        # Fallback to sklearn iris dataset
        from sklearn.datasets import load_iris
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/iris_raw.csv", index=False)
        print("Using sklearn iris dataset as fallback")
        return df

if __name__ == "__main__":
    fetch_data()