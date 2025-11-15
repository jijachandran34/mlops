"""
Train model and log metrics with MLFlow
"""
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import argparse
import os
import json

def train_model(data_file, experiment_name, poison_level=0):
    """Train model and log metrics"""
    
    # Set MLFlow tracking
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment(experiment_name)
    
    # Load data
    df = pd.read_csv(data_file)
    
    # Identify target column
    possible_targets = ['target', 'species', 'class', 'label', 'y']
    target_col = None
    for col in possible_targets:
        if col in df.columns:
            target_col = col
            break
    
    if not target_col:
        raise ValueError(f"No target column found. Available columns: {df.columns.tolist()}")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Convert string labels to numeric
    from sklearn.preprocessing import LabelEncoder
    if y.dtype == 'object' or y.dtype.name == 'category':
        le = LabelEncoder()
        y = le.fit_transform(y)
        print(f"Encoded labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Start MLFlow run
    with mlflow.start_run():
        # Train model
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        
        # Predictions
        y_pred = clf.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        
        # Log parameters
        mlflow.log_param("poison_level", poison_level)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("data_file", data_file)
        mlflow.log_param("target_column", target_col)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        
        # Log model
        mlflow.sklearn.log_model(clf, "model")
        
        # Save metrics to file for CI
        metrics = {
            "poison_level": poison_level,
            "accuracy": float(accuracy),
            "f1_score": float(f1),
            "precision": float(precision),
            "recall": float(recall),
            "data_file": data_file,
            "target_column": target_col
        }
        
        os.makedirs("reports", exist_ok=True)
        report_file = f"reports/metrics_poison_{int(poison_level)}.json"
        with open(report_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"\nMetrics for poison level {poison_level}%:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Metrics saved to {report_file}")
        
        return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--experiment", default="iris_experiment")
    parser.add_argument("--poison-level", type=float, default=0)
    args = parser.parse_args()
    
    train_model(args.data, args.experiment, args.poison_level)