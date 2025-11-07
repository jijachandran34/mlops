import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib, os, datetime

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"gs://practice-826/artifacts/{timestamp}/"
os.makedirs("models", exist_ok=True)

df = pd.read_csv("gs://practice-826/train.csv")
df = df.dropna(subset=["Age", "Fare", "Embarked", "Sex"])
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")

joblib.dump(model, "models/model.joblib")
os.system(f"gsutil cp models/model.joblib {output_dir}")
