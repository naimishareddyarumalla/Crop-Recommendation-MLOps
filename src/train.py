import numpy as np
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy", allow_pickle=True)

os.makedirs("models", exist_ok=True)

models = {
    "logistic_regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "svm": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=10, kernel="rbf", probability=True))
    ]),

    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f"models/{name}.pkl")
    print(f"{name} trained and saved.")

print("Training completed.")