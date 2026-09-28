import numpy as np
import joblib
import json
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy", allow_pickle=True)

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    rf,
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best Parameters:")
print(grid.best_params_)

print("Best CV Accuracy:")
print(grid.best_score_)

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/metrics", exist_ok=True)

joblib.dump(
    grid.best_estimator_,
    "models/crop_recommendation_model.pkl"
)

with open(
    "outputs/metrics/best_parameters.json", "w"
) as f:
    json.dump(grid.best_params_, f, indent=4)

print("Tuned model saved.")