import numpy as np
import mlflow
import mlflow.sklearn
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train = np.load("data/processed/X_train.npy")
y_train = np.load("data/processed/y_train.npy", allow_pickle=True)

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

mlflow.set_experiment("Crop_Recommendation_MLOps")

models = {
    "Logistic_Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=10, kernel="rbf", probability=True))
    ]),

    "Random_Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

os.makedirs("models", exist_ok=True)

for name, model in models.items():

    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_param("model_name", name)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(
    model,
    "model",
    serialization_format="pickle"
)

        joblib.dump(model, f"models/{name.lower()}.pkl")

        print(f"\n{name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

print("\n[SUCCESS] MLflow Experiment Tracking Completed!")