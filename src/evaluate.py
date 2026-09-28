import numpy as np
import joblib
import os
import json

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

X_test = np.load("data/processed/X_test.npy")
y_test = np.load("data/processed/y_test.npy", allow_pickle=True)

os.makedirs("outputs/metrics", exist_ok=True)
os.makedirs("outputs/reports", exist_ok=True)

models = [
    "logistic_regression",
    "svm",
    "random_forest"
]

results = {}

for name in models:

    model = joblib.load(f"models/{name}.pkl")
    prediction = model.predict(X_test)

    accuracy = accuracy_score(y_test, prediction)
    precision = precision_score(
        y_test, prediction, average="weighted"
    )
    recall = recall_score(
        y_test, prediction, average="weighted"
    )
    f1 = f1_score(
        y_test, prediction, average="weighted"
    )

    results[name] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    print("\n", name)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    report = classification_report(y_test, prediction)

    with open(
        f"outputs/reports/{name}_report.txt", "w"
    ) as f:
        f.write(report)

    matrix = confusion_matrix(y_test, prediction)

    np.savetxt(
        f"outputs/metrics/{name}_confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d"
    )

with open(
    "outputs/metrics/model_metrics.json", "w"
) as f:
    json.dump(results, f, indent=4)

print("\nEvaluation completed.")