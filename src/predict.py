import numpy as np
import joblib
import pandas as pd
import os

MODEL_PATH = "models/crop_recommendation_model.pkl"

model = joblib.load(MODEL_PATH)

X_test = np.load(
    "data/processed/X_test.npy"
)

y_test = np.load(
    "data/processed/y_test.npy",
    allow_pickle=True
)

predictions = model.predict(X_test)

probabilities = model.predict_proba(X_test)

results = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions,
    "Probability": probabilities.max(axis=1)
})

os.makedirs("outputs", exist_ok=True)

results.to_csv(
    "outputs/predictions.csv",
    index=False
)

print("Predictions saved to outputs/predictions.csv")

print("\nFirst 10 predictions:")
print(results.head(10))