import pandas as pd
import numpy as np
import json
import os

from sklearn.model_selection import train_test_split

DATA_PATH = "data/raw/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

X = df[
    ["N", "P", "K", "temperature",
     "humidity", "ph", "rainfall"]
]

y = df["label"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print(y.name)

print("\nFeature Statistics:")
print(X.describe())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

os.makedirs("data/processed", exist_ok=True)

np.save(
    "data/processed/X_train.npy",
    X_train.to_numpy()
)

np.save(
    "data/processed/X_test.npy",
    X_test.to_numpy()
)

np.save(
    "data/processed/y_train.npy",
    y_train.to_numpy()
)

np.save(
    "data/processed/y_test.npy",
    y_test.to_numpy()
)

metadata = {
    "features": X.columns.tolist(),
    "target": "label",
    "train_size": len(X_train),
    "test_size": len(X_test),
    "test_ratio": 0.20,
    "random_state": 42
}

with open(
    "data/processed/dataset_metadata.json",
    "w"
) as f:
    json.dump(metadata, f, indent=4)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print("\nPreprocessing completed successfully.")