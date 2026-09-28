import os

required_files = [
    "data/processed/X_train.npy",
    "data/processed/X_test.npy",
    "data/processed/y_train.npy",
    "data/processed/y_test.npy",
    "data/processed/dataset_metadata.json",
    "models/logistic_regression.pkl",
    "models/svm.pkl",
    "models/random_forest.pkl"
]

failed = False

print("Validating project files...\n")

for file in required_files:

    if os.path.exists(file):
        print("FOUND:", file)

    else:
        print("MISSING:", file)
        failed = True

if failed:
    print("\nValidation failed.")
else:
    print("\nValidation passed successfully.")