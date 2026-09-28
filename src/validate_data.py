import pandas as pd

DATA_PATH = "data/raw/Crop_recommendation.csv"

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nClass Distribution:")
print(df["label"].value_counts())

if df.isnull().sum().sum() == 0:
    print("\nNo missing values found.")

if df.duplicated().sum() == 0:
    print("No duplicate rows found.")

print("\nData validation completed.")