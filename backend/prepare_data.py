
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------------------
# 1. Define project paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "PS_20174392719_1491204439457_log.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "fraud_dataset.csv"

# -----------------------------------------
# 2. Load original dataset
# -----------------------------------------

print("Loading PaySim dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, low_memory=False)

print("\nOriginal dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())

# -----------------------------------------
# 3. Validate required columns
# -----------------------------------------

required_columns = [
    "step", "type", "amount",
    "nameOrig", "nameDest",
    "isFraud"
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing columns: {missing_columns}")

# -----------------------------------------
# 4. Inspect dataset quality
# -----------------------------------------

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nFraud distribution:")
print(df["isFraud"].value_counts())

print("\nFraud percentage:")
print(df["isFraud"].value_counts(normalize=True) * 100)

# -----------------------------------------
# 5. Remove invalid records
# -----------------------------------------

df = df.dropna(subset=required_columns)

df = df[
    (df["amount"] >= 0) &
    (df["step"] >= 1)
].copy()

# -----------------------------------------
# 6. Feature engineering
# -----------------------------------------

print("\nCreating useful transaction features...")

# Extract account categories without retaining raw IDs
df["origin_account_type"] = df["nameOrig"].str[0]
df["destination_account_type"] = df["nameDest"].str[0]

# Identify merchant destination accounts
df["destination_is_merchant"] = (
    df["nameDest"].str.startswith("M")
).astype(int)

# Transform transaction amount to reduce skew
df["log_amount"] = np.log1p(df["amount"])

# -----------------------------------------
# 7. Select features
# -----------------------------------------

# Exclude balance columns and isFlaggedFraud:
# they can introduce misleading shortcuts.

feature_columns = [
    "step",
    "amount",
    "log_amount",
    "type",
    "origin_account_type",
    "destination_account_type",
    "destination_is_merchant"
]

X = df[feature_columns].copy()
y = df["isFraud"].astype(int)

# One-hot encode categorical features
X = pd.get_dummies(
    X,
    columns=[
        "type",
        "origin_account_type",
        "destination_account_type"
    ],
    dtype=int
)

# -----------------------------------------
# 8. Handle class imbalance
# -----------------------------------------

print("\nOriginal class distribution:")
print(y.value_counts())

# Keep all fraud cases and sample legitimate cases.
TARGET_ROWS = 500_000

fraud_data = X[y == 1].copy()
legitimate_data = X[y == 0].copy()

if len(fraud_data) >= TARGET_ROWS:
    raise ValueError(
        "Fraud records exceed target sample size. "
        "Increase TARGET_ROWS."
    )

legitimate_count = TARGET_ROWS - len(fraud_data)

if legitimate_count > len(legitimate_data):
    legitimate_count = len(legitimate_data)

legitimate_sample = legitimate_data.sample(
    n=legitimate_count,
    random_state=42
)

fraud_sample = fraud_data

X_sample = pd.concat(
    [legitimate_sample, fraud_sample]
)

y_sample = pd.concat([
    pd.Series(
        0,
        index=legitimate_sample.index,
        name="isFraud"
    ),
    pd.Series(
        1,
        index=fraud_sample.index,
        name="isFraud"
    )
])

# Shuffle rows together
sampled_data = X_sample.copy()
sampled_data["isFraud"] = y_sample

sampled_data = sampled_data.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# -----------------------------------------
# 9. Save processed dataset
# -----------------------------------------

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

sampled_data.to_csv(OUTPUT_PATH, index=False)

print("\nPreprocessing completed!")
print("Processed dataset shape:", sampled_data.shape)

print("\nFinal class distribution:")
print(sampled_data["isFraud"].value_counts())

print("\nFraud percentage:")
print(
    sampled_data["isFraud"].value_counts(normalize=True) * 100
)

print("\nSaved to:", OUTPUT_PATH)
