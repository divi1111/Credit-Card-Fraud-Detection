
import pandas as pd
import numpy as np
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score
)

# -----------------------------------------
# 1. Define paths
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "processed" / "fraud_dataset.csv"

MODEL_PATH = BASE_DIR / "backend" / "model.joblib"

# -----------------------------------------
# 2. Load processed dataset
# -----------------------------------------

print("Loading processed dataset...")

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["isFraud"])
y = df["isFraud"]

print("Dataset shape:", df.shape)

# -----------------------------------------
# 3. Split training and validation data
# -----------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)

print("\nTraining samples:", len(X_train))
print("Validation samples:", len(X_val))
print("Test samples:", len(X_test))

# -----------------------------------------
# 4. Train Random Forest
# -----------------------------------------

print("\nTraining Random Forest model...")

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=18,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced_subsample",
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training completed!")

# -----------------------------------------
# 5. Select classification threshold
# -----------------------------------------

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.05)

best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:

    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = threshold

print("\nSelected threshold:", best_threshold)
print("Validation F1:", best_f1)

# -----------------------------------------
# 6. Evaluate on held-out test set
# -----------------------------------------

test_probabilities = model.predict_proba(X_test)[:, 1]

test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        test_predictions,
        target_names=["Legitimate", "Fraud"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, test_predictions))

print("\nPrecision:", precision_score(
    y_test, test_predictions, zero_division=0
))

print("Recall:", recall_score(
    y_test, test_predictions, zero_division=0
))

print("F1 Score:", f1_score(
    y_test, test_predictions, zero_division=0
))

print("PR-AUC:", average_precision_score(
    y_test, test_probabilities
))

print("ROC-AUC:", roc_auc_score(
    y_test, test_probabilities
))

# -----------------------------------------
# 7. Save model and threshold
# -----------------------------------------

artifact = {
    "model": model,
    "threshold": best_threshold,
    "features": X.columns.tolist()
}

joblib.dump(artifact, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
