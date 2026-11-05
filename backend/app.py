
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal


# -----------------------------------------
# 1. Project configuration
# -----------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"


# -----------------------------------------
# 2. Initialize FastAPI
# -----------------------------------------

app = FastAPI(
    title="Financial Fraud Detection API",
    description="Machine Learning based transaction fraud detection",
    version="2.0.0"
)

# Temporary CORS configuration for deployment.
# Restrict origins to your frontend URL before production use.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://fraudguard-frontend-5dgq.onrender.com"
],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)


# -----------------------------------------
# 3. Load trained model
# -----------------------------------------

try:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    threshold = artifact["threshold"]
    feature_columns = artifact["features"]

    print("Model loaded successfully!")
    print("Threshold:", threshold)
    print("Number of features:", len(feature_columns))

except Exception as error:
    print("Model loading failed:", error)
    model = None
    threshold = None
    feature_columns = None


# -----------------------------------------
# 4. Input validation
# -----------------------------------------

class TransactionInput(BaseModel):

    step: int = Field(ge=1, le=743)

    amount: float = Field(ge=0, le=1_000_000_000)

    transaction_type: Literal[
        "PAYMENT",
        "TRANSFER",
        "CASH_OUT",
        "DEBIT",
        "CASH_IN"
    ]

    origin_account_type: Literal["C"] = "C"

    destination_account_type: Literal["C", "M"] = "C"


# -----------------------------------------
# 5. Health endpoint
# -----------------------------------------

@app.get("/")
def root():

    return {
        "message": "Financial Fraud Detection API",
        "status": "running",
        "version": "2.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy" if model is not None else "unavailable",
        "model_loaded": model is not None
    }


# -----------------------------------------
# 6. Feature preprocessing
# -----------------------------------------

def prepare_features(transaction):

    data = pd.DataFrame([{
        "step": transaction.step,
        "amount": transaction.amount,
        "log_amount": np.log1p(transaction.amount),
        "type": transaction.transaction_type,
        "origin_account_type": transaction.origin_account_type,
        "destination_account_type": transaction.destination_account_type,
        "destination_is_merchant": int(
            transaction.destination_account_type == "M"
        )
    }])

    data = pd.get_dummies(
        data,
        columns=[
            "type",
            "origin_account_type",
            "destination_account_type"
        ],
        dtype=int
    )

    # Ensure exactly the same feature order as training.
    data = data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    return data


# -----------------------------------------
# 7. Fraud prediction endpoint
# -----------------------------------------

@app.post("/predict")
def predict(transaction: TransactionInput):

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is unavailable"
        )

    try:

        features = prepare_features(transaction)

        probabilities = model.predict_proba(features)[0]

        fraud_index = list(model.classes_).index(1)

        fraud_score = float(probabilities[fraud_index])

        prediction = (
            "Fraudulent Transaction"
            if fraud_score >= threshold
            else "Legitimate Transaction"
        )

        return {
            "prediction": prediction,
            "fraud_score": round(fraud_score * 100, 2),
            "threshold": round(float(threshold), 3),
            "risk_status": (
                "HIGH RISK"
                if fraud_score >= threshold
                else "BELOW DETECTION THRESHOLD"
            ),
            "transaction_details": {
                "amount": transaction.amount,
                "type": transaction.transaction_type,
                "step": transaction.step
            }
        }

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
