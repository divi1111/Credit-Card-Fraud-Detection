"""
Credit Card Fraud Detection - FastAPI Backend
This script creates a REST API endpoint for fraud prediction.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import uvicorn
from typing import Literal

# Initialize FastAPI app
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description="API for detecting fraudulent credit card transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model and label encoder
try:
    model = joblib.load('model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    print("Model and label encoder loaded successfully!")
except FileNotFoundError as e:
    print(f"ERROR: Could not load model files. Please run model_train.py first.")
    print(f"Details: {e}")
    model = None
    label_encoder = None

# Define input schema using Pydantic
class TransactionInput(BaseModel):
    """
    Schema for incoming transaction data from frontend.
    All fields match the expected input from the HTML form.
    """
    step: int = Field(..., ge=1, description="Time step of the transaction")
    types: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"] = Field(
        ..., description="Type of transaction"
    )
    amount: float = Field(..., ge=0, description="Transaction amount")
    oldbalanceorig: float = Field(..., ge=0, description="Original balance before transaction")
    newbalanceorig: float = Field(..., ge=0, description="New balance after transaction")
    oldbalancedest: float = Field(..., ge=0, description="Destination original balance")
    newbalancedest: float = Field(..., ge=0, description="Destination new balance")
    isflaggedfraud: int = Field(..., ge=0, le=1, description="System fraud flag (0 or 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "types": "TRANSFER",
                "amount": 150000.0,
                "oldbalanceorig": 200000.0,
                "newbalanceorig": 50000.0,
                "oldbalancedest": 100000.0,
                "newbalancedest": 250000.0,
                "isflaggedfraud": 0
            }
        }

# Define output schema
class PredictionOutput(BaseModel):
    """
    Schema for prediction response.
    """
    prediction: str
    probability: float = Field(..., description="Confidence score (0-1)")
    details: dict

@app.get("/")
async def root():
    """
    Root endpoint to check if API is running.
    """
    return {
        "message": "Credit Card Fraud Detection API is running!",
        "status": "active",
        "endpoints": {
            "/predict": "POST - Make fraud prediction",
            "/health": "GET - Check API health",
            "/docs": "GET - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint to verify model is loaded.
    """
    if model is None or label_encoder is None:
        return {
            "status": "error",
            "message": "Model not loaded. Please run model_train.py first."
        }
    return {
        "status": "healthy",
        "model_loaded": True,
        "message": "API is ready to make predictions"
    }

@app.post("/predict", response_model=PredictionOutput)
async def predict_fraud(transaction: TransactionInput):
    """
    Main prediction endpoint.
    Receives transaction data and returns fraud prediction.
    """
    # Check if model is loaded
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run model_train.py first."
        )
    
    try:
        # Encode the transaction type
        type_encoded = label_encoder.transform([transaction.types])[0]
        
        # Prepare features in the correct order for the model
        features = np.array([[
            transaction.step,
            type_encoded,
            transaction.amount,
            transaction.oldbalanceorig,
            transaction.newbalanceorig,
            transaction.oldbalancedest,
            transaction.newbalancedest,
            transaction.isflaggedfraud
        ]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        prediction_proba = model.predict_proba(features)[0]
        
        # Get probability of fraud (class 1)
        fraud_probability = float(prediction_proba[1])
        
        # Determine result message
        if prediction == 1:
            result = "Fraudulent Transaction"
            risk_level = "HIGH RISK"
        else:
            result = "Legitimate Transaction"
            risk_level = "LOW RISK"
        
        # Prepare detailed response
        response = {
            "prediction": result,
            "probability": fraud_probability,
            "details": {
                "risk_level": risk_level,
                "confidence": f"{max(prediction_proba) * 100:.2f}%",
                "transaction_type": transaction.types,
                "amount": transaction.amount,
                "flagged_by_system": bool(transaction.isflaggedfraud)
            }
        }
        
        print(f"Prediction made: {result} (probability: {fraud_probability:.4f})")
        
        return response
    
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.post("/predict-batch")
async def predict_batch(transactions: list[TransactionInput]):
    """
    Batch prediction endpoint for multiple transactions.
    """
    if model is None or label_encoder is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please run model_train.py first."
        )
    
    results = []
    for transaction in transactions:
        result = await predict_fraud(transaction)
        results.append(result)
    
    return {
        "total_transactions": len(transactions),
        "predictions": results
    }

def main():
    """
    Run the FastAPI application.
    """
    print("=" * 60)
    print("Starting Credit Card Fraud Detection API")
    print("=" * 60)
    print("\nAPI will be available at: http://127.0.0.1:8000")
    print("Interactive docs at: http://127.0.0.1:8000/docs")
    print("Alternative docs at: http://127.0.0.1:8000/redoc")
    print("\nPress CTRL+C to stop the server")
    print("=" * 60 + "\n")
    
    # Run the server
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,  # Auto-reload on code changes
        log_level="info"
    )

if __name__ == "__main__":
    main()