"""
Credit Card Fraud Detection - Model Training Script
This script creates and trains a Random Forest model for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import os

def create_synthetic_dataset(n_samples=100000):
    """
    Creates a synthetic credit card transaction dataset similar to PaySim.
    In production, replace this with actual dataset loading.
    """
    print("Generating synthetic dataset...")
    
    np.random.seed(42)
    
    # Generate transaction data
    data = {
        'step': np.random.randint(1, 744, n_samples),
        'type': np.random.choice(['PAYMENT', 'TRANSFER', 'CASH_OUT', 'DEBIT', 'CASH_IN'], n_samples),
        'amount': np.random.exponential(scale=100000, size=n_samples),
        'oldbalanceOrig': np.random.exponential(scale=200000, size=n_samples),
        'newbalanceOrig': np.random.exponential(scale=200000, size=n_samples),
        'oldbalanceDest': np.random.exponential(scale=200000, size=n_samples),
        'newbalanceDest': np.random.exponential(scale=200000, size=n_samples),
        'isFlaggedFraud': np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
    }
    
    df = pd.DataFrame(data)
    
    # Create fraud labels (realistic distribution: ~0.13% fraud)
    fraud_probability = np.where(
        (df['type'].isin(['TRANSFER', 'CASH_OUT'])) & 
        (df['amount'] > 200000),
        0.05,  # Higher fraud probability for large transfers
        0.001  # Lower probability for other transactions
    )
    
    df['isFraud'] = np.random.binomial(1, fraud_probability)
    
    print(f"Dataset created with {len(df)} transactions")
    print(f"Fraud cases: {df['isFraud'].sum()} ({df['isFraud'].mean()*100:.2f}%)")
    
    return df

def preprocess_data(df):
    """
    Preprocesses the dataset for model training.
    """
    print("\nPreprocessing data...")
    
    # Encode categorical 'type' column
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    # Select features for training
    feature_columns = [
        'step', 'type_encoded', 'amount', 'oldbalanceOrig', 
        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
    ]
    
    X = df[feature_columns]
    y = df['isFraud']
    
    # Save the label encoder for later use
    joblib.dump(le, 'label_encoder.joblib')
    print("Label encoder saved as 'label_encoder.joblib'")
    
    return X, y, le

def train_model(X, y):
    """
    Trains a Random Forest Classifier for fraud detection.
    """
    print("\nSplitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")
    
    print("\nTraining Random Forest model...")
    # Using class_weight='balanced' to handle imbalanced dataset
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    model.fit(X_train, y_train)
    print("Model training completed!")
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    y_pred = model.predict(X_test)
    
    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Feature importance
    feature_names = [
        'step', 'type', 'amount', 'oldbalanceOrig', 
        'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
    ]
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return model

def main():
    """
    Main function to orchestrate the training pipeline.
    """
    print("=" * 60)
    print("Credit Card Fraud Detection - Model Training")
    print("=" * 60)
    
    # Create or load dataset
    # To use your own dataset, replace the next line with:
    # df = pd.read_csv('your_dataset.csv')
    df = create_synthetic_dataset(n_samples=100000)
    
    # Preprocess data
    X, y, label_encoder = preprocess_data(df)
    
    # Train model
    model = train_model(X, y)
    
    # Save the trained model
    print("\nSaving model...")
    joblib.dump(model, 'model.joblib')
    print("Model saved as 'model.joblib'")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - model.joblib (trained model)")
    print("  - label_encoder.joblib (for encoding transaction types)")
    print("\nYou can now run the FastAPI backend using:")
    print("  python app.py")

if __name__ == "__main__":
    main()