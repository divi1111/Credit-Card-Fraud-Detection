# FraudGuard – Financial Transaction Fraud Detection System

An end-to-end machine learning application that detects potentially fraudulent financial transactions using a Random Forest classifier, with a FastAPI backend and an interactive web dashboard.

## Overview

FraudGuard is a machine learning-based fraud detection system designed to identify suspicious financial transactions and help users assess transaction risk.

The system uses transaction-related features to generate fraud predictions and a fraud score. Users can enter transaction details through a web dashboard and receive predictions from the deployed machine learning model through a REST API.

## Key Features

- **Machine Learning:** Random Forest classification for fraud detection.
- **Data Preprocessing:** Transaction cleaning, feature engineering, encoding, and class balancing through legitimate transaction sampling.
- **Fraud Scoring:** Generates a fraud score and classifies transactions using a validation-selected threshold.
- **REST API:** FastAPI backend for prediction and health-check endpoints.
- **Interactive Dashboard:** Web-based interface for submitting transactions and viewing prediction results.
- **Input Validation:** Validates transaction details before making predictions.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn, Random Forest |
| Data Processing | Pandas, NumPy |
| Model Persistence | Joblib |
| Backend | FastAPI, Uvicorn |
| Frontend | HTML, CSS, JavaScript |
| Dataset | PaySim Synthetic Financial Dataset |
| Deployment | Render |

## Dataset

This project uses the PaySim synthetic financial transaction dataset.

The original dataset contains approximately 6.36 million transactions and includes features such as:

- Transaction step
- Transaction type
- Transaction amount
- Origin and destination account identifiers
- Account balance information
- Fraud labels

For the current training pipeline, all 8,213 fraudulent transactions are retained, along with a sampled subset of legitimate transactions, producing a processed dataset of 500,000 rows.

The processed dataset is used for model training and evaluation.

**Note:** PaySim is a synthetic mobile-money transaction dataset, not a real credit-card transaction dataset.

## Machine Learning Workflow

1. Load and inspect the transaction dataset.
2. Handle data preparation and feature engineering.
3. Encode categorical transaction features.
4. Retain fraudulent transactions and sample legitimate transactions.
5. Split the processed dataset into training, validation, and testing sets.
6. Train a Random Forest classifier.
7. Select a prediction threshold using validation data.
8. Evaluate the model using precision, recall, F1-score, ROC-AUC, and PR-AUC.
9. Save the trained model for backend integration.

## Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── backend/
│   ├── app.py
│   ├── model_train.py
│   ├── prepare_data.py
│   ├── requirements.txt
│   └── model.joblib
│
├── frontend/
│   ├── index.html
│   ├── styles.css
│   └── script.js
│
├── data/
│   ├── raw/
│   └── processed/
│
├── .gitignore
├── .python-version
└── README.md
```

## Running the Project Locally

### 1. Clone the repository

```bash
git clone https://github.com/divi1111/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install dependencies

```bash
cd backend
python -m pip install -r requirements.txt
```

### 3. Start the FastAPI backend

```bash
python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

### 4. Open the frontend

Open `frontend/index.html` in your browser.

### 5. Explore the API

Visit:

```text
http://127.0.0.1:8000/docs
```

Use the interactive Swagger documentation to test the prediction endpoint.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API information |
| GET | `/health` | Backend health check |
| POST | `/predict` | Predict transaction fraud risk |

## Model Evaluation

The current model was evaluated on a stratified test split of the processed dataset.

| Metric | Result |
|---|---:|
| Precision | 62.92% |
| Recall | 47.24% |
| F1-score | 53.96% |
| ROC-AUC | 95.00% |
| PR-AUC | 54.80% |

**Evaluation note:** These results come from a dataset where legitimate transactions were downsampled. They should not be interpreted as real-world performance or as estimates of performance at the original fraud prevalence.

## Future Improvements

- Improve fraud detection recall while controlling false positives.
- Evaluate performance on data with the original transaction distribution.
- Add transaction monitoring and analytics.
- Improve model explainability.
- Implement more robust deployment and monitoring.

## Disclaimer

This project is developed for educational and demonstration purposes. It uses synthetic transaction data and should not be used as a production financial fraud prevention system without further validation, testing, and security improvements.
