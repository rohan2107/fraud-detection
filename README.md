# Fraud Detection API

Simple FastAPI service that uses an IsolationForest model to flag anomalous (potentially fraudulent) transactions.

## What’s included
- `src/train_model.py` — trains a model on `data/creditcard.csv` and writes `fraud_model.pkl`.
- `src/main.py` — FastAPI app with `/predict` endpoint.
- `tests/` — basic tests using pytest + TestClient.

## Run locally
Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

# Train the model (requires data/creditcard.csv):
```bash
python src/train_model.py
```
The dataset is not stored in this repo.
Download it from Kaggle Credit Card Fraud Dataset
and place creditcard.csv inside the data/ folder.

# Run API
```bash
uvicorn main:app --reload --port 8000
```

## Usage
Example request (using included sample JSONs):

Predict a **normal** transaction:
```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "@tests/samples/sample_normal.json"
```

Predict a **fraudulent** transaction:
```bash
curl.exe -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "@tests/samples/sample_fraud.json"
```

## Run tests

Run automated tests with pytest:
```bash
pytest -q
```