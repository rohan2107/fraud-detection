# Fraud Detection API

Simple FastAPI service that uses an IsolationForest model to flag anomalous (potentially fraudulent) transactions.

## What’s included
- `train_model.py` — trains a model on `data/creditcard.csv` and writes `fraud_model.pkl`.
- `main.py` — FastAPI app with `/predict` endpoint.
- `tests/` — basic tests using pytest + TestClient.

## Run locally (using Python)
```bash
python -m venv .venv
# PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Train
python train_model.py

# Run API
uvicorn main:app --reload --port 8000
