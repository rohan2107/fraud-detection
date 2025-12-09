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

Optional: install dev tools (ruff, black) and use the helper make targets:
```bash
pip install -r requirements-dev.txt
make lint
make test
```

Environment variables (see `env.example`):
- `FRAUD_MODEL_PATH` (default: `fraud_model.pkl`)
- `DATA_PATH` (default: `data/creditcard.csv`)
- `LOG_LEVEL` (default: `INFO`)

# Train the model (requires data/creditcard.csv):
```bash
python src/train_model.py
```
The dataset is not stored in this repo.
Download it from Kaggle Credit Card Fraud Dataset
and place creditcard.csv inside the data/ folder.

# Run API
```bash
uvicorn src.main:app --reload --port 8000
```

## Run with Docker
Build and run the API locally:
```bash
docker build -t fraud-api .
docker run --rm -p 8000:8000 --env-file env.example fraud-api
```
The container expects `fraud_model.pkl` to exist at the repo root when building (or mount one at runtime with `-v $(pwd)/fraud_model.pkl:/app/fraud_model.pkl`).

## Model artifact safety

Be careful when loading serialized model artifacts (pickle / joblib files). These
files may execute arbitrary code when deserialized if they come from an
untrusted source. For local development this repository provides test fixtures
and a helper that attempts `joblib` loading first and falls back to `pickle`.

Recommendations:
- Verify artifacts using a checksum (e.g. SHA256) or a signed release before
  loading in production.
- Only load artifacts from trusted sources or CI-published build artifacts.
- In production, prefer model formats with safer runtimes (ONNX, TF SavedModel,
  etc.) or use isolated environments for deserialization.

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