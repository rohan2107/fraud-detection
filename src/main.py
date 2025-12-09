# src/main.py
import logging
from contextlib import asynccontextmanager
from typing import Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import settings
from .model_io import load_model

LOG = logging.getLogger("fraud-api")
LOG.setLevel(settings.log_level.upper())
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
LOG.addHandler(handler)

MODEL_PATH = settings.model_path

# global model objects (populated during lifespan startup)
scaler = None
model = None
FEATURE_NAMES: Optional[list] = None
MODEL_META: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager — runs at startup and shutdown.
    Loads model if available; does not crash the app if missing.
    """
    global scaler, model, FEATURE_NAMES, MODEL_META
    try:
        if MODEL_PATH.exists():
            LOG.info("Loading model from %s", MODEL_PATH)
            payload = load_model(MODEL_PATH)
            scaler = payload.get("scaler")
            model = payload.get("model")
            FEATURE_NAMES = payload.get("feature_names")
            MODEL_META = payload.get("meta", {})
            LOG.info("Model loaded successfully.")
        else:
            LOG.warning(
                f"{MODEL_PATH} not found — API will start without a model. Add the pickle to enable /predict."
            )
    except Exception as exc:  # keep startup resilient but log error
        LOG.exception("Failed to load model at startup: %s", exc)
    yield
    # optional: any shutdown cleanup here
    LOG.info("Shutting down app.")


app = FastAPI(title="Fraud Detection API", lifespan=lifespan)


# --- Pydantic model using CSV-style names as aliases ---
class Transaction(BaseModel):
    Time: float = Field(..., alias="Time")
    V1: float = Field(..., alias="V1")
    V2: float = Field(..., alias="V2")
    V3: float = Field(..., alias="V3")
    V4: float = Field(..., alias="V4")
    V5: float = Field(..., alias="V5")
    V6: float = Field(..., alias="V6")
    V7: float = Field(..., alias="V7")
    V8: float = Field(..., alias="V8")
    V9: float = Field(..., alias="V9")
    V10: float = Field(..., alias="V10")
    V11: float = Field(..., alias="V11")
    V12: float = Field(..., alias="V12")
    V13: float = Field(..., alias="V13")
    V14: float = Field(..., alias="V14")
    V15: float = Field(..., alias="V15")
    V16: float = Field(..., alias="V16")
    V17: float = Field(..., alias="V17")
    V18: float = Field(..., alias="V18")
    V19: float = Field(..., alias="V19")
    V20: float = Field(..., alias="V20")
    V21: float = Field(..., alias="V21")
    V22: float = Field(..., alias="V22")
    V23: float = Field(..., alias="V23")
    V24: float = Field(..., alias="V24")
    V25: float = Field(..., alias="V25")
    V26: float = Field(..., alias="V26")
    V27: float = Field(..., alias="V27")
    V28: float = Field(..., alias="V28")
    Amount: float = Field(..., alias="Amount")

    # Pydantic v2 config
    model_config = ConfigDict(validate_by_name=True, extra="forbid")


@app.get("/")
def home():
    return {
        "message": "Fraud Detection API is running!",
        "model_loaded": model is not None,
        "model_meta_available": bool(MODEL_META),
    }


@app.get("/metrics")
def metrics():
    """Return model metadata (metrics) if available."""
    if not MODEL_META:
        raise HTTPException(status_code=404, detail="Model metadata not available.")
    return {"model_meta": MODEL_META}


@app.post("/predict")
def predict(transaction: Transaction):
    """
    Predict whether a single transaction is fraudulent.
    Returns 503 if model not loaded.
    """
    if model is None or scaler is None or FEATURE_NAMES is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure fraud_model.pkl exists in the working directory.",
        )

    # Use CSV-style alias keys (Time, V1..V28, Amount)
    data = transaction.model_dump(by_alias=True)

    # Ensure all features expected by the trained model are present
    missing = [c for c in FEATURE_NAMES if c not in data]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing features: {missing}")

    # Build single-row DataFrame in the exact order
    row = [data[c] for c in FEATURE_NAMES]
    transaction_data = pd.DataFrame([row], columns=FEATURE_NAMES)

    try:
        transaction_scaled = scaler.transform(transaction_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Scaler transform failed: {e}")

    try:
        prediction = model.predict(transaction_scaled)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    is_fraud = int(prediction[0] == -1)

    return {
        "prediction": "Fraud" if is_fraud else "Not Fraud",
        "raw_prediction": int(prediction[0]),
        "model_meta": MODEL_META,
    }
