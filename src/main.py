# main.py
import pickle
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict

app = FastAPI(title="Fraud Detection API")

# Load model & scaler
try:
    with open("fraud_model.pkl", "rb") as f:
        payload = pickle.load(f)
        scaler = payload["scaler"]
        model = payload["model"]
        FEATURE_NAMES = payload["feature_names"]
        MODEL_META = payload.get("meta", {})
except FileNotFoundError:
    raise RuntimeError("fraud_model.pkl not found in working dir")
except Exception as e:
    raise RuntimeError(f"Failed to load model pickle: {e}")

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

    model_config = ConfigDict(validate_by_name=True, extra="forbid")

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Use alias keys (CSV-style) so data dict contains "Time","V1",...,"Amount"
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
        "model_meta": MODEL_META
    }