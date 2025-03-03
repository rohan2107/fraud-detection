import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model & scaler
with open("fraud_model.pkl", "rb") as f:
    scaler, model = pickle.load(f)

class Transaction(BaseModel):
    time: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float
    amount: float

@app.get("/")
def home():
    return {"message": "Fraud Detection API is running!"}

@app.post("/predict")
def predict(transaction: Transaction):
    # Convert transaction data to DataFrame
    transaction_data = pd.DataFrame([transaction.dict().values()], columns=transaction.dict().keys())

    # Scale the input
    transaction_scaled = scaler.transform(transaction_data)

    # Predict with Isolation Forest
    prediction = model.predict(transaction_scaled)
    is_fraud = prediction[0] == -1  # -1 means anomaly (fraud)

    return {"prediction": "Fraud" if is_fraud else "Not Fraud"}
