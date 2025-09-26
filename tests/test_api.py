# tests/test_api.py
import pickle
import pandas as pd
from fastapi.testclient import TestClient
from main import app, FEATURE_NAMES

client = TestClient(app)

def test_health():
    r = client.get("/")
    assert r.status_code == 200
    assert "Fraud Detection API" in r.json().get("message", "")

def test_predict_dummy():
    # load model to get feature names and sample values
    with open("fraud_model.pkl", "rb") as f:
        payload = pickle.load(f)
    fnames = payload["feature_names"]
    # build a dummy input: zeros or mean
    sample = {k: 0.0 for k in fnames}
    # send request
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    assert "prediction" in r.json()
