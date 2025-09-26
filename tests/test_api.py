# tests/test_api.py
import json
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def load_sample(name):
    with open(f"tests/{name}.json") as f:
        return json.load(f)

def test_health():
    r = client.get("/")
    assert r.status_code == 200

def test_predict_normal():
    payload = load_sample("sample_normal")
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()

def test_predict_fraud():
    payload = load_sample("sample_fraud")
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert "prediction" in r.json()
