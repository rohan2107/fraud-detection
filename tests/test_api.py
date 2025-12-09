# tests/test_api.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.main import app

BASE_DIR = Path(__file__).parent / "samples"


def load_sample(name: str):
    with open(BASE_DIR / f"{name}.json") as f:
        return json.load(f)


@pytest.fixture
def client():
    # Create TestClient inside fixture so conftest (which may create fraud_model.pkl)
    # runs first during collection/setup.
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/")
    assert r.status_code == 200


def test_predict_normal(client):
    payload = load_sample("sample_normal")
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("prediction") == "Not Fraud"


def test_predict_fraud(client):
    payload = load_sample("sample_fraud")
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body.get("prediction") == "Fraud"
