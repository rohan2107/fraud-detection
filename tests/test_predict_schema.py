# tests/test_predict_schema.py
import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from src.main import app

BASE_DIR = Path(__file__).parent / "samples"

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def load_sample(name: str):
    with open(BASE_DIR / f"{name}.json") as f:
        return json.load(f)

def test_predict_schema(client):
    payload = load_sample("sample_normal")
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert "raw_prediction" in body
    assert "model_meta" in body
