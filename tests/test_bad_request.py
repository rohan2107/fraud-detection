# tests/test_bad_request.py
import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_empty_payload(client):
    # sending an empty body should produce a 422 Unprocessable Entity because required fields are missing
    r = client.post("/predict", json={})
    assert r.status_code == 422


def test_missing_fields(client):
    # V1 and V2 are missing
    payload = {"Time": 0.0, "V3": -1.35, "Amount": 149.62}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_extra_fields(client):
    # The default Pydantic model forbids extra fields
    payload = {
        "Time": 0.0,
        "V1": -1.35,
        "V2": -0.48,
        "V3": 1.76,
        "V4": -0.41,
        "V5": 0.52,
        "V6": -0.63,
        "V7": -0.01,
        "V8": 0.58,
        "V9": -0.23,
        "V10": -0.16,
        "V11": 0.38,
        "V12": -0.89,
        "V13": -0.44,
        "V14": -0.14,
        "V15": 1.25,
        "V16": -1.02,
        "V17": 0.04,
        "V18": -0.18,
        "V19": -0.08,
        "V20": 0.25,
        "V21": -0.01,
        "V22": 0.23,
        "V23": -0.11,
        "V24": 0.09,
        "V25": -0.42,
        "V26": 0.29,
        "V27": 0.1,
        "V28": 0.02,
        "Amount": 149.62,
        "ExtraField": "should-be-rejected",
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
