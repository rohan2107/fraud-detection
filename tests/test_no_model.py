# tests/test_no_model.py
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# We need to MUTATE the app state for this test, so we import the app factory
# and override dependencies carefully.
from src.main import app

# Get paths BEFORE changing directory
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "fraud_model.pkl"
SAMPLE_PATH = ROOT / "tests" / "samples" / "sample_normal.json"


@pytest.fixture
def client_no_model():
    """
    Test client fixture where we guarantee no model file exists.
    This allows testing the API's startup and error-handling
    when the model is not available.
    """
    # Stash the real model if it exists
    model_existed = False
    stashed_model = None
    if MODEL_PATH.exists():
        model_existed = True
        stashed_model = MODEL_PATH.read_bytes()
        MODEL_PATH.unlink()

    # Create a TestClient instance
    with TestClient(app) as client:
        yield client

    # Restore the original model
    if model_existed and stashed_model:
        MODEL_PATH.write_bytes(stashed_model)


def test_predict_returns_503_if_model_missing(
    client_no_model,
):
    # This test uses the `client_no_model` fixture which ensures
    # the model does not exist.
    with open(SAMPLE_PATH) as f:
        payload = f.read()

    r = client_no_model.post("/predict", data=payload)
    assert r.status_code == 503
    assert "Model not loaded" in r.json()["detail"]


def test_metrics_returns_404_if_model_missing(
    client_no_model,
):
    r = client_no_model.get("/metrics")
    assert r.status_code == 404
    assert "Model metadata not available" in r.json()["detail"]
