# tests/test_metrics_endpoint.py
from fastapi.testclient import TestClient
from src.main import app
import pytest

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_metrics(client):
    r = client.get("/metrics")
    # conftest ensures a model exists in test runs; metrics should be present
    assert r.status_code == 200
    body = r.json()
    assert "model_meta" in body
