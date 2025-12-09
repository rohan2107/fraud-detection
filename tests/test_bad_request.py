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
