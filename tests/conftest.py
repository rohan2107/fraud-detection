# tests/conftest.py
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Insert project root (one level above tests/) to sys.path so "import src" works
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PKL_PATH = ROOT / "fraud_model.pkl"
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]


def _write_synthetic_model():
    rng = np.random.RandomState(42)
    df = pd.DataFrame(rng.normal(size=(200, len(FEATURE_NAMES))), columns=FEATURE_NAMES)
    scaler = StandardScaler().fit(df)
    Xs = scaler.transform(df)
    model = IsolationForest(contamination=0.01, random_state=42).fit(Xs)
    payload = {
        "scaler": scaler,
        "model": model,
        "feature_names": FEATURE_NAMES,
        "meta": {"note": "synthetic model for tests"},
    }
    PKL_PATH.write_bytes(pickle.dumps(payload))


@pytest.fixture(autouse=True)
def ensure_model_available():
    """
    Ensure fraud_model.pkl exists for API tests.
    Recreates if a test removed it.
    """
    if not PKL_PATH.exists():
        _write_synthetic_model()
