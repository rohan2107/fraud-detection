"""Model serialization helpers.

Provides a small abstraction over joblib/pickle to save and load model
artifacts reliably while remaining backward-compatible with older pickle
blobs created by tests or user workflows.
"""
from pathlib import Path
import logging
import joblib
import pickle
from typing import Any

LOG = logging.getLogger("fraud-api")


def save_model(path: Path, payload: Any) -> None:
    """Persist a model payload using joblib.

    Args:
        path: destination file path
        payload: object to serialize (dict with scaler/model/...)
    """
    joblib.dump(payload, path)
    LOG.info("Model saved to %s", path)


def load_model(path: Path) -> Any:
    """Load a model artifact. Try joblib first, fall back to pickle.

    This keeps the application resilient when tests or older artifacts were
    written with plain pickle.
    """
    try:
        return joblib.load(path)
    except Exception as exc:  # pragma: no cover - fallback path
        LOG.debug("joblib.load failed (%s), attempting pickle.load", exc)
        with open(path, "rb") as fh:
            return pickle.load(fh)
