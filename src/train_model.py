"""Train and persist a fraud detection model.

This module now exposes a `main()` function and only runs when executed
as a script. It uses `logging` instead of printing directly so it is
friendlier to CI and importers.
"""

import sys
from pathlib import Path
import logging


import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import settings  # noqa: E402
from src.model_io import save_model  # noqa: E402 (import after adjusting sys.path)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    # Validate data file exists
    if not settings.data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {settings.data_path}. "
            f"Please ensure the dataset exists at this path."
        )

    df = pd.read_csv(settings.data_path)

    # Features (all except Class)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Save feature names so inference uses exact ordering
    feature_names = list(X.columns)

    # Train/test split to get a quick evaluation (we treat '1' as fraud)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Normalize
    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Train IsolationForest on normal transactions only (or on full as one strategy)
    model = IsolationForest(contamination=0.01, random_state=42).fit(X_train_s)

    # Predict on test set (IsolationForest: -1 anomaly, 1 normal)
    pred_test = model.predict(X_test_s)
    # convert to label space used by dataset: model -1 -> predicted fraud (1); model 1 -> predicted not fraud (0)
    pred_labels = (pred_test == -1).astype(int)

    precision = precision_score(y_test, pred_labels, zero_division=0)
    recall = recall_score(y_test, pred_labels, zero_division=0)
    f1 = f1_score(y_test, pred_labels, zero_division=0)

    meta = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}

    # Save everything using our model_io helper (joblib under the hood)
    payload = {
        "scaler": scaler,
        "model": model,
        "feature_names": feature_names,
        "meta": meta,
    }
    save_model(Path(settings.model_path), payload)

    logging.info("Model trained and saved to %s", Path(settings.model_path))
    logging.info("Eval metrics: %s", meta)


if __name__ == "__main__":
    main()
