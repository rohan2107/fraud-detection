# tests/test_train_output.py
import subprocess
import joblib
from pathlib import Path
import pandas as pd
import numpy as np
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"
PKL_PATH = ROOT / "fraud_model.pkl"


def make_small_csv():
    DATA_DIR.mkdir(exist_ok=True)
    # columns: Time, V1..V28, Amount, Class
    cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
    # create 50 small rows
    rng = np.random.RandomState(0)
    arr = rng.normal(size=(50, len(cols)))
    # make Class mostly zeros; few ones
    arr[:, -1] = 0
    arr[:2, -1] = 1  # two fraud labels
    df = pd.DataFrame(arr, columns=cols)
    df.to_csv(CSV_PATH, index=False)


def test_train_creates_pickle_and_contents():
    # cleanup from earlier runs if present
    if PKL_PATH.exists():
        PKL_PATH.unlink()
    if CSV_PATH.exists():
        CSV_PATH.unlink()

    try:
        make_small_csv()
        # run training script (using the same python interpreter)
        subprocess.run([os.sys.executable, "src/train_model.py"], check=True, cwd=ROOT)

        assert PKL_PATH.exists(), "fraud_model.pkl should be created by train_model.py"

        payload = joblib.load(PKL_PATH)

        assert isinstance(payload, dict)
        assert "scaler" in payload and "model" in payload and "meta" in payload

        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest

        assert isinstance(payload["scaler"], StandardScaler)
        assert isinstance(payload["model"], IsolationForest)
        assert "precision" in payload["meta"]
        assert "recall" in payload["meta"]
        assert "f1" in payload["meta"]

    finally:
        # cleanup
        if PKL_PATH.exists():
            PKL_PATH.unlink()
        if CSV_PATH.exists():
            CSV_PATH.unlink()
