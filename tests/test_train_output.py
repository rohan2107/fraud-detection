# tests/test_train_output.py
import subprocess
import pickle
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

        with PKL_PATH.open("rb") as f:
            payload = pickle.load(f)

        # payload could be either a tuple (scaler, model) or dict depending on your train_model implementation.
        # Support both patterns for compatibility:
        if isinstance(payload, tuple):
            scaler, model = payload
            assert scaler is not None
            assert model is not None
        elif isinstance(payload, dict):
            assert "scaler" in payload and "model" in payload
        else:
            raise AssertionError("Unexpected payload format in fraud_model.pkl")
    finally:
        # cleanup
        if PKL_PATH.exists():
            PKL_PATH.unlink()
        if CSV_PATH.exists():
            CSV_PATH.unlink()
