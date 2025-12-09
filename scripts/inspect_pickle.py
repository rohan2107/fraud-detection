import sys
import subprocess
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CSV_PATH = DATA_DIR / "creditcard.csv"
PKL_PATH = ROOT / "fraud_model.pkl"

DATA_DIR.mkdir(exist_ok=True)
cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
rng = np.random.RandomState(0)
arr = rng.normal(size=(50, len(cols)))
arr[:, -1] = 0
arr[:2, -1] = 1

df = pd.DataFrame(arr, columns=cols)
df.to_csv(CSV_PATH, index=False)

print("Running training subprocess...")
subprocess.run([sys.executable, "src/train_model.py"], check=True, cwd=ROOT)
print("Subprocess done. PKL exists?", PKL_PATH.exists())

with PKL_PATH.open("rb") as f:
    payload = pickle.load(f)

print("Payload type:", type(payload))
if isinstance(payload, dict):
    scaler = payload.get("scaler")
    model = payload.get("model")
else:
    scaler, model = payload

print("scaler type:", type(scaler))
print("scaler class module:", getattr(scaler.__class__, "__module__", None))
print("model type:", type(model))
print("model class module:", getattr(model.__class__, "__module__", None))

# cleanup
if PKL_PATH.exists():
    print("Removing PKL")
    PKL_PATH.unlink()
if CSV_PATH.exists():
    CSV_PATH.unlink()
print("Done")
