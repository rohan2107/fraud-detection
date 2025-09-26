# train_model.py
import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("data/creditcard.csv")

# Features (all except Class)
X = df.drop(columns=["Class"])
y = df["Class"]

# Save feature names so inference uses exact ordering
feature_names = list(X.columns)

# Train/test split to get a quick evaluation (we treat '1' as fraud)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

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

# Save everything
payload = {"scaler": scaler, "model": model, "feature_names": feature_names, "meta": meta}
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(payload, f)

print("Model trained and saved as fraud_model.pkl")
print("Eval metrics:", meta)
