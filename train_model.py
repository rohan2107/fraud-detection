import pandas as pd
import pickle
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("creditcard.csv")

# Drop target column for training
X = df.drop(columns=["Class"])  # 'y' is unused

# Normalize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(X_scaled)

# Save the model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump((scaler, model), f)

print("✅ Model trained and saved as fraud_model.pkl")
