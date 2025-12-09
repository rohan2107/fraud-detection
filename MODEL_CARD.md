# Model Card — Fraud Detection IsolationForest

**Model name:** IsolationForest (unsupervised anomaly detection)  
**Version:** v0.1 (development)  
**Date trained:** 2025-09-26

## Overview
This repository contains a small demo model that uses scikit-learn's `IsolationForest` to flag potentially fraudulent credit-card transactions as anomalies.

The model is trained on the publicly available "Credit Card Fraud Detection" dataset (Kaggle), which contains anonymized PCA-transformed features (`V1`..`V28`) plus transaction `Time` and `Amount`. This demo trains an IsolationForest on the feature set (unsupervised) and treats -1 predictions as anomalies (possible fraud).

## Intended use
- Demonstration/prototyping of anomaly-detection model serving with FastAPI.
- Resume / interview demo to show end-to-end ML -> API flow.

## Not intended for production
- The current model and training pipeline have **not** been hardened for production usage.
- No access controls, rate limiting, monitoring, or model governance are present.

## Data
**Source:** Kaggle Credit Card Fraud Detection dataset.  
**Link:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

**Note:** The dataset is *not* included in this repo. See `data/README.md` for download instructions. Do not commit raw datasets or real user data to this repository.

## Training & evaluation
- Preprocessing: standard scaling (`StandardScaler`) applied to all features.
- Model: `IsolationForest` trained with `contamination=0.01`.
- Evaluation: basic precision/recall/F1 reported during training in `train_model.py` (example; these metrics are not production-grade).

## Limitations & risks
- High false positive rate — IsolationForest detects anomalies, not confirmed fraud.
- The model is unsupervised and requires careful thresholding and human review before action.
- Beware of using pickle files from untrusted sources — they may execute code if loaded.

## Model artifact safety

Model artifact files (pickles, joblib dumps) can execute arbitrary code during
deserialization if they are created by an attacker or come from an untrusted
source. For safety:

- Prefer artifacts published by your CI/CD with checksums or signatures.
- Consider storing models in an artifact registry with immutability and access
	controls.
- Use safer interchange formats for production where possible (ONNX, TF
	SavedModel).


## Next steps / improvements
- Use a supervised model or a hybrid approach with labeled fraud examples for better precision/recall.
- Add monitoring for model drift, acceptance tests, and retraining pipelines.
- Add authentication, rate limiting, and logging/metrics for production readiness.

