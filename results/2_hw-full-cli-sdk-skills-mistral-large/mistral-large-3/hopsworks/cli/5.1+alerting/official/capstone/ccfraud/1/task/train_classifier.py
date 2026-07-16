#!/usr/bin/env python3
"""
Train a fraud classifier using the training dataset and register it as ccmodelc444ca.
"""

import hopsworks
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Read training dataset
td = fs.get_training_dataset("cctfv444ca_1", version=1)
df = td.read()

# Split into features and label
X = df.drop(columns=["is_fraud", "datetime", "transaction_id", "merchant", "category"])
y = df["is_fraud"]

# Train a RandomForest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X, y)

# Predict and evaluate
y_pred_proba = clf.predict_proba(X)[:, 1]
auc = roc_auc_score(y, y_pred_proba)
print(f"ROC AUC: {auc:.4f}")
print(classification_report(y, clf.predict(X)))

# Save model and metrics
joblib.dump(clf, "fraud_model.pkl")
metrics = {
    "roc_auc": auc,
}

# Register model
model = mr.python.create_model(
    name="ccmodelc444ca",
    description="Fraud detection classifier",
    metrics=metrics,
)
model.save("fraud_model.pkl")

print("Model trained and registered successfully.")