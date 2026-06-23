import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Read training dataset
td_df = fs.get_training_dataset("cctdc444ca", version=1).read()

# Prepare features and target
feature_columns = [
    "transaction_velocity_1h", "transaction_velocity_24h", 
    "amount_mean_24h", "amount_std_24h", 
    "geo_distance", "time_since_last_tx", "amount"
]
X = td_df[feature_columns]
y = td_df["is_fraud"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
model.fit(X_train, y_train)

# Evaluate model
y_pred_proba = model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC: {roc_auc}")
print(classification_report(y_test, model.predict(X_test)))

# Register model
mr = project.get_model_registry()
model_dir = "ccmodelc444ca"
os.makedirs(model_dir, exist_ok=True)

# Save model and metrics
joblib.dump(model, f"{model_dir}/model.pkl")
with open(f"{model_dir}/metrics.json", "w") as f:
    f.write(f"{{\"roc_auc\": {roc_auc}}}")

# Register model
cc_model = mr.sklearn.create_model(
    name="ccmodelc444ca",
    version=1,
    metrics={"roc_auc": roc_auc},
    description="Fraud detection model for credit-card transactions",
    model_schema={
        "input_schema": X_train.dtypes.to_dict(),
        "output_schema": {"fraud_probability": "float64"},
    },
)
cc_model.save(model_dir)