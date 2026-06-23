"""Update predictions in airqpred3c8c0c using the registered model."""
import hopsworks
import pandas as pd
import numpy as np
import joblib
import json
import os

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load model
model_obj = mr.get_model("airqmodel3c8c0c", version=1)
model_dir = model_obj.download()
model = joblib.load(os.path.join(model_dir, "model.pkl"))
with open(os.path.join(model_dir, "features.json")) as f:
    feature_names = json.load(f)
print(f"Model loaded. Features: {feature_names}")

# Read forecast data from predictions FG
pred_fg = fs.get_feature_group("airqpred3c8c0c", version=1)
pred_data = pred_fg.read()
print(f"Forecast data shape: {pred_data.shape}")
print(f"Forecast data columns: {list(pred_data.columns)}")

# Build feature matrix
missing = [c for c in feature_names if c not in pred_data.columns]
print(f"Missing features: {missing}")
pred_features = pred_data[[c for c in feature_names if c in pred_data.columns]].copy()

y_forecast = model.predict(pred_features)
print(f"Predictions shape: {y_forecast.shape}")

# Update with all columns
update_df = pred_data.copy()
update_df["pm25_pred"] = y_forecast.astype(float)
print("Predictions preview:")
print(update_df[["date", "pm25_pred"]].head(10))

pred_fg.insert(update_df, write_options={"wait_for_job": True})
print("Predictions inserted successfully!")
