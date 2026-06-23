"""Inference job — runs on Hopsworks platform.

Reads forecast features from the 'airqforecast3c8c0c' feature group,
loads the registered model, predicts pm25, stores in 'airqpred3c8c0c'.
"""
import hopsworks
import pandas as pd
import numpy as np
import joblib
import json
import os

project = hopsworks.login()
fs = project.get_feature_store()

# Read forecast features from the feature group (uploaded by the orchestrator)
forecast_fg = fs.get_feature_group(name="airqforecast3c8c0c", version=1)
forecast_df = forecast_fg.read()
print("Forecast rows:", len(forecast_df))
print(forecast_df.dtypes)

# Load model
mr = project.get_model_registry()
model_meta = mr.get_model(name="airqmodel3c8c0c", version=1)
model_dir  = model_meta.download()

model        = joblib.load(os.path.join(model_dir, "model.pkl"))
feature_cols = json.load(open(os.path.join(model_dir, "feature_cols.json")))
print("Feature cols:", feature_cols)

X_forecast = forecast_df[feature_cols]
pm25_pred  = model.predict(X_forecast)

pred_df = pd.DataFrame({
    "date":      forecast_df["date"].values,
    "pm25_pred": pm25_pred.astype(float),
})
print(pred_df.head(10))

# Store predictions (online-enabled for low-latency lookup)
pred_fg = fs.get_or_create_feature_group(
    name="airqpred3c8c0c",
    version=1,
    primary_key=["date"],
    description="PM2.5 forecast predictions",
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("Predictions stored in airqpred3c8c0c.")
