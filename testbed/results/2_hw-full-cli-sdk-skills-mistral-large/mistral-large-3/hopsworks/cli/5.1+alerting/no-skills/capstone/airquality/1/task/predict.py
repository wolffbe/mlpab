#!/usr/bin/env python3
"""
Predict PM2.5 values for forecast_days.csv and store them in the feature table `airqpred2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Download the forecast data
dataset_api = project.get_dataset_api()
dataset_api.download("Resources/forecast_days.csv", overwrite=True)
df_forecast = pd.read_csv("forecast_days.csv", parse_dates=["date"])

# Load the model
mr = project.get_model_registry()
model_meta = mr.get_model("airqmodel2408fa", version=1)
model_dir = model_meta.download()
model = joblib.load(f"{model_dir}/model.pkl")

# Get the feature group for predictions
try:
    fg_pred = fs.get_feature_group("airqpred2408fa", version=1)
except:
    fg_pred = fs.create_feature_group(
        name="airqpred2408fa",
        version=1,
        description="Predictions for PM2.5 forecasting",
        primary_key=["date"],
        event_time="date",
        online_enabled=True
    )

# Predict and store
X_forecast = df_forecast.drop(columns=["date"])
df_forecast["pm25_pred"] = model.predict(X_forecast)

# Ingest predictions
fg_pred.insert(df_forecast[["date", "pm25_pred"]], write_options={"wait_for_job": True})

print("Predictions stored successfully in airqpred2408fa.")