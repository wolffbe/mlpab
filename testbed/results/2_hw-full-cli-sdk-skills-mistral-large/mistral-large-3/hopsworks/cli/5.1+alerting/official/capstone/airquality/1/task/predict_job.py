#!/usr/bin/env python3
"""
Job to predict PM2.5 for forecast_days.csv and store results in `airqpred2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load model
model_registry = mr.get_model("airqmodel2408fa", version=1)
model_dir = model_registry.download()
model = joblib.load(f"{model_dir}/airqmodel2408fa.pkl")

# Load forecast data
forecast_df = pd.read_csv("/Resources/forecast_days.csv", parse_dates=["date"])

# Sort by date
forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

# Lag features (use last known values from history)
history_df = pd.read_csv("/Resources/airquality_history.csv", parse_dates=["date"])
last_pm25 = history_df["pm25"].iloc[-1]
last_pm25_lag1 = history_df["pm25"].iloc[-2]
last_pm25_lag2 = history_df["pm25"].iloc[-3]

forecast_df["pm25_lag2"] = forecast_df["pm25_lag1"].shift(1).fillna(last_pm25_lag1)
forecast_df["pm25_lag3"] = forecast_df["pm25_lag1"].shift(2).fillna(last_pm25_lag2)

# Rolling features (simplified for forecast)
forecast_df["pm25_rolling_3d"] = forecast_df["pm25_lag1"].rolling(3, min_periods=1).mean()
forecast_df["pm25_rolling_7d"] = forecast_df["pm25_lag1"].rolling(7, min_periods=1).mean()

# Interaction feature
forecast_df["temp_humidity_interaction"] = forecast_df["temperature"] * forecast_df["humidity"]

# Features for prediction
X_forecast = forecast_df.drop(columns=["date"])

# Predict
forecast_df["pm25_pred"] = model.predict(X_forecast)

# Create feature group for predictions
fg_pred = fs.get_or_create_feature_group(
    name="airqpred2408fa",
    version=1,
    description="Predictions for PM2.5 forecasting",
    primary_key=["date"],
    online_enabled=True,
)

# Insert predictions
fg_pred.insert(forecast_df[["date", "pm25_pred"]], write_options={"wait_for_job": True})