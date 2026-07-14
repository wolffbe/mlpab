"""Full FTI pipeline for PM2.5 forecasting, running as a Hopsworks job.

Feature group: airq754fa9
Feature view / training dataset: airqtd754fa9
Model: airqmodel754fa9
Predictions feature group (online-enabled): airqpred754fa9
"""
import json
import math
import os

import joblib
import numpy as np
import pandas as pd
import hopsworks
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error

project = hopsworks.login()
fs = project.get_feature_store()
dataset_api = project.get_dataset_api()

# ---------- Load raw data from HopsFS ----------
hist_path = dataset_api.download("Resources/airq754fa9/airquality_history.csv", overwrite=True)
fcst_path = dataset_api.download("Resources/airq754fa9/forecast_days.csv", overwrite=True)
hist = pd.read_csv(hist_path)
fcst = pd.read_csv(fcst_path)
print(f"history rows={len(hist)} forecast rows={len(fcst)}")

# ---------- Feature engineering ----------
def add_calendar(df):
    d = pd.to_datetime(df["date"])
    doy = d.dt.dayofyear.astype(float)
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    df["month"] = d.dt.month.astype("int64")
    return df

hist = hist.sort_values("date").reset_index(drop=True)
hist = add_calendar(hist)
# lag / rolling air-quality signals (history only; use past values, no leakage)
hist["pm25_lag2"] = hist["pm25"].shift(2)
hist["pm25_roll3"] = hist["pm25"].shift(1).rolling(3).mean()
hist["pm25_roll7"] = hist["pm25"].shift(1).rolling(7).mean()

fcst = fcst.sort_values("date").reset_index(drop=True)
fcst = add_calendar(fcst)

# ---------- Feature group airq754fa9 ----------
fg = fs.get_or_create_feature_group(
    name="airq754fa9",
    version=1,
    primary_key=["date"],
    description="Daily air-quality history: weather + lag/rolling PM2.5 features",
    online_enabled=False,
)
fg.insert(hist, write_options={"wait_for_job": True})
print("feature group airq754fa9 populated")

# ---------- Feature view + training dataset airqtd754fa9 ----------
# Model features: everything available at forecast time (per-row signals).
MODEL_FEATURES = [
    "pm25_lag1", "temperature", "humidity", "wind_speed", "pressure",
    "precipitation", "doy_sin", "doy_cos", "month",
]
query = fg.select(["date"] + MODEL_FEATURES + ["pm25"])
fv = fs.get_or_create_feature_view(
    name="airqtd754fa9",
    version=1,
    query=query,
    labels=["pm25"],
    description="Training view for PM2.5 regressor airqmodel754fa9",
)
td_version, _ = fv.create_train_test_split(
    test_size=0.2,
    description="airqtd754fa9 train/test split",
    data_format="csv",
    write_options={"wait_for_job": True},
)
print(f"training dataset version {td_version} created")

X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)
for frame in (X_train, X_test):
    frame.drop(columns=[c for c in ("date",) if c in frame.columns], inplace=True)
X_train = X_train[MODEL_FEATURES]
X_test = X_test[MODEL_FEATURES]
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

# ---------- Train ----------
model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=4,
    l2_regularization=1.0, random_state=42,
)
model.fit(X_train, y_train)
pred_test = model.predict(X_test)
rmse = float(math.sqrt(mean_squared_error(y_test, pred_test)))
mae = float(np.mean(np.abs(y_test - pred_test)))
r2 = float(1 - np.sum((y_test - pred_test) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
print(f"held-out RMSE={rmse:.4f} MAE={mae:.4f} R2={r2:.4f}")

# Refit on all rows for the final forecaster (metrics above are the held-out eval).
final_model = HistGradientBoostingRegressor(
    max_iter=500, learning_rate=0.05, max_depth=4,
    l2_regularization=1.0, random_state=42,
)
full = hist.dropna(subset=["pm25"])
final_model.fit(full[MODEL_FEATURES], full["pm25"].values)

# ---------- Register model airqmodel754fa9 ----------
model_dir = "airqmodel754fa9_artifact"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(final_model, os.path.join(model_dir, "model.joblib"))
with open(os.path.join(model_dir, "features.json"), "w") as f:
    json.dump(MODEL_FEATURES, f)

mr = project.get_model_registry()
hw_model = mr.python.create_model(
    name="airqmodel754fa9",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="HistGradientBoosting PM2.5 regressor trained on airqtd754fa9",
    input_example=full[MODEL_FEATURES].iloc[:1],
    feature_view=fv,
    training_dataset_version=td_version,
)
hw_model.save(model_dir)
print("model airqmodel754fa9 registered")

# ---------- Batch inference into airqpred754fa9 ----------
fcst["pm25_pred"] = final_model.predict(fcst[MODEL_FEATURES]).astype(float)
pred_df = fcst[["date", "pm25_pred"]].copy()
pred_df["date"] = pred_df["date"].astype(str)

pred_fg = fs.get_or_create_feature_group(
    name="airqpred754fa9",
    version=1,
    primary_key=["date"],
    description="PM2.5 predictions for forecast days (online-enabled)",
    online_enabled=True,
)
pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print(f"predictions written: {len(pred_df)} rows")
print(pred_df.head().to_string())
print("PIPELINE_OK rmse=%.4f" % rmse)
