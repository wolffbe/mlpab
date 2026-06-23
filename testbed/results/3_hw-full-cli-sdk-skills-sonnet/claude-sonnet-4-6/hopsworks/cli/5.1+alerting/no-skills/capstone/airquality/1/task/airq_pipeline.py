"""
Full FTI pipeline for air quality PM2.5 forecasting.
Runs as a Hopsworks Python job.
"""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib
import json
import os

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# --- 1. Download data from HopsFS ---
dataset_api = project.get_dataset_api()
dataset_api.download("Resources/airq_data/airquality_history.csv", "/tmp/airquality_history.csv", overwrite=True)
dataset_api.download("Resources/airq_data/forecast_days.csv", "/tmp/forecast_days.csv", overwrite=True)

history_df = pd.read_csv("/tmp/airquality_history.csv", parse_dates=["date"])
forecast_df = pd.read_csv("/tmp/forecast_days.csv", parse_dates=["date"])

# Cast all numeric cols to float64 to match feature group schema (double)
for col in ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation", "pm25"]:
    if col in history_df.columns:
        history_df[col] = history_df[col].astype(float)
for col in ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]:
    if col in forecast_df.columns:
        forecast_df[col] = forecast_df[col].astype(float)

print(f"History rows: {len(history_df)}, Forecast rows: {len(forecast_df)}")

# --- 2. Insert data into feature group ---
fg = fs.get_feature_group("airq3c8c0c", version=1)
fg.insert(history_df, write_options={"wait_for_job": True})
print("Feature group insert done.")

# --- 3. Get or create feature view ---
feature_cols = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
fv = fs.get_feature_view("airqtd3c8c0c", version=1)
if fv is None:
    query = fg.select(feature_cols + ["pm25"])
    fv = fs.create_feature_view(
        name="airqtd3c8c0c",
        version=1,
        query=query,
        labels=["pm25"],
        description="Air quality PM2.5 feature view for training"
    )
    print("Created feature view.")
else:
    print("Feature view already exists.")

# --- 4. Create training dataset (always create new version) ---
td_version, job = fv.create_train_test_split(
    test_size=0.2,
    description="PM2.5 training dataset",
    data_format="csv",
    write_options={"wait_for_job": True}
)
print(f"Training dataset version: {td_version}")

# --- 5. Get training data ---
X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=td_version)
print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
print(f"Columns: {list(X_train.columns)}")

# Ensure feature_cols exist in X_train
available_features = [c for c in feature_cols if c in X_train.columns]
print(f"Using features: {available_features}")

y_train_arr = y_train.values.ravel() if hasattr(y_train, "values") else np.asarray(y_train).ravel()
y_test_arr = y_test.values.ravel() if hasattr(y_test, "values") else np.asarray(y_test).ravel()

# --- 6. Train model ---
model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train[available_features], y_train_arr)

y_pred_test = model.predict(X_test[available_features])
rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred_test)))
mae = float(np.mean(np.abs(y_test_arr - y_pred_test)))
r2 = float(1.0 - np.sum((y_test_arr - y_pred_test) ** 2) / np.sum((y_test_arr - float(np.mean(y_test_arr))) ** 2))

print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE:  {mae:.4f}")
print(f"Test R2:   {r2:.4f}")

# --- 7. Save and register model ---
model_dir = "/tmp/airq_model_dir"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
with open(os.path.join(model_dir, "feature_columns.json"), "w") as f:
    json.dump(available_features, f)

hw_model = mr.sklearn.create_model(
    name="airqmodel3c8c0c",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description=f"PM2.5 GBM regressor. RMSE={rmse:.4f}",
    feature_view=fv,
    training_dataset_version=td_version,
    input_example=X_train[available_features].iloc[0:1]
)
hw_model.save(model_dir)
print(f"Model registered: airqmodel3c8c0c version {hw_model.version}")

# --- 8. Create predictions feature group ---
pred_fg = fs.get_feature_group("airqpred3c8c0c", version=1)
if pred_fg is None:
    pred_fg = fs.create_feature_group(
        name="airqpred3c8c0c",
        version=1,
        primary_key=["date"],
        event_time="date",
        online_enabled=True,
        description="PM2.5 model predictions"
    )
    print("Created predictions feature group.")
else:
    print("Predictions feature group already exists.")

# --- 9. Make predictions and insert ---
X_forecast = forecast_df[available_features].copy()
pm25_preds = model.predict(X_forecast)

pred_df = pd.DataFrame({
    "date": forecast_df["date"],
    "pm25_pred": pm25_preds.astype(float)
})
print(f"Predictions:\n{pred_df.to_string()}")

pred_fg.insert(pred_df, write_options={"wait_for_job": True})
print("Predictions inserted into airqpred3c8c0c.")
print("Pipeline complete!")
