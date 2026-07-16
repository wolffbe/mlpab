"""Training script that runs as a Hopsworks Job on the platform."""
import hopsworks
import pandas as pd
import numpy as np
import joblib
import json
import os
import tempfile

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

fv_name = "airqfv3c8c0c"
fv = fs.get_feature_view(fv_name, version=1)

# Get latest training dataset
try:
    X_train, X_test, y_train, y_test = fv.get_train_test_split(training_dataset_version=1)
except Exception as e:
    print(f"get_train_test_split v1 failed: {e}")
    X_train, y_train = fv.get_training_data(training_dataset_version=1)
    split = int(len(X_train) * 0.8)
    X_test, y_test = X_train.iloc[split:].copy(), y_train.iloc[split:].copy()
    X_train, y_train = X_train.iloc[:split].copy(), y_train.iloc[:split].copy()

drop_cols = ["date"]
for col in drop_cols:
    if col in X_train.columns:
        X_train = X_train.drop(columns=[col])
        X_test = X_test.drop(columns=[col])

feature_names = list(X_train.columns)
print(f"Training with features: {feature_names}")
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

model = GradientBoostingRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    min_samples_leaf=5, subsample=0.8, random_state=42,
)
y_train_arr = y_train.values.ravel() if hasattr(y_train, 'values') else np.array(y_train).ravel()
y_test_arr = y_test.values.ravel() if hasattr(y_test, 'values') else np.array(y_test).ravel()

model.fit(X_train, y_train_arr)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test_arr, y_pred)))
mae = float(np.mean(np.abs(y_test_arr - y_pred)))
r2 = float(1 - np.sum((y_test_arr - y_pred)**2) / np.sum((y_test_arr - np.mean(y_test_arr))**2))
print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")

model_dir = tempfile.mkdtemp()
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
with open(os.path.join(model_dir, "features.json"), "w") as f:
    json.dump(feature_names, f)

hw_model = mr.sklearn.create_model(
    name="airqmodel3c8c0c",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="GradientBoosting PM2.5 regressor",
)
hw_model.save(model_dir)
print("Model registered.")

# Now predict forecast days
fg = fs.get_feature_group("airq3c8c0c", version=1)
pred_fg = fs.get_feature_group("airqpred3c8c0c", version=1)

# Read forecast from pred_fg (we stored forecast features there)
pred_data = pred_fg.read()
pred_features = pred_data[[c for c in feature_names if c in pred_data.columns]].copy()
missing = [c for c in feature_names if c not in pred_data.columns]
if missing:
    print(f"Missing features in forecast: {missing}")

y_forecast = model.predict(pred_features)
# Build full update_df with all columns (schema must match)
update_df = pred_data.copy()
update_df["pm25_pred"] = y_forecast.astype(float)
print("Predictions:")
print(update_df[["date", "pm25_pred"]].head())

pred_fg.insert(update_df, write_options={"wait_for_job": True})
print("Predictions updated in feature group.")
print(f"Done! RMSE={rmse:.4f}")
