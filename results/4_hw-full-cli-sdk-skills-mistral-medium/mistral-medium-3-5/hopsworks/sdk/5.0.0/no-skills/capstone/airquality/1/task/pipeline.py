#!/usr/bin/env python3
"""
Full FTI pipeline for air quality PM2.5 forecasting on Hopsworks platform.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Connect to Hopsworks
import hopsworks

print("Connecting to Hopsworks...")
hopsworks.login()
fs = hopsworks.feature_store()

# Read the data
print("Reading data...")
history_df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])
forecast_df = pd.read_csv("data/forecast_days.csv", parse_dates=["date"])

print(f"History shape: {history_df.shape}")
print(f"Forecast shape: {forecast_df.shape}")
print(f"\nHistory columns: {history_df.columns.tolist()}")
print(f"Forecast columns: {forecast_df.columns.tolist()}")

# Feature Engineering
# We have: date, pm25_lag1, temperature, humidity, wind_speed, pressure, precipitation, pm25
# Let's create additional features:
# - Rolling means of pm25 (3-day, 7-day)
# - Rolling means of weather features
# - Day of week, month, year
# - Interaction terms

print("\nEngineering features...")

# Sort by date
history_df = history_df.sort_values("date").reset_index(drop=True)

# Create rolling features
history_df["pm25_rolling_3"] = history_df["pm25"].rolling(window=3, min_periods=1).mean()
history_df["pm25_rolling_7"] = history_df["pm25"].rolling(window=7, min_periods=1).mean()
history_df["temp_rolling_3"] = history_df["temperature"].rolling(window=3, min_periods=1).mean()
history_df["humidity_rolling_3"] = history_df["humidity"].rolling(window=3, min_periods=1).mean()

# Create datetime features
history_df["day_of_week"] = history_df["date"].dt.dayofweek
history_df["day_of_month"] = history_df["date"].dt.day
history_df["month"] = history_df["date"].dt.month
history_df["year"] = history_df["date"].dt.year

# Interaction terms
history_df["temp_humidity"] = history_df["temperature"] * history_df["humidity"]
history_df["wind_pressure"] = history_df["wind_speed"] * history_df["pressure"]

# Lag features beyond pm25_lag1
history_df["pm25_lag2"] = history_df["pm25"].shift(1)
history_df["pm25_lag3"] = history_df["pm25"].shift(2)

# Fill NaN values from rolling/lag features
history_df.fillna(method="bfill", inplace=True)
history_df.fillna(method="ffill", inplace=True)

print(f"Engineered history shape: {history_df.shape}")
print(f"Engineered columns: {history_df.columns.tolist()}")

# Create feature group
fg_name = "airq2ce555"
print(f"\nCreating feature group: {fg_name}")

# Check if feature group exists, if not create it
try:
    fg = fs.get_feature_group(fg_name)
    print(f"Feature group {fg_name} already exists")
except:
    fg = fs.create_feature_group(
        name=fg_name,
        version=1,
        description="Air quality and weather features for PM2.5 forecasting",
        primary_key=["date"],
        event_time="date"
    )
    print(f"Created feature group {fg_name}")

# Get the feature group
fg = fs.get_feature_group(fg_name, version=1)

# Prepare data for feature store - select features (exclude pm25 which is the target)
feature_columns = [
    "date", "pm25_lag1", "temperature", "humidity", "wind_speed", 
    "pressure", "precipitation", "pm25_rolling_3", "pm25_rolling_7",
    "temp_rolling_3", "humidity_rolling_3", "day_of_week", "day_of_month",
    "month", "year", "temp_humidity", "wind_pressure", "pm25_lag2", "pm25_lag3"
]

# For the feature group, we need to handle the date column properly
# Convert date to string for feature store
history_features = history_df[feature_columns].copy()
history_features["date"] = history_features["date"].astype(str)

print(f"\nInserting {len(history_features)} rows into feature group...")
fg.insert(history_features, write_options={"wait_for_job": True})

print("Feature group populated successfully")

# Create training dataset
print("\nCreating training dataset...")
dataset_name = "airqtd2ce555"

# Get all features from the feature group
query = fg.select_all()

# Create or get the training dataset
try:
    td = fs.get_training_dataset(dataset_name)
    print(f"Training dataset {dataset_name} already exists")
except:
    # Create training dataset from the feature group
    td = fs.create_training_dataset(
        name=dataset_name,
        query=query,
        data_format="csv"
    )
    print(f"Created training dataset {dataset_name}")

# Wait for the training dataset to be ready
td = fs.get_training_dataset(dataset_name)
print(f"Training dataset status: {td.job_status()}")

# Now train a model
print("\nTraining model...")
import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Get the training data from the dataset
td_path = td.get_path()
print(f"Training dataset path: {td_path}")

# Read the training dataset
training_df = pd.read_csv(td_path)
print(f"Training data shape: {training_df.shape}")
print(f"Training data columns: {training_df.columns.tolist()}")

# We need to merge with the original history to get pm25 target
# The training dataset has the features but we need the target
# Let's use the original history_df which has both features and target

# Prepare training data from history_df
X = history_df[feature_columns].copy()
y = history_df["pm25"].values

# Convert date to numeric for sklearn (we'll keep it as string or extract numeric features)
# Actually, let's use only numeric features for training
numeric_features = [
    "pm25_lag1", "temperature", "humidity", "wind_speed", 
    "pressure", "precipitation", "pm25_rolling_3", "pm25_rolling_7",
    "temp_rolling_3", "humidity_rolling_3", "day_of_week", "day_of_month",
    "month", "year", "temp_humidity", "wind_pressure", "pm25_lag2", "pm25_lag3"
]

X_train = history_df[numeric_features].values

# Split data
X_train_split, X_val, y_train_split, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42, shuffle=False
)

print(f"Train split: {X_train_split.shape}, Val: {X_val.shape}")

# Create and train model
model = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train_split, y_train_split)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"\nValidation RMSE: {rmse:.4f}")
print(f"Validation MAE: {mae:.4f}")
print(f"Validation R2: {r2:.4f}")

# Train on full data
model.fit(X_train, y)

# Save model
model_name = "airqmodel2ce555"
print(f"\nRegistering model: {model_name}")

mr = hopsworks.model_registry()

# Save model locally first
model_dir = "model"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/model.joblib")

# Register the model
model_metadata = {
    "name": model_name,
    "description": "Gradient Boosting Regressor for PM2.5 forecasting",
    "metrics": {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    },
    "framework": "scikit-learn",
    "input_example": X_train[0:1].tolist(),
    "target": "pm25"
}

try:
    model_reg = mr.get_model(model_name)
    print(f"Model {model_name} already exists, updating...")
    # Update the model
    mr.update_model(
        name=model_name,
        model_dir=model_dir,
        description=model_metadata["description"],
        metrics=model_metadata["metrics"]
    )
except:
    # Create new model
    mr.save_model(
        model_dir=model_dir,
        model_name=model_name,
        description=model_metadata["description"],
        metrics=model_metadata["metrics"]
    )

print(f"Model {model_name} registered successfully")

# Now predict on forecast_days.csv
print("\nPredicting on forecast days...")

# Prepare forecast data
forecast_features = forecast_df[numeric_features].copy()

# Make predictions
forecast_predictions = model.predict(forecast_features)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    "date": forecast_df["date"].astype(str),
    "pm25_pred": forecast_predictions
})

print(f"Predictions shape: {predictions_df.shape}")
print(f"Predictions head:\n{predictions_df.head()}")

# Create feature table for predictions
pred_table_name = "airqpred2ce555"
print(f"\nCreating predictions feature table: {pred_table_name}")

# Check if feature group exists for predictions
try:
    pred_fg = fs.get_feature_group(pred_table_name)
    print(f"Predictions table {pred_table_name} already exists")
except:
    pred_fg = fs.create_feature_group(
        name=pred_table_name,
        version=1,
        description="PM2.5 predictions for forecast days",
        primary_key=["date"],
        event_time="date"
    )
    print(f"Created predictions feature group {pred_table_name}")

# Get the feature group
pred_fg = fs.get_feature_group(pred_table_name, version=1)

# Insert predictions
print(f"Inserting {len(predictions_df)} predictions...")
pred_fg.insert(predictions_df, write_options={"wait_for_job": True})

print("\nPipeline complete!")
print(f"- Feature group: {fg_name}")
print(f"- Training dataset: {dataset_name}")
print(f"- Model: {model_name} (RMSE: {rmse:.4f})")
print(f"- Predictions table: {pred_table_name}")
