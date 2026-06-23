#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

# Read training data from HopsFS - use the original history CSV which has pm25
# In a job, we need to use the full HopsFS path
train_df = pd.read_csv("/hopsfs/Resources/airquality_history.csv")
print(f"Training data shape: {train_df.shape}")
print(f"Training data columns: {train_df.columns.tolist()}")

# Separate features and label - exclude date from features
X = train_df.drop(columns=["pm25", "date"])
feature_columns = X.columns.tolist()
y = train_df["pm25"]
print(f"Features (excluding date): {feature_columns}")

print(f"Features: {feature_columns}")
print(f"Label shape: {y.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Save model to HopsFS
model_dir = "/hopsfs/Models/airqmodel2ce555"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
print(f"Model saved to {model_dir}")

# Now handle predictions
# Read forecast data
forecast_df = pd.read_csv("/hopsfs/Resources/forecast_days.csv")
print(f"Forecast data shape: {forecast_df.shape}")

# Make predictions
X_forecast = forecast_df[feature_columns]
forecast_predictions = model.predict(X_forecast)

# Create predictions dataframe
predictions_df = pd.DataFrame({
    "date": forecast_df["date"].astype(str),
    "pm25_pred": forecast_predictions
})

print(f"Predictions shape: {predictions_df.shape}")
print(f"Sample predictions:\n{predictions_df.head()}")

# Save predictions to CSV in HopsFS
predictions_df.to_csv("/hopsfs/Resources/predictions_airq.csv", index=False)
print(f"Predictions saved to /hopsfs/Resources/predictions_airq.csv")

print("Training and prediction completed!")
