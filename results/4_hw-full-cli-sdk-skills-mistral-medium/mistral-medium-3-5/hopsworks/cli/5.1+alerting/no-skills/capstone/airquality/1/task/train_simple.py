#!/usr/bin/env python3
import sys
sys.path.insert(0, '/srv/hops/anaconda/envs/hopsworks_environment/lib/python3.12/site-packages')

import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

# Login
hopsworks.login()

# Get feature store
try:
    fs = hopsworks.feature_store
    print("Got feature store via hopsworks.feature_store")
except AttributeError:
    # Try alternative
    from hopsworks.core import feature_store_api
    fs = feature_store_api.FeatureStore()
    print("Got feature store via FeatureStore()")

# Get the feature view
try:
    fv = fs.get_feature_view("airq_fv_2ce555", version=1)
    print(f"Got feature view: {fv}")
except Exception as e:
    print(f"Error getting feature view: {e}")
    sys.exit(1)

# Get training data
try:
    X, y = fv.get_training_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")
except Exception as e:
    print(f"Error getting training data: {e}")
    sys.exit(1)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
print(f"Validation RMSE: {rmse:.4f}")

# Save model
model_dir = "Models/airqmodel2ce555"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
print(f"Model saved to {model_dir}")

# Register model
try:
    mr = hopsworks.model_registry
    print("Got model registry via hopsworks.model_registry")
except AttributeError:
    from hopsworks.core import model_registry_api
    mr = model_registry_api.Registry()
    print("Got model registry via Registry()")

try:
    registered_model = mr.register(
        name="airqmodel2ce555",
        path=model_dir,
        description="Random Forest regressor for PM2.5 forecasting",
        metrics={"rmse": float(rmse)},
        framework="scikit-learn"
    )
    print(f"Model registered: {registered_model}")
except Exception as e:
    print(f"Error registering model: {e}")
    import traceback
    traceback.print_exc()

# Now handle predictions
# Read forecast data
forecast_df = pd.read_csv("Resources/forecast_days.csv")
print(f"Forecast data shape: {forecast_df.shape}")

# Prepare features - need to match the feature view
feature_columns = [col for col in X.columns]
print(f"Feature columns: {feature_columns}")

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

# Create predictions feature group
try:
    pred_fg = fs.get_feature_group("airqpred2ce555", version=1)
    print("Predictions feature group already exists")
except:
    pred_fg = fs.create_feature_group(
        name="airqpred2ce555",
        version=1,
        primary_key=["date"],
        online_enabled=True,
        description="PM2.5 predictions for forecast days"
    )
    print("Created predictions feature group")

# Save schema for predictions
try:
    from hopsworks.feature_store.api.util import create_schema
    pred_schema = create_schema(date="string", pm25_pred="double")
    pred_fg.save(pred_schema)
except Exception as e:
    print(f"Error saving schema: {e}")

# Insert predictions
try:
    pred_fg.insert(predictions_df, write_options={"wait_for_job": True})
    print(f"Inserted {len(predictions_df)} predictions")
except Exception as e:
    print(f"Error inserting predictions: {e}")
    import traceback
    traceback.print_exc()

print("Pipeline completed!")
