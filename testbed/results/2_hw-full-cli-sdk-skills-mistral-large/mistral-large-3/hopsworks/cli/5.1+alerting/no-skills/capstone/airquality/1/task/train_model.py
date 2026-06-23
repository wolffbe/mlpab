#!/usr/bin/env python3
"""
Train a PM2.5 regressor on the training dataset `airqtd2408fa` and register it as `airqmodel2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get the feature view and training dataset
fv = fs.get_feature_view("airqfv2408fa", version=1)
td = fv.get_training_data(1)[0]  # Get the first split

# Split into features and target
X = td.drop(columns=["pm25"])
y = td["pm25"]

# Train a regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Predict and evaluate
y_pred = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"RMSE: {rmse}")

# Save the model
mr = project.get_model_registry()
model_dir = "airqmodel2408fa"
joblib.dump(model, f"{model_dir}/model.pkl")

# Register the model
model_meta = mr.python.create_model(
    name="airqmodel2408fa",
    description="Random Forest regressor for PM2.5 forecasting",
    metrics={"rmse": rmse},
    model_schema=None,
    input_example=X.iloc[0:1].values,
    version=1
)

model_meta.save(model_dir)
print("Model trained and registered successfully.")