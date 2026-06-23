#!/usr/bin/env python3
"""
Job to train a PM2.5 regressor and register it as `airqmodel2408fa`.
"""

import hopsworks
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Load training dataset
td = fs.get_training_dataset("airqfv2408fa_1", version=1)
df = td.read()

# Features and target
X = df.drop(columns=["pm25", "date"])
y = df["pm25"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Save model
joblib.dump(model, "airqmodel2408fa.pkl")

# Register model
model_registry = mr.python.create_model(
    name="airqmodel2408fa",
    description="Random Forest regressor for PM2.5 forecasting",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    input_example=X_test.iloc[0:1].values,
    model_schema=None,
)

model_registry.save("airqmodel2408fa.pkl")