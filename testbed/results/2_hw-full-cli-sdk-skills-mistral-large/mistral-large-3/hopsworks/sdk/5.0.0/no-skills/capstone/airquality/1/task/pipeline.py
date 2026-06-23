#!/usr/bin/env python3

import hopsworks
import pandas as pd
import numpy as np
import os
from hsfs.feature import Feature
from hsfs.constructor import FeatureGroup

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Load data
history_df = pd.read_csv("data/airquality_history.csv", parse_dates=["date"])
forecast_df = pd.read_csv("data/forecast_days.csv", parse_dates=["date"])

# Feature Engineering: Create rolling averages and additional lag features
history_df = history_df.sort_values("date")
history_df["pm25_lag2"] = history_df["pm25"].shift(2)
history_df["pm25_lag3"] = history_df["pm25"].shift(3)
history_df["pm25_rolling_3"] = history_df["pm25"].rolling(3).mean()
history_df["pm25_rolling_7"] = history_df["pm25"].rolling(7).mean()
history_df = history_df.dropna()

# Create Feature Group
feature_group = fs.create_feature_group(
    name="airq2408fa",
    version=1,
    description="Air quality features including lag and rolling averages",
    primary_key=["date"],
    event_time="date",
    online_enabled=True,
)

# Ingest data into Feature Group
feature_group.insert(history_df, write_options={"wait_for_job": True})

# Create Training Dataset
query = feature_group.select_all()
td = fs.create_training_dataset(
    name="airqtd2408fa",
    version=1,
    description="Training dataset for PM2.5 prediction",
    data_format="csv",
    splits={"train": 0.8, "test": 0.2},
    label="pm25",
)

td.save(query)

# Split and upload training data to Hopsworks
td_export = td.read()
td_export["split"] = "train"
test_indices = td_export.sample(frac=0.2, random_state=42).index
td_export.loc[test_indices, "split"] = "test"

td_export[td_export["split"] == "train"].drop(columns=["split"]).to_csv("airqtd2408fa_train.csv", index=False)
td_export[td_export["split"] == "test"].drop(columns=["split"]).to_csv("airqtd2408fa_test.csv", index=False)

dataset_api = project.get_dataset_api()
dataset_api.upload("airqtd2408fa_train.csv", "Resources", overwrite=True)
dataset_api.upload("airqtd2408fa_test.csv", "Resources", overwrite=True)

# Define and upload the training script as a remote job
train_script = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import os

def train_model():
    # Load data
    train_df = pd.read_csv("Resources/airqtd2408fa_train.csv")
    test_df = pd.read_csv("Resources/airqtd2408fa_test.csv")
    
    X_train = train_df.drop(columns=["pm25", "date"])
    y_train = train_df["pm25"]
    X_test = test_df.drop(columns=["pm25", "date"])
    y_test = test_df["pm25"]
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"RMSE: {rmse}")
    
    # Save model
    joblib.dump(model, "model.pkl")
    
    # Save metrics
    with open("metrics.txt", "w") as f:
        f.write(f"rmse:{rmse}")

if __name__ == "__main__":
    train_model()
"""

# Upload the training script
dataset_api.upload("train_model.py", "Resources/train_model.py", overwrite=True, data=train_script)

# Create and run the training job
job_api = project.get_jobs_api()
job_config = {
    "name": "train_airqmodel2408fa",
    "job_type": "PYTHON",
    "python_version": "3.8",
    "script": "Resources/train_model.py",
    "resource_config": {
        "memory_mib": 2048,
        "vcores": 1
    },
    "args": []
}

job = job_api.create_job(job_config)
job.run(wait=True)

# Register model
mr = project.get_model_registry()
model_dir = "airqmodel2408fa"
os.makedirs(model_dir, exist_ok=True)

# Download the trained model and metrics
dataset_api.download("Resources/model.pkl", f"{model_dir}/model.pkl", overwrite=True)
dataset_api.download("Resources/metrics.txt", f"{model_dir}/metrics.txt", overwrite=True)

# Read metrics
with open(f"{model_dir}/metrics.txt", "r") as f:
    metrics = f.read().strip().split(":")
    rmse = float(metrics[1])

model_meta = mr.python.create_model(
    name="airqmodel2408fa",
    model_schema={
        "input_schema": td_export.drop(columns=["pm25", "date", "split"]).columns.tolist(),
        "output_schema": ["pm25_pred"],
    },
    description="RandomForestRegressor for PM2.5 prediction",
    metrics={"rmse": rmse},
)

model_meta.save(model_dir)

# Define and upload the prediction script
predict_script = """
import pandas as pd
import joblib
import os

def predict():
    # Load model
    model = joblib.load("Resources/model.pkl")
    
    # Load forecast data
    forecast_df = pd.read_csv("Resources/forecast_days.csv", parse_dates=["date"])
    history_df = pd.read_csv("Resources/airquality_history.csv", parse_dates=["date"])
    
    # Fill missing features with mean values from history
    forecast_features = forecast_df.copy()
    for col in ["pm25_lag2", "pm25_lag3", "pm25_rolling_3", "pm25_rolling_7"]:
        forecast_features[col] = history_df[col].mean()
    
    # Predict
    forecast_features["pm25_pred"] = model.predict(forecast_features.drop(columns=["date"]))
    
    # Save predictions
    forecast_features[["date", "pm25_pred"]].to_csv("predictions.csv", index=False)

if __name__ == "__main__":
    predict()
"""

# Upload the prediction script
dataset_api.upload("predict.py", "Resources/predict.py", overwrite=True, data=predict_script)

# Upload forecast data and model
dataset_api.upload("data/forecast_days.csv", "Resources/forecast_days.csv", overwrite=True)
dataset_api.upload(f"{model_dir}/model.pkl", "Resources/model.pkl", overwrite=True)

# Create and run the prediction job
predict_job_config = {
    "name": "predict_airqpred2408fa",
    "job_type": "PYTHON",
    "python_version": "3.8",
    "script": "Resources/predict.py",
    "resource_config": {
        "memory_mib": 2048,
        "vcores": 1
    },
    "args": []
}

predict_job = job_api.create_job(predict_job_config)
predict_job.run(wait=True)

# Download predictions
dataset_api.download("Resources/predictions.csv", "predictions.csv", overwrite=True)

# Create predictions feature group
prediction_fg = fs.create_feature_group(
    name="airqpred2408fa",
    version=1,
    description="Predictions for PM2.5 on forecast days",
    primary_key=["date"],
    event_time="date",
    online_enabled=True,
)

# Ingest predictions
predictions_df = pd.read_csv("predictions.csv", parse_dates=["date"])
prediction_fg.insert(predictions_df, write_options={"wait_for_job": True})

print("Pipeline completed successfully.")