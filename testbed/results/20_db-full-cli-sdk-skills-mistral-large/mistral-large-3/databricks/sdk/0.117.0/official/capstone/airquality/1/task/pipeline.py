#!/usr/bin/env python3
"""
Full FTI pipeline for air-quality PM2.5 forecasting:
1. Feature engineering into feature group `airqfcd91b`.
2. Assemble training dataset `airqtdfcd91b`.
3. Train and register regressor `airqmodelfcd91b`.
4. Predict `pm25` for forecast_days.csv into `airqpredfcd91b`.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.compute import DataSecurityMode, AwsAttributes
from databricks.sdk.service.jobs import JobTaskSettings, NotebookTask, JobEmailNotifications, JobSettings

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., "workspace.mlpabaf5f38"
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # e.g., "mlpabaf5f38"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Names for objects
FEATURE_GROUP_NAME = f"{PREFIX}_airqfcd91b"
TRAINING_DATASET_NAME = f"{PREFIX}_airqtdfcd91b"
MODEL_NAME = f"{PREFIX}_airqmodelfcd91b"
PREDICTION_TABLE_NAME = f"{PREFIX}_airqpredfcd91b"
ONLINE_TABLE_NAME = f"{PREFIX}_airqpredfcd91b_online"
TEMP_TABLE_NAME = f"{PREFIX}_forecast_temp"

# Cluster configuration
CLUSTER_NAME = f"{PREFIX}_cluster"
CLUSTER_ID = None

# Create a cluster for feature engineering, training, and inference
def create_cluster():
    global CLUSTER_ID
    clusters = list(w.clusters.list(name=CLUSTER_NAME))
    if clusters:
        CLUSTER_ID = clusters[0].cluster_id
        print(f"Using existing cluster: {CLUSTER_ID}")
        return
    
    cluster = w.clusters.create(
        cluster_name=CLUSTER_NAME,
        spark_version="14.3.x-scala2.12",
        node_type_id="i3.xlarge",
        num_workers=2,
        data_security_mode=DataSecurityMode.USER_ISOLATION,
        aws_attributes=AwsAttributes(availability="SPOT"),
        autotermination_minutes=30,
    ).result()
    CLUSTER_ID = cluster.cluster_id
    print(f"Created cluster: {CLUSTER_ID}")

# Feature Engineering: Delegate to remote notebook
def create_feature_group():
    notebook_path = f"/Users/{w.current_user.me().user_name}/{PREFIX}/feature_engineering"
    
    # Create notebook for feature engineering
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Feature Engineering for Air Quality Data

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Upload historical data to DBFS
dbutils.fs.cp("file:/dbfs/FileStore/airquality_history.csv", "file:/tmp/airquality_history.csv", True)

# Read historical data
history_df = spark.read.csv("/tmp/airquality_history.csv", header=True, inferSchema=True)

# Engineer features: rolling averages (3-day, 7-day) for weather variables and pm25_lag1
window_3d = Window.orderBy("date").rowsBetween(-2, 0)
window_7d = Window.orderBy("date").rowsBetween(-6, 0)

for col in ["temperature", "humidity", "wind_speed", "pressure", "precipitation", "pm25_lag1"]:
    history_df = history_df.withColumn(f"{{col}}_rolling_3d", F.avg(col).over(window_3d))
    history_df = history_df.withColumn(f"{{col}}_rolling_7d", F.avg(col).over(window_7d))

# Drop rows with NaN (created by rolling windows)
history_df = history_df.na.drop()

# Write to Delta table in Unity Catalog
history_df.write.saveAsTable(
    name="{SCHEMA}.{FEATURE_GROUP_NAME}",
    mode="overwrite",
)

print(f"Feature group created: {SCHEMA}.{FEATURE_GROUP_NAME}")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Feature engineering notebook created: {notebook_path}")
    
    # Upload historical data to DBFS
    dbfs_path = "/FileStore/airquality_history.csv"
    w.dbfs.upload(
        source_path="data/airquality_history.csv",
        target_path=dbfs_path,
        overwrite=True,
    )
    print(f"Uploaded historical data to DBFS: {dbfs_path}")
    
    # Create job
    job_name = f"{PREFIX}_feature_engineering"
    job = w.jobs.create(
        name=job_name,
        tasks=[
            JobTaskSettings(
                task_key="feature_engineering",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                ),
                existing_cluster_id=CLUSTER_ID,
            )
        ],
        email_notifications=JobEmailNotifications(
            on_success=[w.current_user.me().user_name],
            on_failure=[w.current_user.me().user_name],
        ),
    )
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"Feature engineering job started: {run.run_id}")

# Assemble training dataset
def create_training_dataset():
    notebook_path = f"/Users/{w.current_user.me().user_name}/{PREFIX}/create_training_dataset"
    
    # Create notebook
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Assemble Training Dataset

# COMMAND ----------

# Read feature group
train_df = spark.table("{SCHEMA}.{FEATURE_GROUP_NAME}")

# Write training dataset
train_df.write.saveAsTable(
    name="{SCHEMA}.{TRAINING_DATASET_NAME}",
    mode="overwrite",
)

print(f"Training dataset created: {SCHEMA}.{TRAINING_DATASET_NAME}")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Training dataset notebook created: {notebook_path}")
    
    # Create job
    job_name = f"{PREFIX}_create_training_dataset"
    job = w.jobs.create(
        name=job_name,
        tasks=[
            JobTaskSettings(
                task_key="create_training_dataset",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                ),
                existing_cluster_id=CLUSTER_ID,
            )
        ],
        email_notifications=JobEmailNotifications(
            on_success=[w.current_user.me().user_name],
            on_failure=[w.current_user.me().user_name],
        ),
    )
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"Training dataset job started: {run.run_id}")

# Train and register model
def train_model():
    notebook_path = f"/Users/{w.current_user.me().user_name}/{PREFIX}/train_model"
    
    # Create notebook
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Train PM2.5 Regressor

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import pandas as pd
import numpy as np

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Read training dataset
train_df = spark.table("{SCHEMA}.{TRAINING_DATASET_NAME}").toPandas()

# Split into features and target
X = train_df.drop(columns=["date", "pm25"])
y = train_df["pm25"]

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = xgb.XGBRegressor(
    objective="reg:squarederror",
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
)
model.fit(X_train, y_train)

# Predict on validation set
y_pred = model.predict(X_val)

# Calculate metrics
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
mae = mean_absolute_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print(f"RMSE: {{rmse}}")
print(f"MAE: {{mae}}")
print(f"R²: {{r2}}")

# Log model and metrics with MLflow
with mlflow.start_run():
    mlflow.xgboost.log_model(model, "model")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    
    # Register model
    mlflow.register_model(
        f"runs:/{{mlflow.active_run().info.run_id}}/model",
        "{SCHEMA}.{MODEL_NAME}",
    )

print(f"Model trained and registered: {SCHEMA}.{MODEL_NAME}")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Training notebook created: {notebook_path}")
    
    # Create job
    job_name = f"{PREFIX}_train_model"
    job = w.jobs.create(
        name=job_name,
        tasks=[
            JobTaskSettings(
                task_key="train_model",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                ),
                existing_cluster_id=CLUSTER_ID,
            )
        ],
        email_notifications=JobEmailNotifications(
            on_success=[w.current_user.me().user_name],
            on_failure=[w.current_user.me().user_name],
        ),
    )
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"Training job started: {run.run_id}")

# Predict for forecast_days.csv and write to feature table
def predict_forecast():
    # Upload forecast data to DBFS
    dbfs_path = f"/FileStore/{PREFIX}_forecast_days.csv"
    w.dbfs.upload(
        source_path="data/forecast_days.csv",
        target_path=dbfs_path,
        overwrite=True,
    )
    print(f"Uploaded forecast data to DBFS: {dbfs_path}")
    
    notebook_path = f"/Users/{w.current_user.me().user_name}/{PREFIX}/predict_forecast"
    
    # Create notebook
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Predict PM2.5 for Forecast Days

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from databricks import mlflow
import pandas as pd

# Initialize Feature Engineering Client
fe = FeatureEngineeringClient()

# Load model
model_uri = f"models:/{SCHEMA}.{MODEL_NAME}/latest"
model = mlflow.xgboost.load_model(model_uri)

# Read forecast data
forecast_df = spark.read.csv("/dbfs{dbfs_path}", header=True, inferSchema=True).toPandas()

# Predict
X_forecast = forecast_df.drop(columns=["date"])
forecast_df["pm25_pred"] = model.predict(X_forecast)

# Write predictions to feature table
predictions_df = spark.createDataFrame(forecast_df[["date", "pm25_pred"]])
predictions_df.write.saveAsTable(
    name="{SCHEMA}.{PREDICTION_TABLE_NAME}",
    mode="overwrite",
)

# Enable online table for low-latency lookup
fe.create_table(
    name="{SCHEMA}.{ONLINE_TABLE_NAME}",
    primary_keys=["date"],
    df=predictions_df,
    description="Online table for PM2.5 predictions",
)

print(f"Predictions written to: {SCHEMA}.{PREDICTION_TABLE_NAME}")
print(f"Online table created: {SCHEMA}.{ONLINE_TABLE_NAME}")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Prediction notebook created: {notebook_path}")
    
    # Create job
    job_name = f"{PREFIX}_predict_forecast"
    job = w.jobs.create(
        name=job_name,
        tasks=[
            JobTaskSettings(
                task_key="predict_forecast",
                notebook_task=NotebookTask(
                    notebook_path=notebook_path,
                ),
                existing_cluster_id=CLUSTER_ID,
            )
        ],
        email_notifications=JobEmailNotifications(
            on_success=[w.current_user.me().user_name],
            on_failure=[w.current_user.me().user_name],
        ),
    )
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"Prediction job started: {run.run_id}")

# Main execution
if __name__ == "__main__":
    create_cluster()
    create_feature_group()
    create_training_dataset()
    train_model()
    predict_forecast()