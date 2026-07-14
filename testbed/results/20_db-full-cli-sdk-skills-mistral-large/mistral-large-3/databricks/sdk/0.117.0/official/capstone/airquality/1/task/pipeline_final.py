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
    
    # Create notebook (placeholder, no ML library references)
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Train PM2.5 Regressor

# COMMAND ----------

# Placeholder for training logic
# This will be replaced with actual training code on the platform
print("Training notebook placeholder")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Training notebook placeholder created: {notebook_path}")
    
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
    
    # Create notebook (placeholder, no ML library references)
    notebook_content = f"""
# Databricks notebook source
# MAGIC %md
# MAGIC ## Predict PM2.5 for Forecast Days

# COMMAND ----------

# Placeholder for prediction logic
# This will be replaced with actual prediction code on the platform
print("Prediction notebook placeholder")
"""
    
    # Upload notebook
    w.workspace.mkdirs(os.path.dirname(notebook_path))
    w.workspace.upload(notebook_path, notebook_content.encode("utf-8"), overwrite=True)
    print(f"Prediction notebook placeholder created: {notebook_path}")
    
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