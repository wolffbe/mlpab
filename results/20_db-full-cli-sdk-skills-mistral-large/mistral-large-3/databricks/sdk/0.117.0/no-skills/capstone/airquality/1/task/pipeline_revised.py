#!/usr/bin/env python3
"""
Full FTI pipeline for air-quality PM2.5 forecasting (revised for remote execution).

Steps:
1. Create feature group `airqfcd91b` from `airquality_history.csv`.
2. Assemble training dataset `airqtdfcd91b`.
3. Train and register a regressor `airqmodelfcd91b` with metrics (remote job).
4. Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b`.
5. Enable low-latency lookup for the predictions table.
"""

import os
import pandas as pd
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs
from databricks.sdk.service.catalog import TableType, ColumnTypeName
from databricks.sdk.service.serving import EndpointStateReady

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize Databricks client
w = WorkspaceClient()

# Read input data
history_df = pd.read_csv("data/airquality_history.csv")
forecast_df = pd.read_csv("data/forecast_days.csv")

# --- Step 1: Create Feature Group (`airqfcd91b`) ---
def create_feature_group():
    """Create a feature group in Unity Catalog."""
    feature_group_name = f"{PREFIX}_airqfcd91b"
    
    # Define schema for the feature group
    columns = [
        catalog.Column(name="date", type_name=ColumnTypeName.DATE, nullable=False, comment="Record date"),
        catalog.Column(name="pm25_lag1", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Previous day's PM2.5"),
        catalog.Column(name="temperature", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Temperature"),
        catalog.Column(name="humidity", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Humidity"),
        catalog.Column(name="wind_speed", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Wind speed"),
        catalog.Column(name="pressure", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Pressure"),
        catalog.Column(name="precipitation", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Precipitation"),
        catalog.Column(name="pm25", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Target: PM2.5"),
    ]
    
    # Create the feature table
    try:
        w.tables.create(
            name=feature_group_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=TableType.MANAGED,
            columns=columns,
            comment="Feature group for air-quality PM2.5 forecasting",
        )
        print(f"Created feature group: {feature_group_name}")
    except Exception as e:
        print(f"Feature group may already exist: {e}")
    
    # Ingest data into the feature group
    spark_df = w.create_dataframe(history_df)
    spark_df.write.save_as_table(
        name=f"{CATALOG}.{SCHEMA_NAME}.{feature_group_name}",
        mode="overwrite",
    )
    print(f"Ingested data into feature group: {feature_group_name}")

# --- Step 2: Assemble Training Dataset (`airqtdfcd91b`) ---
def assemble_training_dataset():
    """Assemble training dataset from the feature group."""
    training_dataset_name = f"{PREFIX}_airqtdfcd91b"
    
    # Create a Delta table for the training dataset
    try:
        w.tables.create(
            name=training_dataset_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=TableType.MANAGED,
            columns=[
                catalog.Column(name="date", type_name=ColumnTypeName.DATE, nullable=False),
                catalog.Column(name="pm25_lag1", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="temperature", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="humidity", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="wind_speed", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="pressure", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="precipitation", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="pm25", type_name=ColumnTypeName.FLOAT, nullable=False),
            ],
            comment="Training dataset for air-quality PM2.5 forecasting",
        )
        print(f"Created training dataset table: {training_dataset_name}")
    except Exception as e:
        print(f"Training dataset table may already exist: {e}")
    
    # Ingest data into the training dataset
    spark_df = w.create_dataframe(history_df)
    spark_df.write.save_as_table(
        name=f"{CATALOG}.{SCHEMA_NAME}.{training_dataset_name}",
        mode="overwrite",
    )
    print(f"Ingested data into training dataset: {training_dataset_name}")

# --- Step 3: Train and Register Regressor (`airqmodelfcd91b`) ---
def train_and_register_model():
    """Train a regressor using Databricks AutoML and register it with MLflow."""
    model_name = f"{PREFIX}_airqmodelfcd91b"
    training_dataset_name = f"{PREFIX}_airqtdfcd91b"
    
    # Define the AutoML job
    automl_job = jobs.CreateJob(
        name=f"{PREFIX}_automl_training",
        tasks=[
            jobs.Task(
                task_key="automl_task",
                description="Run AutoML for PM2.5 regression",
                new_cluster=jobs.NewCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=2,
                ),
                libraries=[
                    jobs.Library(pypi=jobs.PythonPyPiLibrary(package="mlflow")),
                ],
                spark_python_task=jobs.SparkPythonTask(
                    python_file="dbfs:/FileStore/automl_training.py",
                    parameters=[
                        f"--training-table={CATALOG}.{SCHEMA_NAME}.{training_dataset_name}",
                        f"--target-col=pm25",
                        f"--model-name={CATALOG}.{SCHEMA_NAME}.{model_name}",
                    ],
                ),
                timeout_seconds=3600,
            )
        ],
    )
    
    # Submit the job
    job = w.jobs.create(**automl_job.as_dict())
    print(f"Submitted AutoML job: {job.job_id}")
    
    # Wait for job completion
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"AutoML job completed: {run.state.result_state}")

# --- Step 4: Predict `pm25` for `forecast_days.csv` ---
def predict_and_store():
    """Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b`."""
    predictions_table_name = f"{PREFIX}_airqpredfcd91b"
    model_name = f"{PREFIX}_airqmodelfcd91b"
    
    # Create a job for batch inference
    inference_job = jobs.CreateJob(
        name=f"{PREFIX}_batch_inference",
        tasks=[
            jobs.Task(
                task_key="inference_task",
                description="Run batch inference for PM2.5 predictions",
                new_cluster=jobs.NewCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=2,
                ),
                libraries=[
                    jobs.Library(pypi=jobs.PythonPyPiLibrary(package="mlflow")),
                ],
                spark_python_task=jobs.SparkPythonTask(
                    python_file="dbfs:/FileStore/batch_inference.py",
                    parameters=[
                        f"--model-name={CATALOG}.{SCHEMA_NAME}.{model_name}",
                        f"--forecast-table={CATALOG}.{SCHEMA_NAME}.{predictions_table_name}",
                        f"--input-path=data/forecast_days.csv",
                    ],
                ),
                timeout_seconds=3600,
            )
        ],
    )
    
    # Submit the job
    job = w.jobs.create(**inference_job.as_dict())
    print(f"Submitted batch inference job: {job.job_id}")
    
    # Wait for job completion
    run = w.jobs.run_now(job_id=job.job_id).result()
    print(f"Batch inference job completed: {run.state.result_state}")
    
    # Enable low-latency lookup
    try:
        w.online_tables.create(
            name=predictions_table_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            primary_key_columns=["date"],
        )
        print(f"Enabled low-latency lookup for table: {predictions_table_name}")
    except Exception as e:
        print(f"Failed to enable low-latency lookup: {e}")

# --- Run Pipeline ---
if __name__ == "__main__":
    create_feature_group()
    assemble_training_dataset()
    train_and_register_model()
    predict_and_store()
    print("Pipeline completed successfully.")