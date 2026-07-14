#!/usr/bin/env python3
"""
Full FTI pipeline for air-quality PM2.5 forecasting (SQL + AutoML version).

Steps:
1. Create feature group `airqfcd91b` from `airquality_history.csv` (SQL).
2. Assemble training dataset `airqtdfcd91b` (SQL).
3. Train and register a regressor `airqmodelfcd91b` with metrics using Databricks AutoML.
4. Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b` (remote job).
5. Enable low-latency lookup for the predictions table.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, ml

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize Databricks client
w = WorkspaceClient()

# --- Step 1: Create Feature Group (`airqfcd91b`) ---
def create_feature_group():
    """Create a feature group in Unity Catalog (SQL)."""
    feature_group_name = f"{PREFIX}_airqfcd91b"
    
    # Create the feature table using SQL
    w.statement_execution.execute_statement(
        warehouse_id="<WAREHOUSE_ID>",  # Replace with a valid warehouse ID
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {feature_group_name} (
            date DATE,
            pm25_lag1 FLOAT,
            temperature FLOAT,
            humidity FLOAT,
            wind_speed FLOAT,
            pressure FLOAT,
            precipitation FLOAT,
            pm25 FLOAT
        )
        """,
    )
    print(f"Created feature group: {feature_group_name}")
    
    # Ingest data into the feature group using SQL
    w.statement_execution.execute_statement(
        warehouse_id="<WAREHOUSE_ID>",  # Replace with a valid warehouse ID
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        COPY INTO {feature_group_name}
        FROM 'dbfs:/FileStore/data/airquality_history.csv'
        FILEFORMAT = CSV
        FORMAT_OPTIONS ('mergeSchema' = 'true', 'header' = 'true')
        """,
    )
    print(f"Ingested data into feature group: {feature_group_name}")

# --- Step 2: Assemble Training Dataset (`airqtdfcd91b`) ---
def assemble_training_dataset():
    """Assemble training dataset from the feature group (SQL)."""
    training_dataset_name = f"{PREFIX}_airqtdfcd91b"
    feature_group_name = f"{PREFIX}_airqfcd91b"
    
    # Create the training dataset table using SQL
    w.statement_execution.execute_statement(
        warehouse_id="<WAREHOUSE_ID>",  # Replace with a valid warehouse ID
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {training_dataset_name} AS
        SELECT * FROM {feature_group_name}
        """,
    )
    print(f"Created training dataset: {training_dataset_name}")

# --- Step 3: Train and Register Regressor (`airqmodelfcd91b`) ---
def train_and_register_model():
    """Train a regressor using Databricks AutoML and register it with MLflow."""
    model_name = f"{PREFIX}_airqmodelfcd91b"
    training_dataset_name = f"{CATALOG}.{SCHEMA_NAME}.{PREFIX}_airqtdfcd91b"
    
    # Run AutoML experiment
    experiment = w.automl.regressions.create(
        table_name=training_dataset_name,
        target_col="pm25",
        timeout_minutes=30,
        experiment_name=f"{PREFIX}_automl_pm25",
    )
    
    # Wait for AutoML to complete
    run = experiment.result()
    print(f"AutoML run completed: {run.run_id}")
    
    # Register the best model
    best_model = run.best_trial.model_path
    registered_model = w.model_registry.register_model(
        name=model_name,
        model_path=best_model,
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
    )
    print(f"Registered model: {registered_model.full_name}")

# --- Step 4: Predict `pm25` for `forecast_days.csv` ---
def predict_and_store():
    """Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b` (remote job)."""
    predictions_table_name = f"{PREFIX}_airqpredfcd91b"
    model_name = f"{PREFIX}_airqmodelfcd91b"
    
    # Define the batch inference job
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
                    jobs.Library(pypi=jobs.PythonPyPiLibrary(package="pandas")),
                ],
                spark_python_task=jobs.SparkPythonTask(
                    python_file="dbfs:/FileStore/batch_inference.py",
                    parameters=[
                        f"--model-name={CATALOG}.{SCHEMA_NAME}.{model_name}",
                        f"--input-path=data/forecast_days.csv",
                        f"--output-table={CATALOG}.{SCHEMA_NAME}.{predictions_table_name}",
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