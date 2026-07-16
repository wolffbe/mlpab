#!/usr/bin/env python3
"""
Fraud Detection Pipeline using Databricks SDK.

Steps:
1. Create or update feature group `cctxne0b071` with fraud features.
2. Assemble training dataset `cctde0b071`.
3. Train and register classifier `ccmodele0b071`.
4. Score `score_transactions.csv` into `ccprede0b071` with `fraud_probability`.
5. Enable low-latency lookup for predictions.
"""

import os
import uuid
from datetime import datetime, timedelta

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, ml, sql
from databricks.sdk.service.compute import DataSecurityMode, AwsAvailability
from databricks.sdk.service.jobs import JobCluster, JobSettings, SparkPythonTask, PythonWheelTask
from databricks.sdk.service.pipelines import PipelineClusterAutoscale, PipelineSettings
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedModelInput

# --- Config ---
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
USER = os.environ.get("DATABRICKS_USER", "unknown")

# Unity Catalog objects
FEATURE_GROUP_NAME = "cctxne0b071"
TRAINING_DATASET_NAME = "cctde0b071"
MODEL_NAME = "ccmodele0b071"
PREDICTIONS_TABLE_NAME = "ccprede0b071"

# Prefixed names for jobs, pipelines, endpoints
FEATURE_PIPELINE_NAME = f"{PREFIX}_feature_pipeline"
TRAINING_JOB_NAME = f"{PREFIX}_training_job"
SCORING_JOB_NAME = f"{PREFIX}_scoring_job"
SERVING_ENDPOINT_NAME = f"{PREFIX}_serving_endpoint"

# --- SDK Client ---
w = WorkspaceClient()

# --- Helper: Wait for job completion ---
def wait_for_job(job_id: str) -> None:
    """Wait for a job to complete and raise if it fails."""
    run = w.jobs.run_now(job_id).result()
    if run.state.result_state != jobs.RunResultState.SUCCESS:
        raise RuntimeError(f"Job {job_id} failed: {run.state}")


# --- 1. Feature Engineering ---
def create_feature_group() -> None:
    """Create or update the feature group with fraud-specific features."""
    # Define the feature table
    feature_table = catalog.TableInfo(
        name=f"{SCHEMA}.{FEATURE_GROUP_NAME}",
        catalog_name=SCHEMA.split(".")[0],
        schema_name=SCHEMA.split(".")[1],
        table_type=catalog.TableType.FEATURE_TABLE,
        data_source_format=catalog.DataSourceFormat.DELTA,
        columns=[
            catalog.ColumnInfo(name="transaction_id", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="cc_num", type_name=catalog.ColumnTypeName.LONG, nullable=False),
            catalog.ColumnInfo(name="datetime", type_name=catalog.ColumnTypeName.TIMESTAMP, nullable=False),
            catalog.ColumnInfo(name="amount", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="merchant", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="category", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="lat", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="long", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="is_fraud", type_name=catalog.ColumnTypeName.INT, nullable=False),
            # Engineered features
            catalog.ColumnInfo(name="txn_velocity_1h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="txn_velocity_24h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_velocity_1h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_velocity_24h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="geo_distance_km", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_zscore", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
        ],
        primary_key_columns=["transaction_id"],
    )

    # Create or update the feature table
    try:
        w.catalog.create_table(feature_table)
        print(f"Created feature table {SCHEMA}.{FEATURE_GROUP_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Feature table {SCHEMA}.{FEATURE_GROUP_NAME} already exists")
        else:
            raise e

    # Create a DLT pipeline to populate the feature table
    pipeline_settings = PipelineSettings(
        name=FEATURE_PIPELINE_NAME,
        storage=f"/Users/{USER}/{PREFIX}/feature_pipeline",
        configuration={
            "input.path": "/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/data/transactions.csv",
            "output.table": f"{SCHEMA}.{FEATURE_GROUP_NAME}",
        },
        clusters=[
            PipelineClusterAutoscale(
                label="default",
                num_workers=1,
                node_type_id="i3.xlarge",
                autoscale=PipelineClusterAutoscale(min_workers=1, max_workers=4),
            )
        ],
        libraries=[
            pipelines.PipelineLibrary(
                notebook=pipelines.NotebookLibrary(path=f"/Users/{USER}/{PREFIX}/feature_engineering"))
        ],
        continuous=False,
    )

    try:
        w.pipelines.create(pipeline_settings)
        print(f"Created DLT pipeline {FEATURE_PIPELINE_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"DLT pipeline {FEATURE_PIPELINE_NAME} already exists")
        else:
            raise e

    # Run the pipeline
    w.pipelines.start(pipeline_id=FEATURE_PIPELINE_NAME)
    print(f"Started DLT pipeline {FEATURE_PIPELINE_NAME}")


# --- 2. Training Dataset ---
def create_training_dataset() -> None:
    """Assemble the training dataset from the feature group."""
    # Create a table for the training dataset
    training_table = catalog.TableInfo(
        name=f"{SCHEMA}.{TRAINING_DATASET_NAME}",
        catalog_name=SCHEMA.split(".")[0],
        schema_name=SCHEMA.split(".")[1],
        table_type=catalog.TableType.MANAGED,
        data_source_format=catalog.DataSourceFormat.DELTA,
        columns=[
            catalog.ColumnInfo(name="transaction_id", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="cc_num", type_name=catalog.ColumnTypeName.LONG, nullable=False),
            catalog.ColumnInfo(name="datetime", type_name=catalog.ColumnTypeName.TIMESTAMP, nullable=False),
            catalog.ColumnInfo(name="amount", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="merchant", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="category", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="lat", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="long", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
            catalog.ColumnInfo(name="is_fraud", type_name=catalog.ColumnTypeName.INT, nullable=False),
            catalog.ColumnInfo(name="txn_velocity_1h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="txn_velocity_24h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_velocity_1h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_velocity_24h", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="geo_distance_km", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
            catalog.ColumnInfo(name="amount_zscore", type_name=catalog.ColumnTypeName.DOUBLE, nullable=True),
        ],
    )

    try:
        w.catalog.create_table(training_table)
        print(f"Created training table {SCHEMA}.{TRAINING_DATASET_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Training table {SCHEMA}.{TRAINING_DATASET_NAME} already exists")
        else:
            raise e

    # Create a job to populate the training dataset
    job_settings = JobSettings(
        name=TRAINING_JOB_NAME,
        tasks=[
            jobs.Task(
                task_key="assemble_training_data",
                new_cluster=JobCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=2,
                    data_security_mode=DataSecurityMode.USER_ISOLATION,
                    aws_attributes=jobs.AwsAttributes(availability=AwsAvailability.SPOT),
                ),
                spark_python_task=SparkPythonTask(
                    python_file=f"dbfs:/Users/{USER}/{PREFIX}/assemble_training_data.py",
                    parameters=[f"--schema={SCHEMA}", f"--feature-table={FEATURE_GROUP_NAME}", f"--output-table={TRAINING_DATASET_NAME}"],
                ),
            )
        ],
    )

    try:
        w.jobs.create(job_settings)
        print(f"Created training job {TRAINING_JOB_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Training job {TRAINING_JOB_NAME} already exists")
        else:
            raise e

    # Run the job
    wait_for_job(TRAINING_JOB_NAME)
    print(f"Training dataset {SCHEMA}.{TRAINING_DATASET_NAME} assembled")


# --- 3. Train and Register Classifier ---
def train_classifier() -> None:
    """Train a classifier and register it with metrics."""
    # Create a job to train the model
    job_settings = JobSettings(
        name=TRAINING_JOB_NAME,
        tasks=[
            jobs.Task(
                task_key="train_model",
                new_cluster=JobCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=2,
                    data_security_mode=DataSecurityMode.USER_ISOLATION,
                    aws_attributes=jobs.AwsAttributes(availability=AwsAvailability.SPOT),
                ),
                spark_python_task=SparkPythonTask(
                    python_file=f"dbfs:/Users/{USER}/{PREFIX}/train_model.py",
                    parameters=[
                        f"--schema={SCHEMA}",
                        f"--training-table={TRAINING_DATASET_NAME}",
                        f"--model-name={MODEL_NAME}",
                    ],
                ),
            )
        ],
    )

    try:
        w.jobs.create(job_settings)
        print(f"Created training job {TRAINING_JOB_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Training job {TRAINING_JOB_NAME} already exists")
        else:
            raise e

    # Run the job
    wait_for_job(TRAINING_JOB_NAME)
    print(f"Model {MODEL_NAME} trained and registered")


# --- 4. Score Unlabelled Data ---
def score_unlabelled_data() -> None:
    """Score `score_transactions.csv` and write results to `ccprede0b071`."""
    # Create a table for predictions
    predictions_table = catalog.TableInfo(
        name=f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}",
        catalog_name=SCHEMA.split(".")[0],
        schema_name=SCHEMA.split(".")[1],
        table_type=catalog.TableType.MANAGED,
        data_source_format=catalog.DataSourceFormat.DELTA,
        columns=[
            catalog.ColumnInfo(name="transaction_id", type_name=catalog.ColumnTypeName.STRING, nullable=False),
            catalog.ColumnInfo(name="fraud_probability", type_name=catalog.ColumnTypeName.DOUBLE, nullable=False),
        ],
        primary_key_columns=["transaction_id"],
    )

    try:
        w.catalog.create_table(predictions_table)
        print(f"Created predictions table {SCHEMA}.{PREDICTIONS_TABLE_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Predictions table {SCHEMA}.{PREDICTIONS_TABLE_NAME} already exists")
        else:
            raise e

    # Create a job to score the data
    job_settings = JobSettings(
        name=SCORING_JOB_NAME,
        tasks=[
            jobs.Task(
                task_key="score_data",
                new_cluster=JobCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="i3.xlarge",
                    num_workers=2,
                    data_security_mode=DataSecurityMode.USER_ISOLATION,
                    aws_attributes=jobs.AwsAttributes(availability=AwsAvailability.SPOT),
                ),
                spark_python_task=SparkPythonTask(
                    python_file=f"dbfs:/Users/{USER}/{PREFIX}/score_data.py",
                    parameters=[
                        f"--schema={SCHEMA}",
                        f"--model-name={MODEL_NAME}",
                        f"--input-path=/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/data/score_transactions.csv",
                        f"--output-table={PREDICTIONS_TABLE_NAME}",
                    ],
                ),
            )
        ],
    )

    try:
        w.jobs.create(job_settings)
        print(f"Created scoring job {SCORING_JOB_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Scoring job {SCORING_JOB_NAME} already exists")
        else:
            raise e

    # Run the job
    wait_for_job(SCORING_JOB_NAME)
    print(f"Predictions written to {SCHEMA}.{PREDICTIONS_TABLE_NAME}")


# --- 5. Enable Low-Latency Lookup ---
def enable_low_latency_lookup() -> None:
    """Enable low-latency lookup for the predictions table."""
    # Create an online table
    online_table = catalog.OnlineTable(
        name=f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}_online",
        primary_key_columns=["transaction_id"],
        source_table_full_name=f"{SCHEMA}.{PREDICTIONS_TABLE_NAME}",
        run_trigger=catalog.OnlineTableTrigger.CONTINUOUS,
    )

    try:
        w.catalog.create_online_table(online_table)
        print(f"Created online table for {SCHEMA}.{PREDICTIONS_TABLE_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Online table for {SCHEMA}.{PREDICTIONS_TABLE_NAME} already exists")
        else:
            raise e

    # Create a serving endpoint for the model
    endpoint_config = EndpointCoreConfigInput(
        name=SERVING_ENDPOINT_NAME,
        served_models=[
            ServedModelInput(
                model_name=MODEL_NAME,
                model_version="1",
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )

    try:
        w.serving_endpoints.create_and_wait(endpoint_config)
        print(f"Created serving endpoint {SERVING_ENDPOINT_NAME}")
    except Exception as e:
        if "already exists" in str(e):
            print(f"Serving endpoint {SERVING_ENDPOINT_NAME} already exists")
        else:
            raise e


# --- Main ---
def main() -> None:
    """Run the full pipeline."""
    print("Starting fraud detection pipeline...")
    
    # 1. Feature Engineering
    create_feature_group()
    
    # 2. Training Dataset
    create_training_dataset()
    
    # 3. Train and Register Classifier
    train_classifier()
    
    # 4. Score Unlabelled Data
    score_unlabelled_data()
    
    # 5. Enable Low-Latency Lookup
    enable_low_latency_lookup()
    
    print("Pipeline completed successfully.")


if __name__ == "__main__":
    main()